// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Compare this same benchmark binary source on the base and candidate commits.
//! Optional latency/bandwidth is simulated, never a claim about actual OSS.
use paimon_vindex_core::index::{
    VectorIndexConfig, VectorIndexReader, VectorIndexTrainer, VectorIndexWriter, VectorSearchParams,
};
use paimon_vindex_core::io::{PosWriter, ReadRequest, SeekRead};
use rand::{rngs::StdRng, Rng, SeedableRng};
use roaring::RoaringTreemap;
use std::collections::HashMap;
use std::hint::black_box;
use std::io::{self, Cursor};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

const D: usize = 64;
const NLIST: usize = 64;
const N: usize = 32_768;
const K: usize = 10;

#[derive(Default, Clone)]
struct Stats {
    calls: usize,
    ranges: usize,
    bytes: usize,
}
struct Input {
    inner: Cursor<Arc<[u8]>>,
    stats: Arc<Mutex<Stats>>,
    mib_per_sec: f64,
}
impl SeekRead for Input {
    fn pread(&mut self, ranges: &mut [ReadRequest<'_>]) -> io::Result<()> {
        let bytes = ranges.iter().map(|r| r.buf.len()).sum::<usize>();
        if self.mib_per_sec > 0.0 {
            std::thread::sleep(Duration::from_secs_f64(
                0.002 + bytes as f64 / (self.mib_per_sec * 1024.0 * 1024.0),
            ));
        }
        let mut stats = self.stats.lock().unwrap();
        stats.calls += 1;
        stats.ranges += ranges.len();
        stats.bytes += bytes;
        self.inner.pread(ranges)
    }
}
fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .map(|v| v.parse().unwrap())
        .unwrap_or(default)
}
fn fingerprint(ids: &[i64], distances: &[f32]) -> u64 {
    ids.iter()
        .zip(distances)
        .fold(0xcbf29ce484222325, |hash, (id, distance)| {
            (hash ^ *id as u64).wrapping_mul(0x100000001b3) ^ distance.to_bits() as u64
        })
}
fn main() {
    let rounds = env_usize("VINDEX_BENCH_ROUNDS", 9);
    let nq = env_usize("VINDEX_BENCH_QUERIES", 16);
    let metric = std::env::var("VINDEX_BENCH_METRIC").unwrap_or_else(|_| "l2".into());
    let dump_results = std::env::var_os("VINDEX_BENCH_DUMP_RESULTS").is_some();
    let mib_per_sec = std::env::var("VINDEX_BENCH_IO_MIB_PER_SEC")
        .map(|v| v.parse::<f64>().unwrap())
        .unwrap_or(0.0);
    let mut rng = StdRng::seed_from_u64(42);
    let data = (0..N)
        .flat_map(|row| {
            (0..D)
                .map(|col| {
                    (row % NLIST) as f32 * 0.1
                        + col as f32 * 0.001
                        + rng.gen_range(-0.002f32..0.002)
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let ids = (0..N as i64).collect::<Vec<_>>();
    // Queries stay near the first cluster. Later rows have different noise.
    let queries = (0..nq)
        .flat_map(|i| {
            let row = (i * NLIST) % N;
            data[row * D..(row + 1) * D].to_vec()
        })
        .collect::<Vec<_>>();
    eprintln!("n={N} d={D} nlist={NLIST} k={K} nq={nq} rounds={rounds} threads={} simulated_mib_per_sec={mib_per_sec}", rayon::current_num_threads());
    println!("index,case,round,us_per_query,calls_per_query,ranges_per_query,bytes_per_query,result_hash,valid_per_query");
    for kind in ["ivf_flat", "ivf_sq", "ivf_pq", "ivf_rq"] {
        let options = HashMap::from([
            ("index.type".into(), kind.into()),
            ("dimension".into(), D.to_string()),
            ("nlist".into(), NLIST.to_string()),
            ("metric".into(), metric.clone()),
        ]);
        let training =
            VectorIndexTrainer::train(VectorIndexConfig::from_options(&options).unwrap(), &data, N)
                .unwrap();
        let mut writer = VectorIndexWriter::new(training);
        writer.add_vectors(&ids, &data, N).unwrap();
        let mut payload = Vec::new();
        writer.write(&mut PosWriter::new(&mut payload)).unwrap();
        drop(writer);
        let stats = Arc::new(Mutex::new(Stats::default()));
        let mut reader = VectorIndexReader::open(Input {
            inner: Cursor::new(Arc::<[u8]>::from(payload)),
            stats: stats.clone(),
            mib_per_sec,
        })
        .unwrap();
        reader.optimize_for_search().unwrap();
        for case in ["expand", "no_expand", "fixed", "unfiltered"] {
            let mut filter = RoaringTreemap::new();
            for row in 0..N {
                let selected = if case == "expand" {
                    row == 0 || row % NLIST >= NLIST / 2
                } else {
                    row % NLIST < NLIST / 2
                };
                if selected {
                    filter.insert(row as u64);
                }
            }
            let mut filter_bytes = Vec::new();
            filter.serialize_into(&mut filter_bytes).unwrap();
            let params = if case == "fixed" {
                VectorSearchParams::new(K, 16)
            } else {
                VectorSearchParams::automatic(K)
            };
            let mut expected = None;
            // Round zero warms the exact query path and is excluded from output.
            for round in 0..=rounds {
                *stats.lock().unwrap() = Stats::default();
                let start = Instant::now();
                let mut hash = 0u64;
                let mut valid = 0usize;
                for (query_index, query) in queries.chunks(D).enumerate() {
                    let (ids, distances) = if case == "unfiltered" {
                        reader.search(query, params)
                    } else {
                        reader.search_with_roaring_filter(query, params, &filter_bytes)
                    }
                    .unwrap();
                    if round == 0 && dump_results {
                        println!("result,{kind},{case},{query_index},{ids:?},{distances:?}");
                    }
                    let count = distances.iter().filter(|&&d| d != f32::MAX).count();
                    // IVF-PQ's existing short-result behavior is outside this change.
                    if kind != "ivf_pq" {
                        assert_eq!(count, K);
                    }
                    valid += count;
                    hash = hash.rotate_left(7) ^ fingerprint(&ids, &distances);
                    black_box((&ids, &distances));
                }
                let elapsed = start.elapsed().as_secs_f64() * 1e6 / nq as f64;
                if let Some(expected) = expected {
                    assert_eq!(hash, expected);
                }
                expected = Some(hash);
                if round > 0 {
                    let s = stats.lock().unwrap();
                    println!(
                        "{kind},{case},{round},{elapsed:.3},{:.3},{:.3},{:.3},{hash:016x},{:.3}",
                        s.calls as f64 / nq as f64,
                        s.ranges as f64 / nq as f64,
                        s.bytes as f64 / nq as f64,
                        valid as f64 / nq as f64
                    );
                }
            }
        }
    }
}
