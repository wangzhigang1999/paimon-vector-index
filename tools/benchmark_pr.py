#!/usr/bin/env python3
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

"""Compare two disposable checkouts using the candidate's ANN benchmark driver."""

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import time


INDEXES = ("IVF_FLAT", "IVF_SQ", "IVF_PQ", "IVF_RQ", "DISKANN")
WORKLOAD = {
    "ANN_DATASET_NAME": "ci-clustered-v1",
    "ANN_N": "10000",
    "ANN_TRAIN_N": "4096",
    "ANN_NQ": "2048",
    "ANN_D": "64",
    "ANN_K": "10",
    "ANN_NLIST": "64",
    "ANN_NPROBE": "8",
    "ANN_PQ_M": "8",
    "ANN_RQ_BITS": "4",
    "ANN_CLUSTERS": "32",
    "ANN_NOISE_DIMENSIONS": "64",
    "ANN_SEED": "42",
    "ANN_DISKANN_L_SEARCHES": "100",
    "ANN_DISKANN_MEMORY_BUDGET_BYTES": str(256 * 1024 * 1024),
    "ANN_DISKANN_BUILD_DISTANCE": "full_precision",
    "ANN_DISKANN_RAW_VECTOR_ENCODING": "f32",
    "ANN_STORAGE_CASES": "local_ssd_warm_cache",
    "RAYON_NUM_THREADS": "2",
}
# These identify the workload, not measured results. Never compare unlike cases.
CASE_FIELDS = (
    "dataset", "index", "storage", "n", "train_n", "nq", "d", "k", "nlist",
    "nprobe", "pq_m", "rq_bits", "diskann_build_distance",
    "diskann_raw_vector_encoding", "l_search",
)
METRICS = (
    ("recall_at_10", "Batch Recall@10", "recall"),
    ("sequential_qps", "Sequential QPS", "higher"),
    ("batch_qps", "Batch QPS", "higher"),
    ("p95_query_us", "Sequential P95 (µs)", "lower"),
    ("first_query_us", "First query (µs)", "lower"),
    ("sequential_pread_rounds", "Read rounds / sequential query", "lower"),
    ("sequential_pread_bytes", "Read bytes / sequential query", "lower"),
    ("batch_pread_rounds", "Read rounds / batch query", "lower"),
    ("batch_pread_bytes", "Read bytes / batch query", "lower"),
    ("build_ms", "Build (ms)", "lower"),
    ("peak_rss_bytes", "Process peak RSS through build (MiB)", "lower"),
    ("file_bytes", "Index size (MiB)", "lower"),
)


def command_output(command, cwd=None):
    return subprocess.check_output(command, cwd=cwd, text=True).strip()


def read_sample(path, index):
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    if len(rows) != 1:
        raise ValueError(f"{path}: expected exactly one case, found {len(rows)}")
    row = rows[0]
    if row.get("index") != index:
        raise ValueError(f"{path}: expected index {index}")
    if any(not row.get(field) for field in CASE_FIELDS):
        raise ValueError(f"{path}: missing workload fields")
    for field, _, _ in METRICS:
        value = float(row[field])
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{path}: invalid {field}: {value}")
    if not 0 <= float(row["recall_at_10"]) <= 1 or int(row["nq"]) <= 0:
        raise ValueError(f"{path}: invalid recall or query count")
    if float(row["sequential_qps"]) <= 0 or float(row["batch_qps"]) <= 0:
        raise ValueError(f"{path}: throughput must be positive")
    return row


def metric_value(row, field):
    value = float(row[field])
    if "pread_" in field:
        return value / int(row["nq"])
    if field in ("peak_rss_bytes", "file_bytes"):
        return value / (1024 * 1024)
    return value


def delta_text(base, candidate, direction):
    if direction == "recall":
        return f"{(candidate - base) * 100:+.2f} pp"
    if base == 0:
        return "0.0%" if candidate == 0 else "n/a (base=0)"
    return f"{(candidate / base - 1) * 100:+.1f}%"


def summarize(samples, rounds):
    results = {}
    for index in INDEXES:
        sides = samples[index]
        if any(len(sides[side]) != rounds for side in ("base", "candidate")):
            raise ValueError(f"{index}: incomplete samples")
        cases = {
            tuple(row[field] for field in CASE_FIELDS)
            for rows in sides.values() for row in rows
        }
        if len(cases) != 1:
            raise ValueError(f"{index}: workload differs between samples")
        result = {}
        for field, label, direction in METRICS:
            entry = {"label": label, "direction": direction}
            for side in ("base", "candidate"):
                values = [metric_value(row, field) for row in sides[side]]
                entry[side] = {
                    "median": statistics.median(values),
                    "min": min(values), "max": max(values), "samples": values,
                }
            entry["delta"] = delta_text(
                entry["base"]["median"], entry["candidate"]["median"], direction
            )
            result[field] = entry
        results[index] = result
    return results


def render_report(metadata, results):
    lines = [
        "# PR / base benchmark", "",
        f"- Base: `{metadata['base_sha']}`",
        f"- Candidate (merge result in PR CI): `{metadata['candidate_sha']}`",
        f"- Shared benchmark driver SHA-256: `{metadata['driver_sha256']}`",
        f"- Rust: `{metadata['rustc'].splitlines()[0]}`",
        f"- Runner: {metadata['platform']}; {metadata['cpu']}",
        f"- {metadata['rounds']} fresh processes per version per index; "
        "alternating base→candidate / candidate→base pairs; 2 Rayon threads.",
        "- Fixed synthetic L2 workload: 10,000 × 64D, 4,096 training vectors, "
        "2,048 queries, top-10, seed 42, nlist=64, nprobe=8, PQ m=8, DiskANN L=100.",
        "- Local warm page cache. Each process builds its own index. Reader optimization "
        "and one first query precede sequential queries; batch uses a separate optimized reader.",
        "- Values are medians [min, max]. ↑ means higher is better, ↓ means lower is better. "
        "Delta is PR/base − 1; recall delta is in percentage points (pp).",
        "- Timing changes are observations, not a merge gate "
        "or a statistical significance claim. Recall is measured on batch results.",
        "- Peak RSS includes dataset/ground truth and is sampled after build; "
        "it is not search peak memory. First query is not a cold-disk measurement.",
        "",
    ]
    lines += ["| Index | Recall base → PR | Sequential QPS Δ | Batch QPS Δ | P95 Δ | Build Δ |",
              "|---|---:|---:|---:|---:|---:|"]
    for index, metrics in results.items():
        recall = metrics["recall_at_10"]
        lines.append(
            f"| {index} | {recall['base']['median'] * 100:.2f}% → "
            f"{recall['candidate']['median'] * 100:.2f}% | "
            f"{metrics['sequential_qps']['delta']} | {metrics['batch_qps']['delta']} | "
            f"{metrics['p95_query_us']['delta']} | {metrics['build_ms']['delta']} |"
        )
    lines += [""]
    for index, metrics in results.items():
        lines += [f"<details><summary>{index}: measurements and sample ranges</summary>", ""]
        recall = metrics["recall_at_10"]
        if recall["candidate"]["median"] < recall["base"]["median"]:
            lines += ["**Recall decreased. Do not interpret faster queries as a quality-preserving improvement.**", ""]
        lines += ["| Metric | Base [min, max] | PR [min, max] | Δ PR/base |",
                  "|---|---:|---:|---:|"]
        for field, entry in metrics.items():
            values = []
            for side in ("base", "candidate"):
                summary = entry[side]
                scale = 100 if entry["direction"] == "recall" else 1
                unit = "%" if scale == 100 else ""
                precision = 4 if field.endswith("pread_rounds") else 2
                values.append(
                    f"{summary['median'] * scale:,.{precision}f}{unit} "
                    f"[{summary['min'] * scale:,.{precision}f}, {summary['max'] * scale:,.{precision}f}]"
                )
            arrow = "↑" if entry["direction"] in ("higher", "recall") else "↓"
            lines.append(f"| {entry['label']} {arrow} | {values[0]} | {values[1]} | {entry['delta']} |")
        lines += ["", "</details>", ""]
    lines += ["Raw CSVs, stderr logs, build logs, environment metadata and summary.json "
              "are available in the workflow artifact.", ""]
    return "\n".join(lines)


def build(checkout, output, side, env):
    # Separate target directories prevent artifacts from one revision leaking into the other.
    command = ["cargo", "bench", "--locked", "-p", "paimon-vindex-core", "--bench",
               "ann_bench", "--no-run", "--message-format=json", "--target-dir",
               str(output / "build" / side)]
    print(f"Building {side}", flush=True)
    messages_path = output / f"build-{side}.jsonl"
    with messages_path.open("w") as messages, (output / f"build-{side}.log").open("w") as log:
        subprocess.run(command, cwd=checkout, env=env, stdout=messages, stderr=log,
                       check=True, timeout=900)
    executables = []
    for line in messages_path.read_text().splitlines():
        message = json.loads(line)
        if (message.get("reason") == "compiler-artifact"
                and message.get("target", {}).get("name") == "ann_bench"
                and message.get("executable")):
            executables.append(message["executable"])
    if len(executables) != 1:
        raise ValueError(f"{side}: expected one benchmark executable")
    return executables[0]


def run(args):
    output = args.output.resolve()
    base, candidate = args.base.resolve(), args.candidate.resolve()
    if base == candidate:
        raise ValueError("base and candidate must be separate disposable checkouts")
    if command_output(["git", "status", "--porcelain"], base):
        raise ValueError("base checkout must be clean before copying the shared driver")
    metadata = {
        "base_sha": command_output(["git", "rev-parse", "HEAD"], base),
        "candidate_sha": command_output(["git", "rev-parse", "HEAD"], candidate),
        "rustc": command_output(["rustc", "--version", "--verbose"]),
        "platform": platform.platform(),
        "cpu": platform.processor(),
        "rounds": args.rounds, "workload": WORKLOAD,
        "candidate_dirty": bool(command_output(["git", "status", "--porcelain"], candidate)),
        "execution_order": [],
        "rustflags": "-C target-cpu=x86-64" if platform.machine() == "x86_64" else "",
    }
    if shutil.which("lscpu"):
        metadata["cpu"] = next((line.split(":", 1)[1].strip()
                                for line in command_output(["lscpu"]).splitlines()
                                if line.startswith("Model name:")), metadata["cpu"])
    # Use candidate driver on both versions, including its support module. API incompatibility
    # fails the run explicitly instead of silently comparing different benchmark programs.
    digest = hashlib.sha256()
    for relative in ("core/benches/ann_bench.rs", "core/benches/support/ann_bench_support.rs"):
        data = (candidate / relative).read_bytes()
        digest.update(relative.encode() + b"\0" + data)
        (base / relative).parent.mkdir(parents=True, exist_ok=True)
        (base / relative).write_bytes(data)
    metadata["driver_sha256"] = digest.hexdigest()
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    env = {key: value for key, value in os.environ.items()
           if not key.startswith(("ANN_", "DISKANN_BENCH_", "CARGO_PROFILE_"))
           and key not in ("RUSTFLAGS", "CARGO_ENCODED_RUSTFLAGS", "CARGO_BUILD_RUSTFLAGS")}
    env.update(WORKLOAD)
    env.update({"RUSTFLAGS": metadata["rustflags"], "LC_ALL": "C", "CARGO_INCREMENTAL": "0"})
    executables = {side: build(checkout, output, side, env)
                   for side, checkout in (("base", base), ("candidate", candidate))}
    samples = {index: {"base": [], "candidate": []} for index in INDEXES}
    raw = output / "raw"
    raw.mkdir()
    for index in INDEXES:
        for repetition in range(args.rounds):
            order = ("base", "candidate") if repetition % 2 == 0 else ("candidate", "base")
            for side in order:
                prefix = raw / f"{index}-{repetition + 1}-{side}"
                print(f"Running {prefix.name}", flush=True)
                sample_env = dict(env, ANN_INDEXES=index, ANN_OUTPUT_DIR=str(output / "indexes"))
                started = time.monotonic()
                with prefix.with_suffix(".csv").open("w") as csv_file, prefix.with_suffix(".log").open("w") as log:
                    subprocess.run([executables[side]], env=sample_env, cwd=output,
                                   stdout=csv_file, stderr=log, timeout=args.timeout, check=True)
                samples[index][side].append(read_sample(prefix.with_suffix(".csv"), index))
                metadata["execution_order"].append({
                    "index": index, "round": repetition + 1, "side": side,
                    "wall_seconds": round(time.monotonic() - started, 3),
                })
                (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = summarize(samples, args.rounds)
    (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "summary.md").write_text(render_report(metadata, results))
    print(f"Report: {output / 'summary.md'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True, help="Disposable base checkout (driver is overwritten)")
    parser.add_argument("--candidate", type=Path, required=True, help="Candidate checkout providing the shared driver")
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    parser.add_argument("--rounds", type=int, default=4, help="Samples per version/index; even number >= 2")
    parser.add_argument("--timeout", type=int, default=180, help="Timeout in seconds per sample")
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or args.timeout <= 0:
        parser.error("rounds must be even and >= 2; timeout must be positive")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    try:
        run(args)
    except Exception as error:
        (args.output / "summary.md").write_text(
            "# PR / base benchmark failed\n\n"
            "The comparison is incomplete; no performance conclusion is available. "
            "See the uploaded build/sample logs for details.\n\n"
            f"```text\n{error}\n```\n"
        )
        raise


if __name__ == "__main__":
    main()
