#!/usr/bin/env python3
"""
Generate 4 user-partitioned 768D vector embedding datasets from Cohere 1M source data.

Source: Cohere/wikipedia-22-12 768D float vectors (1M rows from VectorDBBench)

Datasets:
  - dataset_768d_1w:   100 users x 10,000 vectors   = 1M total
  - dataset_768d_10w:  100 users x 100,000 vectors   = 10M total
  - dataset_768d_50w:  100 users x 500,000 vectors   = 50M total
  - dataset_768d_100w: 100 users x 1,000,000 vectors = 100M total

Output format: TSV (tab-separated), gzip compressed, sharded by ~1M rows.
Columns: user_id<TAB>id<TAB>[v1,v2,...,v768]

Usage:
  python3 generate_datasets_768d.py                    # generate all 4
  python3 generate_datasets_768d.py dataset_768d_1w    # generate only 1w
"""

import gzip
import os
import sys
import time
import numpy as np
from multiprocessing import Pool
import pyarrow.parquet as pq

# Configuration
SOURCE_FILE = "datasets/source_768d/cohere_1m_train.parquet"
OUTPUT_BASE = "datasets"
NUM_USERS = 100
SHARD_SIZE = 1_000_000   # rows per shard
SEED = 42
WRITE_CHUNK = 20_000     # lines to batch before writing
NUM_WORKERS = 32
COMPRESS_LEVEL = 1        # fast gzip
FLOAT_FMT = "%.6g"       # 6 significant digits (float32 has ~7)

DATASETS = [
    ("dataset_768d_1w",   10_000),
    ("dataset_768d_10w",  100_000),
    ("dataset_768d_50w",  500_000),
    ("dataset_768d_100w", 1_000_000),
]

# ─── Phase 1: Load and pre-format source vectors ──────────────────
# Globals inherited by fork (copy-on-write)
EMBEDDINGS = []     # list of pre-formatted vector strings


def _format_chunk(chunk_arr):
    """Format a chunk of numpy vectors into string list. Used by parallel pre-format."""
    result = []
    for vec in chunk_arr:
        result.append("[" + ",".join(FLOAT_FMT % v for v in vec) + "]")
    return result


def load_and_preformat(filepath):
    """Load parquet → numpy array → pre-format all vectors as strings in parallel."""
    print(f"Phase 1: Loading source data from {filepath}...")
    t0 = time.time()

    pf = pq.ParquetFile(filepath)
    num_rows = pf.metadata.num_rows
    print(f"  Parquet: {num_rows:,} rows, {pf.metadata.num_row_groups} row groups")

    # Read all vectors into numpy
    all_vecs = []
    loaded = 0
    for batch in pf.iter_batches(batch_size=50_000, columns=["emb"]):
        emb_list = batch.column("emb").to_pylist()
        all_vecs.append(np.array(emb_list, dtype=np.float32))
        loaded += len(emb_list)
        print(f"  Reading parquet: {loaded:,}/{num_rows:,}...", end="\r", flush=True)

    arr = np.vstack(all_vecs)
    del all_vecs
    t1 = time.time()
    print(f"  Loaded {arr.shape[0]:,} x {arr.shape[1]}D in {t1 - t0:.1f}s "
          f"({arr.nbytes / 1e9:.2f} GB)     ")

    # Pre-format in parallel: split array into chunks for workers
    print(f"  Pre-formatting {arr.shape[0]:,} vector strings ({NUM_WORKERS} workers)...")
    chunk_size = max(1, arr.shape[0] // NUM_WORKERS)
    chunks = [arr[i:i + chunk_size] for i in range(0, arr.shape[0], chunk_size)]

    embeddings = []
    with Pool(processes=min(NUM_WORKERS, len(chunks))) as pool:
        for chunk_result in pool.imap(_format_chunk, chunks):
            embeddings.extend(chunk_result)
            print(f"  Formatted: {len(embeddings):,}/{num_rows:,}...", end="\r", flush=True)

    del arr  # free numpy array
    t2 = time.time()

    avg_len = sum(len(s) for s in embeddings[:1000]) / 1000
    mem_mb = len(embeddings) * avg_len / (1024 * 1024)
    print(f"  Pre-formatted {len(embeddings):,} vectors in {t2 - t1:.1f}s "
          f"(avg {avg_len:.0f} chars, ~{mem_mb:.0f} MB strings)   ")
    print(f"  Total Phase 1: {t2 - t0:.1f}s")
    return embeddings


# ─── Phase 2: Generate datasets (string lookup only) ─────────────

def generate_shard(args):
    """Worker function: generate one shard file.
    Uses fork-inherited EMBEDDINGS list (string lookups only, no formatting).
    """
    shard_idx, user_ids, vectors_per_user, output_dir, base_seed = args
    source_size = len(EMBEDDINGS)
    shard_path = os.path.join(output_dir, f"shard_{shard_idx:03d}.tsv.gz")
    total_written = 0

    with gzip.open(shard_path, "wb", compresslevel=COMPRESS_LEVEL) as fout:
        for user_id in user_ids:
            rng = np.random.default_rng(base_seed + user_id)
            indices = rng.integers(0, source_size, size=vectors_per_user)

            for chunk_start in range(0, vectors_per_user, WRITE_CHUNK):
                chunk_end = min(chunk_start + WRITE_CHUNK, vectors_per_user)
                chunk_indices = indices[chunk_start:chunk_end]

                lines = []
                for i, idx in enumerate(chunk_indices, start=chunk_start):
                    lines.append(f"{user_id}\t{i}\t{EMBEDDINGS[idx]}")
                chunk_bytes = ("\n".join(lines) + "\n").encode("utf-8")
                fout.write(chunk_bytes)
                total_written += len(chunk_indices)

    file_size = os.path.getsize(shard_path)
    return shard_idx, total_written, file_size


def generate_dataset(dataset_name, vectors_per_user):
    """Generate a dataset using multiprocessing across shards."""
    total_rows = NUM_USERS * vectors_per_user
    output_dir = os.path.join(OUTPUT_BASE, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    users_per_shard = max(1, SHARD_SIZE // vectors_per_user)
    shard_tasks = []
    shard_idx = 0
    for start_user in range(0, NUM_USERS, users_per_shard):
        end_user = min(start_user + users_per_shard, NUM_USERS)
        user_ids = list(range(start_user, end_user))
        shard_tasks.append((shard_idx, user_ids, vectors_per_user, output_dir, SEED))
        shard_idx += 1

    num_shards = len(shard_tasks)
    workers = min(NUM_WORKERS, num_shards)

    print(f"\n{'=' * 60}")
    print(f"Phase 2: {dataset_name} ({NUM_USERS} users x {vectors_per_user:,} = {total_rows:,} rows)")
    print(f"  Shards: {num_shards} ({users_per_shard} users/shard)")
    print(f"  Workers: {workers}")
    print(f"  Output: {output_dir}/")
    print(f"{'=' * 60}")

    t0 = time.time()
    total_written = 0
    total_size = 0

    with Pool(processes=workers) as pool:
        for result in pool.imap_unordered(generate_shard, shard_tasks):
            s_idx, s_rows, s_size = result
            total_written += s_rows
            total_size += s_size
            elapsed = time.time() - t0
            pct = total_written / total_rows * 100
            speed = total_written / elapsed if elapsed > 0 else 0
            print(f"  Shard {s_idx:03d} done | "
                  f"{total_written:>12,}/{total_rows:,} ({pct:5.1f}%) | "
                  f"{elapsed:7.1f}s | {speed:,.0f} rows/s")

    elapsed = time.time() - t0
    print(f"\n  Done! {total_written:,} rows in {elapsed:.1f}s "
          f"({total_written / elapsed:,.0f} rows/s)")
    print(f"  Compressed size: {total_size / (1024 ** 3):.2f} GB ({num_shards} shards)")
    return total_written


def main():
    global EMBEDDINGS

    target = sys.argv[1] if len(sys.argv) > 1 else None

    EMBEDDINGS = load_and_preformat(SOURCE_FILE)

    t0 = time.time()
    for dataset_name, vectors_per_user in DATASETS:
        if target and dataset_name != target:
            continue
        generate_dataset(dataset_name, vectors_per_user)

    elapsed = time.time() - t0
    print(f"\nAll done! Total generation time: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
