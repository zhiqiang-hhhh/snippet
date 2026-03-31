#!/usr/bin/env python3
"""
Doris Vector Search Benchmark

Tests brute-force vector search performance with user_id filter.
Generates detailed CSV results and visualization charts.

Test scenarios:
  1.  Scale test       - Fixed user, fixed LIMIT, across 4 table sizes
  2.  Top-K test       - Fixed table, varying LIMIT (1, 10, 50, 100, 500)
  3.  Filter test      - Varying number of users in WHERE clause
  4.  Distance fn test - inner_product / l2_distance / cosine_similarity
  5.  Concurrency test - Concurrent query throughput
  6.  Large Top-K test - Extended LIMIT range (100, 500, 1000, 2000, 5000)
  7.  Full scan test   - No user_id filter, brute-force across entire table
  8.  Cache effect test - Same query repeated many times, measure cache warming
  9.  Concurrent scale  - Concurrency test across different table sizes
  10. High concurrency  - Extended concurrency levels (1, 4, 8, 16, 32, 64)
"""

import os
import sys
import time
import json
import random
import argparse
import warnings
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import pymysql
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

warnings.filterwarnings("ignore")

# ─── Configuration ───────────────────────────────────────────────
DORIS_HOST = "127.0.0.1"
DORIS_PORT = 9930
DORIS_USER = "root"
DORIS_PASS = ""
DORIS_DB   = "test_demo"

TABLES_128D = {
    "1w":   "sift_user_1w",
    "10w":  "sift_user_10w",
    "50w":  "sift_user_50w",
    "100w": "sift_user_100w",
}
TABLES_768D = {
    "1w":   "cohere_user_1w",
    "10w":  "cohere_user_10w",
    "50w":  "cohere_user_50w",
    "100w": "cohere_user_100w",
}

# Per-dimension defaults: (tables, default_dist_fn, order_dir, query_source_table)
DIM_CONFIG = {
    128: {
        "tables":       TABLES_128D,
        "dist_fn":      "l2_distance",
        "order":        "ASC",
        "query_table":  "sift_user_1w",
        "label":        "128D (SIFT)",
        "suffix":       "128d",
    },
    768: {
        "tables":       TABLES_768D,
        "dist_fn":      "cosine_similarity",
        "order":        "DESC",
        "query_table":  "cohere_user_1w",
        "label":        "768D (Cohere)",
        "suffix":       "768d",
    },
}

# Back-compat: default TABLES alias (overridden at runtime)
TABLES = TABLES_128D

WARMUP_RUNS  = 2
BENCH_RUNS   = 10   # iterations per test case
NUM_QUERIES  = 5    # different query vectors to average over
SEED         = 42
OUTPUT_DIR   = "benchmark_results"

# Session variables to SET on each new connection (populated by --parallel etc.)
SESSION_VARS = {}

# ─── Helpers ─────────────────────────────────────────────────────

def get_conn():
    conn = pymysql.connect(
        host=DORIS_HOST, port=DORIS_PORT,
        user=DORIS_USER, password=DORIS_PASS,
        database=DORIS_DB, autocommit=True,
    )
    if SESSION_VARS:
        with conn.cursor() as cur:
            for k, v in SESSION_VARS.items():
                cur.execute(f"SET {k} = {v}")
    return conn


def load_query_vectors(dim_conf, n=NUM_QUERIES, seed=SEED):
    """Pick n random embedding vectors from the smallest table as query vectors."""
    conn = get_conn()
    rng = random.Random(seed)
    user_ids = rng.sample(range(100), min(n, 100))
    query_table = dim_conf["query_table"]
    vectors = []
    with conn.cursor() as cur:
        for uid in user_ids:
            eid = rng.randint(0, 9999)
            cur.execute(
                f"SELECT embedding FROM {query_table} "
                f"WHERE user_id={uid} AND id={eid} LIMIT 1"
            )
            row = cur.fetchone()
            if row:
                vectors.append(row[0])
    conn.close()
    print(f"Loaded {len(vectors)} query vectors from {query_table}")
    return vectors


def format_vec(vec_str):
    """Ensure vector string is in [v1,v2,...] format."""
    v = vec_str.strip()
    if not v.startswith("["):
        v = "[" + v + "]"
    return v


def run_query(conn, sql):
    """Execute a query and return (latency_seconds, row_count)."""
    t0 = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    elapsed = time.perf_counter() - t0
    return elapsed, len(rows)


def percentile(data, p):
    """Calculate p-th percentile."""
    sorted_d = sorted(data)
    k = (len(sorted_d) - 1) * p / 100.0
    f = int(k)
    c = f + 1 if f + 1 < len(sorted_d) else f
    return sorted_d[f] + (k - f) * (sorted_d[c] - sorted_d[f])


def run_bench(conn, sql_template, query_vecs, warmup=WARMUP_RUNS, runs=BENCH_RUNS):
    """Run benchmark for a SQL template with {vec} placeholder.

    Returns dict with timing stats (in milliseconds).
    """
    latencies = []

    for qi, qv in enumerate(query_vecs):
        vec = format_vec(qv)
        sql = sql_template.replace("{vec}", vec)

        # warmup
        for _ in range(warmup):
            run_query(conn, sql)

        # bench
        for _ in range(runs):
            elapsed, _ = run_query(conn, sql)
            latencies.append(elapsed * 1000)  # ms

    return {
        "avg_ms":  round(np.mean(latencies), 2),
        "p50_ms":  round(percentile(latencies, 50), 2),
        "p90_ms":  round(percentile(latencies, 90), 2),
        "p99_ms":  round(percentile(latencies, 99), 2),
        "min_ms":  round(min(latencies), 2),
        "max_ms":  round(max(latencies), 2),
        "std_ms":  round(np.std(latencies), 2),
        "runs":    len(latencies),
    }


# ─── Test Scenarios ──────────────────────────────────────────────

def test_scale(conn, query_vecs, dim_conf):
    """Scenario 1: Same query across different table sizes."""
    tables = dim_conf["tables"]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]

    print("\n" + "=" * 60)
    print(f"Scenario 1: Scale Test [{dim_conf['label']}] (user_id=42, LIMIT 10)")
    print("  How does latency change with per-user data volume?")
    print("=" * 60)

    results = []
    for label, table in tables.items():
        sql = (f"SELECT id FROM {table} "
               f"WHERE user_id = 42 "
               f"ORDER BY {dist_fn}(embedding, {{vec}}) {order} "
               f"LIMIT 10")
        print(f"  Testing {table} ({label}/user)...", end="", flush=True)
        stats = run_bench(conn, sql, query_vecs)
        stats["table"] = table
        stats["scale"] = label
        results.append(stats)
        print(f"  avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


def test_topk(conn, query_vecs, dim_conf, table_key="10w"):
    """Scenario 2: Varying LIMIT (Top-K)."""
    table = dim_conf["tables"][table_key]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]
    limits = [1, 10, 50, 100, 500]

    print(f"\n{'=' * 60}")
    print(f"Scenario 2: Top-K Test [{dim_conf['label']}] on {table} (user_id=42)")
    print(f"  How does LIMIT K affect latency?")
    print(f"{'=' * 60}")

    results = []
    for k in limits:
        sql = (f"SELECT id FROM {table} "
               f"WHERE user_id = 42 "
               f"ORDER BY {dist_fn}(embedding, {{vec}}) {order} "
               f"LIMIT {k}")
        print(f"  LIMIT {k:>4d}...", end="", flush=True)
        stats = run_bench(conn, sql, query_vecs)
        stats["table"] = table
        stats["limit_k"] = k
        results.append(stats)
        print(f"  avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


def test_filter(conn, query_vecs, dim_conf, table_key="10w"):
    """Scenario 3: Varying filter selectivity (number of users)."""
    table = dim_conf["tables"][table_key]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]
    user_counts = [1, 5, 10, 50, 100]

    print(f"\n{'=' * 60}")
    print(f"Scenario 3: Filter Selectivity Test [{dim_conf['label']}] on {table} (LIMIT 10)")
    print(f"  How does scanning more users affect latency?")
    print(f"{'=' * 60}")

    rng = random.Random(SEED)
    all_users = list(range(100))

    results = []
    for n_users in user_counts:
        uids = sorted(rng.sample(all_users, n_users))
        if n_users == 1:
            where = f"user_id = {uids[0]}"
        else:
            where = f"user_id IN ({','.join(map(str, uids))})"

        sql = (f"SELECT id FROM {table} "
               f"WHERE {where} "
               f"ORDER BY {dist_fn}(embedding, {{vec}}) {order} "
               f"LIMIT 10")
        print(f"  {n_users:>3d} users...", end="", flush=True)
        stats = run_bench(conn, sql, query_vecs)
        stats["table"] = table
        stats["n_users"] = n_users
        results.append(stats)
        print(f"  avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


def test_distance_fn(conn, query_vecs, dim_conf, table_key="10w"):
    """Scenario 4: Compare different distance/similarity functions."""
    table = dim_conf["tables"][table_key]
    functions = {
        "l2_distance":       ("ASC",  "l2_distance(embedding, {vec})"),
        "inner_product":     ("DESC", "inner_product(embedding, {vec})"),
        "cosine_similarity": ("DESC", "cosine_similarity(embedding, {vec})"),
    }

    print(f"\n{'=' * 60}")
    print(f"Scenario 4: Distance Function Test [{dim_conf['label']}] on {table} (user_id=42, LIMIT 10)")
    print(f"  Comparing l2_distance / inner_product / cosine_similarity")
    print(f"{'=' * 60}")

    results = []
    for fn_name, (order, expr) in functions.items():
        sql = (f"SELECT id FROM {table} "
               f"WHERE user_id = 42 "
               f"ORDER BY {expr} {order} "
               f"LIMIT 10")
        print(f"  {fn_name:>20s}...", end="", flush=True)
        stats = run_bench(conn, sql, query_vecs)
        stats["table"] = table
        stats["function"] = fn_name
        results.append(stats)
        print(f"  avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


def test_concurrency(query_vecs, dim_conf, table_key="10w", concurrency_levels=None):
    """Scenario 5: Concurrent query throughput."""
    if concurrency_levels is None:
        concurrency_levels = [1, 2, 4, 8, 16]
    table = dim_conf["tables"][table_key]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]

    print(f"\n{'=' * 60}")
    print(f"Scenario 5: Concurrency Test [{dim_conf['label']}] on {table} (user_id=42, LIMIT 10)")
    print(f"  Measuring throughput at different concurrency levels")
    print(f"{'=' * 60}")

    def worker(vec_str, n_queries=20):
        """Each worker runs n_queries and returns total latency."""
        c = get_conn()
        vec = format_vec(vec_str)
        sql = (f"SELECT id FROM {table} "
               f"WHERE user_id = 42 "
               f"ORDER BY {dist_fn}(embedding, {vec}) {order} "
               f"LIMIT 10")
        latencies = []
        for _ in range(n_queries):
            t0 = time.perf_counter()
            with c.cursor() as cur:
                cur.execute(sql)
                cur.fetchall()
            latencies.append((time.perf_counter() - t0) * 1000)
        c.close()
        return latencies

    results = []
    for c_level in concurrency_levels:
        queries_per_worker = max(10, 40 // c_level)
        vec = format_vec(query_vecs[0])

        # Warmup
        wconn = get_conn()
        sql_w = (f"SELECT id FROM {table} "
                 f"WHERE user_id = 42 "
                 f"ORDER BY {dist_fn}(embedding, {vec}) {order} "
                 f"LIMIT 10")
        for _ in range(3):
            run_query(wconn, sql_w)
        wconn.close()

        t0 = time.perf_counter()
        all_latencies = []
        with ThreadPoolExecutor(max_workers=c_level) as pool:
            futures = [pool.submit(worker, query_vecs[i % len(query_vecs)], queries_per_worker)
                       for i in range(c_level)]
            for f in as_completed(futures):
                all_latencies.extend(f.result())
        wall_time = time.perf_counter() - t0

        total_queries = len(all_latencies)
        qps = total_queries / wall_time

        stats = {
            "concurrency":  c_level,
            "table":        table,
            "total_queries": total_queries,
            "wall_time_s":  round(wall_time, 2),
            "qps":          round(qps, 1),
            "avg_ms":       round(np.mean(all_latencies), 2),
            "p50_ms":       round(percentile(all_latencies, 50), 2),
            "p90_ms":       round(percentile(all_latencies, 90), 2),
            "p99_ms":       round(percentile(all_latencies, 99), 2),
        }
        results.append(stats)
        print(f"  C={c_level:>2d}  QPS={qps:>7.1f}  avg={stats['avg_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


def test_large_topk(conn, query_vecs, dim_conf, table_key="10w"):
    """Scenario 6: Extended Top-K range to test sorting overhead."""
    table = dim_conf["tables"][table_key]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]
    limits = [100, 500, 1000, 2000, 5000]

    print(f"\n{'=' * 60}")
    print(f"Scenario 6: Large Top-K Test [{dim_conf['label']}] on {table} (user_id=42)")
    print(f"  How does very large LIMIT affect latency?")
    print(f"{'=' * 60}")

    results = []
    for k in limits:
        sql = (f"SELECT id FROM {table} "
               f"WHERE user_id = 42 "
               f"ORDER BY {dist_fn}(embedding, {{vec}}) {order} "
               f"LIMIT {k}")
        print(f"  LIMIT {k:>5d}...", end="", flush=True)
        stats = run_bench(conn, sql, query_vecs)
        stats["table"] = table
        stats["limit_k"] = k
        results.append(stats)
        print(f"  avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


def test_full_scan(conn, query_vecs, dim_conf):
    """Scenario 7: No user_id filter - full table brute-force scan."""
    tables = dim_conf["tables"]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]

    print(f"\n{'=' * 60}")
    print(f"Scenario 7: Full Scan Test [{dim_conf['label']}] (no user_id filter, LIMIT 10)")
    print(f"  How expensive is scanning ALL users?")
    print(f"{'=' * 60}")

    results = []
    for label, table in tables.items():
        sql = (f"SELECT user_id, id FROM {table} "
               f"ORDER BY {dist_fn}(embedding, {{vec}}) {order} "
               f"LIMIT 10")
        print(f"  Full scan {table} ({label})...", end="", flush=True)
        stats = run_bench(conn, sql, query_vecs, warmup=1, runs=5)
        stats["table"] = table
        stats["scale"] = label
        results.append(stats)
        print(f"  avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


def test_cache_effect(conn, query_vecs, dim_conf, table_key="10w"):
    """Scenario 8: Repeated identical query to measure cache warming."""
    table = dim_conf["tables"][table_key]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]
    total_iterations = 50

    print(f"\n{'=' * 60}")
    print(f"Scenario 8: Cache Effect Test [{dim_conf['label']}] on {table}")
    print(f"  Running same query {total_iterations} times, no warmup")
    print(f"{'=' * 60}")

    vec = format_vec(query_vecs[0])
    sql = (f"SELECT id FROM {table} "
           f"WHERE user_id = 42 "
           f"ORDER BY {dist_fn}(embedding, {vec}) {order} "
           f"LIMIT 10")

    latencies = []
    for i in range(total_iterations):
        elapsed, _ = run_query(conn, sql)
        latencies.append(elapsed * 1000)

    # Group into windows of 5 to show progression
    window = 5
    results = []
    for start in range(0, total_iterations, window):
        chunk = latencies[start:start + window]
        results.append({
            "iteration_start": start + 1,
            "iteration_end": min(start + window, total_iterations),
            "table": table,
            "avg_ms": round(np.mean(chunk), 2),
            "min_ms": round(min(chunk), 2),
            "max_ms": round(max(chunk), 2),
        })

    for r in results:
        print(f"  Iter {r['iteration_start']:>2d}-{r['iteration_end']:>2d}: "
              f"avg={r['avg_ms']:.1f}ms  min={r['min_ms']:.1f}ms  max={r['max_ms']:.1f}ms")

    # Also return the raw per-iteration data for charting
    raw_df = pd.DataFrame({
        "iteration": list(range(1, total_iterations + 1)),
        "latency_ms": [round(x, 2) for x in latencies],
        "table": table,
    })
    return pd.DataFrame(results), raw_df


def test_concurrent_scale(query_vecs, dim_conf, concurrency_levels=None, duration=30):
    """Scenario 9: Duration-based concurrent scale test.

    For every (table_scale x concurrency_level) combination, run queries for
    *duration* seconds and measure QPS / P50 / P90 / P99.
    """
    import threading

    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16, 32]

    tables = dim_conf["tables"]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]

    print(f"\n{'=' * 60}")
    print(f"Scenario 9: Concurrent Scale Test [{dim_conf['label']}]")
    print(f"  Concurrency levels: {concurrency_levels}")
    print(f"  Duration per case : {duration}s")
    print(f"  Query: WHERE user_id=42 ORDER BY {dist_fn}(...) {order} LIMIT 10")
    print(f"{'=' * 60}")

    def worker(table, vec_str, stop_event):
        """Run queries in a loop until stop_event is set."""
        c = get_conn()
        vec = format_vec(vec_str)
        sql = (f"SELECT id FROM {table} "
               f"WHERE user_id = 42 "
               f"ORDER BY {dist_fn}(embedding, {vec}) {order} "
               f"LIMIT 10")
        latencies = []
        while not stop_event.is_set():
            t0 = time.perf_counter()
            try:
                with c.cursor() as cur:
                    cur.execute(sql)
                    cur.fetchall()
                latencies.append((time.perf_counter() - t0) * 1000)
            except Exception:
                # Reconnect on error and continue
                try:
                    c.close()
                except Exception:
                    pass
                c = get_conn()
        try:
            c.close()
        except Exception:
            pass
        return latencies

    results = []
    for label, table in tables.items():
        # Warmup once per table
        wconn = get_conn()
        vec = format_vec(query_vecs[0])
        sql_w = (f"SELECT id FROM {table} "
                 f"WHERE user_id = 42 "
                 f"ORDER BY {dist_fn}(embedding, {vec}) {order} "
                 f"LIMIT 10")
        for _ in range(3):
            run_query(wconn, sql_w)
        wconn.close()

        for c_level in concurrency_levels:
            stop_event = threading.Event()
            t0 = time.perf_counter()
            all_latencies = []

            with ThreadPoolExecutor(max_workers=c_level) as pool:
                futures = [
                    pool.submit(worker, table, query_vecs[i % len(query_vecs)], stop_event)
                    for i in range(c_level)
                ]
                # Let it run for the specified duration
                time.sleep(duration)
                stop_event.set()

                for f in as_completed(futures):
                    all_latencies.extend(f.result())

            wall_time = time.perf_counter() - t0
            total_queries = len(all_latencies)
            qps = total_queries / wall_time if wall_time > 0 else 0

            stats = {
                "scale":         label,
                "table":         table,
                "concurrency":   c_level,
                "duration_s":    duration,
                "total_queries": total_queries,
                "wall_time_s":   round(wall_time, 2),
                "qps":           round(qps, 1),
                "avg_ms":        round(np.mean(all_latencies), 2) if all_latencies else 0,
                "p50_ms":        round(percentile(all_latencies, 50), 2) if all_latencies else 0,
                "p90_ms":        round(percentile(all_latencies, 90), 2) if all_latencies else 0,
                "p99_ms":        round(percentile(all_latencies, 99), 2) if all_latencies else 0,
            }
            results.append(stats)
            print(f"  {table:>20s}  C={c_level:<3d}  QPS={qps:>7.1f}  "
                  f"avg={stats['avg_ms']:.1f}ms  p50={stats['p50_ms']:.1f}ms  "
                  f"p90={stats['p90_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms  "
                  f"({total_queries} queries in {wall_time:.1f}s)")

    return pd.DataFrame(results)


def test_high_concurrency(query_vecs, dim_conf, table_key="10w", concurrency_levels=None):
    """Scenario 10: Extended concurrency levels."""
    if concurrency_levels is None:
        concurrency_levels = [1, 4, 8, 16, 32, 64]
    table = dim_conf["tables"][table_key]
    dist_fn = dim_conf["dist_fn"]
    order = dim_conf["order"]

    print(f"\n{'=' * 60}")
    print(f"Scenario 10: High Concurrency Test [{dim_conf['label']}] on {table} (LIMIT 10)")
    print(f"  Extended concurrency levels: {concurrency_levels}")
    print(f"{'=' * 60}")

    def worker(vec_str, n_queries=20):
        c = get_conn()
        vec = format_vec(vec_str)
        sql = (f"SELECT id FROM {table} "
               f"WHERE user_id = 42 "
               f"ORDER BY {dist_fn}(embedding, {vec}) {order} "
               f"LIMIT 10")
        latencies = []
        for _ in range(n_queries):
            t0 = time.perf_counter()
            with c.cursor() as cur:
                cur.execute(sql)
                cur.fetchall()
            latencies.append((time.perf_counter() - t0) * 1000)
        c.close()
        return latencies

    results = []
    for c_level in concurrency_levels:
        queries_per_worker = max(10, 60 // c_level)
        vec = format_vec(query_vecs[0])

        # Warmup
        wconn = get_conn()
        sql_w = (f"SELECT id FROM {table} "
                 f"WHERE user_id = 42 "
                 f"ORDER BY {dist_fn}(embedding, {vec}) {order} "
                 f"LIMIT 10")
        for _ in range(3):
            run_query(wconn, sql_w)
        wconn.close()

        t0 = time.perf_counter()
        all_latencies = []
        with ThreadPoolExecutor(max_workers=c_level) as pool:
            futures = [pool.submit(worker, query_vecs[i % len(query_vecs)], queries_per_worker)
                       for i in range(c_level)]
            for f in as_completed(futures):
                all_latencies.extend(f.result())
        wall_time = time.perf_counter() - t0

        total_queries = len(all_latencies)
        qps = total_queries / wall_time
        stats = {
            "concurrency":   c_level,
            "table":         table,
            "total_queries": total_queries,
            "wall_time_s":   round(wall_time, 2),
            "qps":           round(qps, 1),
            "avg_ms":        round(np.mean(all_latencies), 2),
            "p50_ms":        round(percentile(all_latencies, 50), 2),
            "p90_ms":        round(percentile(all_latencies, 90), 2),
            "p99_ms":        round(percentile(all_latencies, 99), 2),
        }
        results.append(stats)
        print(f"  C={c_level:>2d}  QPS={qps:>7.1f}  avg={stats['avg_ms']:.1f}ms  p99={stats['p99_ms']:.1f}ms")

    return pd.DataFrame(results)


# ─── Visualization ───────────────────────────────────────────────

def setup_plot_style():
    plt.rcParams.update({
        "figure.figsize": (10, 6),
        "figure.dpi": 150,
        "font.size": 11,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def plot_scale(df, output_dir):
    """Bar chart: latency across table sizes."""
    fig, ax = plt.subplots()
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["p50_ms"], width, label="P50", color="#4C72B0")
    ax.bar(x,         df["avg_ms"], width, label="Avg", color="#55A868")
    ax.bar(x + width, df["p99_ms"], width, label="P99", color="#C44E52")

    ax.set_xlabel("Vectors per User")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Scenario 1: Latency vs Data Scale\n(user_id=42, LIMIT 10, l2_distance)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["scale"])
    ax.legend()

    for i, row in df.iterrows():
        ax.text(i, row["p99_ms"] + 1, f'{row["p99_ms"]:.0f}', ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "01_scale_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_topk(df, output_dir):
    """Line chart: latency vs LIMIT K."""
    fig, ax = plt.subplots()

    ax.plot(df["limit_k"], df["avg_ms"], "o-", label="Avg", color="#55A868", linewidth=2)
    ax.plot(df["limit_k"], df["p50_ms"], "s--", label="P50", color="#4C72B0")
    ax.plot(df["limit_k"], df["p99_ms"], "^--", label="P99", color="#C44E52")

    ax.set_xlabel("Top-K (LIMIT)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Scenario 2: Latency vs Top-K\n({df['table'].iloc[0]}, user_id=42, l2_distance)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "02_topk_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_filter(df, output_dir):
    """Bar chart: latency vs number of users in filter."""
    fig, ax = plt.subplots()
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["p50_ms"], width, label="P50", color="#4C72B0")
    ax.bar(x,         df["avg_ms"], width, label="Avg", color="#55A868")
    ax.bar(x + width, df["p99_ms"], width, label="P99", color="#C44E52")

    ax.set_xlabel("Number of Users in Filter")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Scenario 3: Latency vs Filter Selectivity\n({df['table'].iloc[0]}, LIMIT 10, l2_distance)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["n_users"])
    ax.legend()

    for i, row in df.iterrows():
        ax.text(i, row["p99_ms"] + 1, f'{row["p99_ms"]:.0f}', ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "03_filter_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_distance_fn(df, output_dir):
    """Horizontal bar chart: comparing distance functions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    y = np.arange(len(df))
    height = 0.25

    ax.barh(y - height, df["p50_ms"], height, label="P50", color="#4C72B0")
    ax.barh(y,          df["avg_ms"], height, label="Avg", color="#55A868")
    ax.barh(y + height, df["p99_ms"], height, label="P99", color="#C44E52")

    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Distance Function")
    ax.set_title(f"Scenario 4: Distance Function Comparison\n({df['table'].iloc[0]}, user_id=42, LIMIT 10)")
    ax.set_yticks(y)
    ax.set_yticklabels(df["function"])
    ax.legend()

    for i, row in df.iterrows():
        ax.text(row["p99_ms"] + 0.5, i + height, f'{row["p99_ms"]:.0f}', va="center", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "04_distance_fn_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_concurrency(df, output_dir):
    """Dual-axis chart: QPS and P99 vs concurrency."""
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    x = df["concurrency"]
    ax1.bar(x, df["qps"], width=0.6, alpha=0.7, color="#4C72B0", label="QPS")
    ax2.plot(x, df["p99_ms"], "o-", color="#C44E52", linewidth=2, markersize=8, label="P99 Latency")
    ax2.plot(x, df["avg_ms"], "s--", color="#55A868", linewidth=2, markersize=6, label="Avg Latency")

    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("QPS", color="#4C72B0")
    ax2.set_ylabel("Latency (ms)", color="#C44E52")
    ax1.set_title(f"Scenario 5: Concurrency Test\n({df['table'].iloc[0]}, user_id=42, LIMIT 10)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(output_dir, "05_concurrency_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_large_topk(df, output_dir):
    """Line chart: latency vs large LIMIT K values."""
    fig, ax = plt.subplots()

    ax.plot(df["limit_k"], df["avg_ms"], "o-", label="Avg", color="#55A868", linewidth=2)
    ax.plot(df["limit_k"], df["p50_ms"], "s--", label="P50", color="#4C72B0")
    ax.plot(df["limit_k"], df["p99_ms"], "^--", label="P99", color="#C44E52")

    ax.set_xlabel("Top-K (LIMIT)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Scenario 6: Large Top-K Test\n({df['table'].iloc[0]}, user_id=42, l2_distance)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "06_large_topk_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_full_scan(df, output_dir):
    """Bar chart: full scan latency across table sizes."""
    fig, ax = plt.subplots()
    x = np.arange(len(df))
    width = 0.25

    ax.bar(x - width, df["p50_ms"], width, label="P50", color="#4C72B0")
    ax.bar(x,         df["avg_ms"], width, label="Avg", color="#55A868")
    ax.bar(x + width, df["p99_ms"], width, label="P99", color="#C44E52")

    ax.set_xlabel("Total Rows in Table")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Scenario 7: Full Table Scan (no user_id filter)\n(LIMIT 10, l2_distance)")
    ax.set_xticks(x)
    ax.set_xticklabels(df["scale"])
    ax.legend()

    for i, row in df.iterrows():
        ax.text(i, row["p99_ms"] * 1.02, f'{row["p99_ms"]:.0f}', ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(output_dir, "07_full_scan_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_cache_effect(raw_df, output_dir):
    """Line chart: per-iteration latency showing cache warming."""
    fig, ax = plt.subplots()

    ax.plot(raw_df["iteration"], raw_df["latency_ms"], "o-", markersize=4,
            color="#4C72B0", linewidth=1.5, alpha=0.8)

    # Add a rolling average line
    if len(raw_df) >= 5:
        rolling = raw_df["latency_ms"].rolling(window=5, min_periods=1).mean()
        ax.plot(raw_df["iteration"], rolling, "-", color="#C44E52",
                linewidth=2.5, label="5-point rolling avg")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(f"Scenario 8: Cache Effect Test\n({raw_df['table'].iloc[0]}, user_id=42, LIMIT 10)")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(output_dir, "08_cache_effect_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


def plot_concurrent_scale(df, output_dir):
    """Multi-panel chart: QPS and latency across table sizes x concurrency levels.

    Generates two charts:
      09a - QPS grouped bar chart (scales on x-axis, one bar group per concurrency)
      09b - P50/P90/P99 latency heatmap-style line chart per concurrency
    """
    concurrency_levels = sorted(df["concurrency"].unique())
    scales = list(dict.fromkeys(df["scale"]))  # preserve order
    dist_fn = "cosine_similarity" if "cohere" in df["table"].iloc[0] else "l2_distance"
    dur = df["duration_s"].iloc[0] if "duration_s" in df.columns else "?"

    # ── Chart 09a: QPS grouped bar ──
    fig, ax = plt.subplots(figsize=(12, 6))
    n_scales = len(scales)
    n_conc = len(concurrency_levels)
    bar_width = 0.8 / n_conc
    cmap = matplotlib.colormaps.get_cmap("Set2")  # type: ignore
    colors = [cmap(i / max(n_conc - 1, 1)) for i in range(n_conc)]

    for i, c_level in enumerate(concurrency_levels):
        sub = df[df["concurrency"] == c_level]
        x = np.arange(n_scales) + i * bar_width
        qps_vals = [sub[sub["scale"] == s]["qps"].values[0] if len(sub[sub["scale"] == s]) else 0
                     for s in scales]
        bars = ax.bar(x, qps_vals, width=bar_width, alpha=0.85, color=colors[i],
                       label=f"C={c_level}")
        # Value labels on bars
        for bar, val in zip(bars, qps_vals):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f"{val:.0f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Table Scale")
    ax.set_ylabel("QPS")
    ax.set_title(f"Scenario 9: Concurrent Scale Test — QPS\n"
                 f"(user_id=42, LIMIT 10, {dist_fn}, {dur}s per case)")
    ax.set_xticks(np.arange(n_scales) + bar_width * (n_conc - 1) / 2)
    ax.set_xticklabels(scales)
    ax.legend(title="Concurrency")
    plt.tight_layout()
    path_a = os.path.join(output_dir, "09a_concurrent_scale_qps.png")
    fig.savefig(path_a)
    plt.close(fig)
    print(f"  Saved {path_a}")

    # ── Chart 09b: Latency (P50/P90/P99) per scale, one subplot per concurrency ──
    n_cols = min(3, n_conc)
    n_rows = (n_conc + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
    for idx, c_level in enumerate(concurrency_levels):
        r, c_idx = divmod(idx, n_cols)
        ax = axes[r][c_idx]
        sub = df[df["concurrency"] == c_level].set_index("scale").reindex(scales)
        x = np.arange(len(scales))
        ax.plot(x, sub["p50_ms"], "o-", label="P50", linewidth=2, markersize=6)
        ax.plot(x, sub["p90_ms"], "s-", label="P90", linewidth=2, markersize=6)
        ax.plot(x, sub["p99_ms"], "^-", label="P99", linewidth=2, markersize=6)
        ax.plot(x, sub["avg_ms"], "d--", label="Avg", linewidth=1.5, markersize=5, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(scales, fontsize=9)
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"C={c_level}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide empty subplots
    for idx in range(n_conc, n_rows * n_cols):
        r, c_idx = divmod(idx, n_cols)
        axes[r][c_idx].set_visible(False)

    fig.suptitle(f"Scenario 9: Concurrent Scale Test — Latency\n"
                 f"(user_id=42, LIMIT 10, {dist_fn}, {dur}s per case)", fontsize=13)
    plt.tight_layout()
    path_b = os.path.join(output_dir, "09b_concurrent_scale_latency.png")
    fig.savefig(path_b)
    plt.close(fig)
    print(f"  Saved {path_b}")

    # ── Also keep a combined 09_ for backward compat ──
    path_c = os.path.join(output_dir, "09_concurrent_scale_test.png")
    # Re-create the QPS chart as the main one
    import shutil
    shutil.copy2(path_a, path_c)
    print(f"  Saved {path_c}")


def plot_high_concurrency(df, output_dir):
    """Dual-axis chart: QPS and latency vs extended concurrency."""
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()

    x = df["concurrency"]
    ax1.bar(x, df["qps"], width=x * 0.15, alpha=0.7, color="#4C72B0", label="QPS")
    ax2.plot(x, df["p99_ms"], "o-", color="#C44E52", linewidth=2, markersize=8, label="P99 Latency")
    ax2.plot(x, df["avg_ms"], "s--", color="#55A868", linewidth=2, markersize=6, label="Avg Latency")

    ax1.set_xlabel("Concurrency")
    ax1.set_ylabel("QPS", color="#4C72B0")
    ax2.set_ylabel("Latency (ms)", color="#C44E52")
    ax1.set_title(f"Scenario 10: High Concurrency Test\n({df['table'].iloc[0]}, user_id=42, LIMIT 10)")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    plt.tight_layout()
    path = os.path.join(output_dir, "10_high_concurrency_test.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved {path}")


# ─── Main ────────────────────────────────────────────────────────

ALL_SCENARIOS = [
    "scale", "topk", "filter", "distfn", "concurrency",
    "largetopk", "fullscan", "cache", "concscale", "highconc",
]

def main():
    global DORIS_HOST, DORIS_PORT, DORIS_USER, DORIS_PASS, DORIS_DB, BENCH_RUNS, NUM_QUERIES

    parser = argparse.ArgumentParser(
        description="Doris Vector Search Benchmark",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--host", default=DORIS_HOST)
    parser.add_argument("--port", type=int, default=DORIS_PORT)
    parser.add_argument("--user", default=DORIS_USER)
    parser.add_argument("--password", default=DORIS_PASS)
    parser.add_argument("--db", default=DORIS_DB)
    parser.add_argument("--output", default=OUTPUT_DIR)
    parser.add_argument("--runs", type=int, default=BENCH_RUNS, help="Iterations per test case")
    parser.add_argument("--queries", type=int, default=NUM_QUERIES, help="Number of query vectors")
    parser.add_argument("--dim", type=int, default=128, choices=[128, 768],
                        help="Vector dimension: 128 (SIFT) or 768 (Cohere)")
    parser.add_argument("--scenarios", nargs="*", default=None,
                        help="Run specific scenarios:\n"
                             "  scale       - Latency vs data scale\n"
                             "  topk        - Latency vs Top-K\n"
                             "  filter      - Latency vs filter selectivity\n"
                             "  distfn      - Distance function comparison\n"
                             "  concurrency - Concurrent query throughput\n"
                             "  largetopk   - Extended Top-K (100-5000)\n"
                             "  fullscan    - Full table scan (no filter)\n"
                             "  cache       - Cache warming effect\n"
                             "  concscale   - Concurrency across table sizes\n"
                             "  highconc    - High concurrency (up to 64)\n"
                             "  (default: run all)")
    parser.add_argument("--parallel", type=int, default=0,
                        help="Set scan/pipeline parallelism (0=auto, 1=serial)")
    parser.add_argument("--concurrency-levels", type=int, nargs="+", default=None,
                        help="Concurrency levels for concscale test (default: 1 4 8 16 32)")
    parser.add_argument("--duration", type=int, default=30,
                        help="Duration in seconds for each concscale test case (default: 30)")
    args = parser.parse_args()

    DORIS_HOST = args.host
    DORIS_PORT = args.port
    DORIS_USER = args.user
    DORIS_PASS = args.password
    DORIS_DB   = args.db
    BENCH_RUNS = args.runs
    NUM_QUERIES = args.queries

    # Apply parallelism constraints if requested
    if args.parallel > 0:
        SESSION_VARS["parallel_pipeline_task_num"] = args.parallel
        SESSION_VARS["parallel_fragment_exec_instance_num"] = args.parallel
        SESSION_VARS["max_scanners_concurrency"] = args.parallel

    dim_conf = DIM_CONFIG[args.dim]
    TABLES = dim_conf["tables"]

    # Append dimension suffix to output dir if not already present
    output_dir = args.output
    if not output_dir.endswith(dim_conf["suffix"]):
        output_dir = f"{output_dir}_{dim_conf['suffix']}"

    os.makedirs(output_dir, exist_ok=True)
    setup_plot_style()

    run_all = args.scenarios is None
    scenarios = set(args.scenarios or [])

    print("=" * 60)
    print(f"  Doris Vector Search Benchmark [{dim_conf['label']}]")
    print(f"  Host: {DORIS_HOST}:{DORIS_PORT}  DB: {DORIS_DB}")
    print(f"  Dim: {args.dim}  Dist: {dim_conf['dist_fn']}")
    print(f"  Runs: {BENCH_RUNS}  Query vectors: {NUM_QUERIES}")
    print(f"  Parallelism: {args.parallel if args.parallel > 0 else 'auto'}")
    print(f"  Output: {output_dir}/")
    if not run_all:
        print(f"  Scenarios: {', '.join(sorted(scenarios))}")
    else:
        print(f"  Scenarios: ALL ({len(ALL_SCENARIOS)} tests)")
    print("=" * 60)

    # Load query vectors
    query_vecs = load_query_vectors(dim_conf, n=NUM_QUERIES)

    conn = get_conn()
    all_results = {}
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Scenario 1: Scale test
    if run_all or "scale" in scenarios:
        df = test_scale(conn, query_vecs, dim_conf)
        all_results["scale"] = df
        plot_scale(df, output_dir)

    # Scenario 2: Top-K test
    if run_all or "topk" in scenarios:
        df = test_topk(conn, query_vecs, dim_conf, table_key="10w")
        all_results["topk"] = df
        plot_topk(df, output_dir)

    # Scenario 3: Filter selectivity
    if run_all or "filter" in scenarios:
        df = test_filter(conn, query_vecs, dim_conf, table_key="10w")
        all_results["filter"] = df
        plot_filter(df, output_dir)

    # Scenario 4: Distance function comparison
    if run_all or "distfn" in scenarios:
        df = test_distance_fn(conn, query_vecs, dim_conf, table_key="10w")
        all_results["distfn"] = df
        plot_distance_fn(df, output_dir)

    # Scenario 6: Large Top-K test
    if run_all or "largetopk" in scenarios:
        df = test_large_topk(conn, query_vecs, dim_conf, table_key="10w")
        all_results["largetopk"] = df
        plot_large_topk(df, output_dir)

    # Scenario 7: Full scan test
    if run_all or "fullscan" in scenarios:
        df = test_full_scan(conn, query_vecs, dim_conf)
        all_results["fullscan"] = df
        plot_full_scan(df, output_dir)

    # Scenario 8: Cache effect test
    if run_all or "cache" in scenarios:
        summary_df, raw_df = test_cache_effect(conn, query_vecs, dim_conf, table_key="10w")
        all_results["cache"] = summary_df
        all_results["cache_raw"] = raw_df
        plot_cache_effect(raw_df, output_dir)

    conn.close()

    # Scenario 5: Concurrency (uses its own connections)
    if run_all or "concurrency" in scenarios:
        df = test_concurrency(query_vecs, dim_conf, table_key="10w")
        all_results["concurrency"] = df
        plot_concurrency(df, output_dir)

    # Scenario 9: Concurrent scale test
    if run_all or "concscale" in scenarios:
        df = test_concurrent_scale(query_vecs, dim_conf,
                                   concurrency_levels=args.concurrency_levels,
                                   duration=args.duration)
        all_results["concscale"] = df
        plot_concurrent_scale(df, output_dir)

    # Scenario 10: High concurrency test
    if run_all or "highconc" in scenarios:
        df = test_high_concurrency(query_vecs, dim_conf, table_key="10w")
        all_results["highconc"] = df
        plot_high_concurrency(df, output_dir)

    # Save all results to CSV
    print(f"\n{'=' * 60}")
    print("Saving results...")
    print(f"{'=' * 60}")
    for name, df in all_results.items():
        csv_path = os.path.join(output_dir, f"{name}_{timestamp}.csv")
        df.to_csv(csv_path, index=False)
        print(f"  {csv_path}")

    # Save combined summary (exclude cache_raw from JSON for brevity)
    summary_path = os.path.join(output_dir, f"summary_{timestamp}.json")
    summary = {}
    for name, df in all_results.items():
        if name == "cache_raw":
            continue
        summary[name] = df.to_dict(orient="records")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  {summary_path}")

    print(f"\nBenchmark complete! Results in {output_dir}/")


if __name__ == "__main__":
    main()
