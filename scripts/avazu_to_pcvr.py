"""Convert Avazu CTR CSV to PCVR Parquet format with schema.json.

Memory-optimized version: processes in chunks, writes directly to parquet.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Avazu CTR CSV to PCVR Parquet")
    parser.add_argument("--input", required=True, help="Path to Avazu train.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for Parquet and schema.json")
    parser.add_argument("--max-seq-len", type=int, default=50, help="Max sequence length per user")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows to process per chunk")
    parser.add_argument("--sample", type=int, default=0, help="If > 0, only process N rows (for testing)")
    return parser.parse_args()


def _parse_hour(hour_str: str) -> int:
    """Parse Avazu hour format YYMMDDHH to timestamp."""
    return int(hour_str)


def _build_vocab_dict(values: pd.Series) -> dict[str, int]:
    """Build a vocabulary mapping from string values to integer IDs."""
    unique_vals = values.dropna().astype(str).unique()
    return {val: idx + 1 for idx, val in enumerate(sorted(unique_vals))}


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Avazu CSV: {input_path}")

    # Define feature mappings
    user_features = ["device_ip", "device_model", "device_type", "device_conn_type", "C1"]
    item_features = [
        "site_id", "site_domain", "site_category",
        "app_id", "app_domain", "app_category", "banner_pos",
        "C14", "C15", "C16", "C17", "C18", "C19", "C20", "C21",
    ]
    all_cat_features = ["device_id"] + user_features + item_features

    # First pass: collect all unique values for label encoding (using hash sets)
    print("First pass: collecting unique values for label encoding...")
    unique_values: dict[str, set[str]] = {col: set() for col in all_cat_features}
    total_rows = 0

    nrows = args.sample if args.sample > 0 else None
    for chunk in pd.read_csv(input_path, chunksize=args.chunk_size, nrows=nrows, dtype=str):
        for col in all_cat_features:
            unique_values[col].update(chunk[col].dropna().unique())
        total_rows += len(chunk)
        if total_rows % 5_000_000 == 0:
            print(f"  Processed {total_rows} rows...")

    print(f"Total rows: {total_rows}")

    # Build vocabulary dictionaries
    print("Building vocabulary dictionaries...")
    vocab_dicts: dict[str, dict[str, int]] = {}
    for col in all_cat_features:
        vocab = {val: idx + 1 for idx, val in enumerate(sorted(unique_values[col]))}
        vocab_dicts[col] = vocab
        print(f"  {col}: {len(vocab)} unique values")
        del unique_values[col]  # Free memory

    # Second pass: transform and build sequences
    print("Second pass: transforming data and building sequences...")

    # Pre-compute vocabulary getters
    device_id_vocab = vocab_dicts["device_id"]
    site_id_vocab = vocab_dicts["site_id"]

    # User histories for sequence building
    user_histories: dict[str, list[int]] = defaultdict(list)

    # Output parquet writer
    parquet_path = output_dir / "avazu.parquet"
    writer = None

    processed = 0
    for chunk in pd.read_csv(input_path, chunksize=args.chunk_size, nrows=nrows, dtype=str):
        # Encode categorical features
        chunk["user_id"] = chunk["device_id"].map(device_id_vocab).fillna(0).astype(np.int64)

        for i, col in enumerate(user_features, 1):
            chunk[f"user_int_feats_{i}"] = chunk[col].map(vocab_dicts[col]).fillna(0).astype(np.int64)

        for i, col in enumerate(item_features, 1):
            chunk[f"item_int_feats_{i}"] = chunk[col].map(vocab_dicts[col]).fillna(0).astype(np.int64)

        # Parse timestamp
        chunk["timestamp"] = chunk["hour"].apply(_parse_hour).astype(np.int64)

        # Label: click 0->1, 1->2
        chunk["label_type"] = (chunk["click"].astype(int) + 1).astype(np.int64)

        # Build sequences
        seq_site_ids = []
        seq_timestamps = []

        for _, row in chunk.iterrows():
            user = row["device_id"]
            current_site = int(row["item_int_feats_1"])

            history = user_histories[user][-args.max_seq_len:]
            seq_site_ids.append(list(history))
            seq_timestamps.append([0] * len(history))

            user_histories[user].append(current_site)

        # Build PyArrow arrays for this chunk
        arrays = []
        fields = []

        # Scalar columns
        scalar_cols = [
            ("timestamp", chunk["timestamp"].values),
            ("label_type", chunk["label_type"].values),
            ("user_id", chunk["user_id"].values),
        ]
        for i in range(1, len(user_features) + 1):
            scalar_cols.append((f"user_int_feats_{i}", chunk[f"user_int_feats_{i}"].values))
        for i in range(1, len(item_features) + 1):
            scalar_cols.append((f"item_int_feats_{i}", chunk[f"item_int_feats_{i}"].values))

        for name, data in scalar_cols:
            arrays.append(pa.array(data, type=pa.int64()))
            fields.append(pa.field(name, pa.int64()))

        # List columns (sequences)
        offsets = [0]
        flat_sites = []
        for lst in seq_site_ids:
            flat_sites.extend(lst)
            offsets.append(len(flat_sites))

        site_arr = pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            pa.array(flat_sites, type=pa.int64()),
        )
        arrays.append(site_arr)
        fields.append(pa.field("domain_a_seq_38", pa.list_(pa.int64())))

        offsets_ts = [0]
        flat_ts = []
        for lst in seq_timestamps:
            flat_ts.extend(lst)
            offsets_ts.append(len(flat_ts))

        ts_arr = pa.ListArray.from_arrays(
            pa.array(offsets_ts, type=pa.int32()),
            pa.array(flat_ts, type=pa.int64()),
        )
        arrays.append(ts_arr)
        fields.append(pa.field("domain_a_seq_0", pa.list_(pa.int64())))

        table_chunk = pa.Table.from_arrays(arrays, schema=pa.schema(fields))

        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table_chunk.schema)
        writer.write_table(table_chunk)

        processed += len(chunk)
        if processed % 5_000_000 == 0:
            print(f"  Processed {processed}/{total_rows} rows...")

    if writer is not None:
        writer.close()

    print(f"Parquet written: {parquet_path}")

    # Generate schema.json
    print("Generating schema.json...")
    user_int = []
    for i, col in enumerate(user_features, 1):
        vocab_size = len(vocab_dicts[col]) + 1
        user_int.append([i, vocab_size, 1])

    item_int = []
    for i, col in enumerate(item_features, 1):
        vocab_size = len(vocab_dicts[col]) + 1
        item_int.append([i, vocab_size, 1])

    schema = {
        "user_int": user_int,
        "item_int": item_int,
        "user_dense": [],
        "item_dense": [],
        "seq": {
            "a": {
                "prefix": "domain_a_seq",
                "ts_fid": 0,
                "features": [[0, 1], [38, len(vocab_dicts["site_id"]) + 1]],
            }
        },
    }

    schema_path = output_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Schema written: {schema_path}")

    # Verify
    verify_table = pq.read_table(parquet_path)
    print(f"Verified: {verify_table.num_rows} rows, {verify_table.num_columns} columns")

    print("Done!")


if __name__ == "__main__":
    main()
