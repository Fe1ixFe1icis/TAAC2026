"""Convert Amazon Electronics Ratings CSV to PCVR Parquet format.

Rating >= 4 -> positive (label_type=2), else negative (label_type=1).
Builds user interaction sequences from historical ratings.
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
    parser = argparse.ArgumentParser(description="Convert Amazon Ratings to PCVR Parquet")
    parser.add_argument("--input", required=True, help="Path to ratings CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--max-seq-len", type=int, default=50, help="Max sequence length")
    parser.add_argument("--chunk-size", type=int, default=500_000, help="Rows per chunk")
    parser.add_argument("--sample", type=int, default=0, help="If > 0, only process N rows")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Amazon CSV: {input_path}")

    # First pass: collect vocabularies
    print("First pass: collecting unique user/item IDs...")
    user_ids: set[str] = set()
    item_ids: set[str] = set()
    total_rows = 0

    nrows = args.sample if args.sample > 0 else None
    for chunk in pd.read_csv(input_path, header=None, names=["user", "item", "rating", "ts"],
                             chunksize=args.chunk_size, nrows=nrows, dtype=str):
        user_ids.update(chunk["user"].dropna().unique())
        item_ids.update(chunk["item"].dropna().unique())
        total_rows += len(chunk)
        if total_rows % 1_000_000 == 0:
            print(f"  Processed {total_rows} rows...")

    print(f"Total rows: {total_rows}")
    print(f"Unique users: {len(user_ids)}, Unique items: {len(item_ids)}")

    # Build vocab
    user_vocab = {val: idx + 1 for idx, val in enumerate(sorted(user_ids))}
    item_vocab = {val: idx + 1 for idx, val in enumerate(sorted(item_ids))}
    del user_ids, item_ids

    # Second pass: sort by user and timestamp, build sequences
    print("Second pass: sorting and building sequences...")

    # Read all data, sort by (user, timestamp)
    print("  Loading full dataset for sorting...")
    df = pd.read_csv(input_path, header=None, names=["user", "item", "rating", "ts"],
                     nrows=nrows, dtype=str)
    df["ts"] = df["ts"].astype(np.int64)
    df["rating"] = df["rating"].astype(float)
    df = df.sort_values(["user", "ts"]).reset_index(drop=True)

    print(f"  Sorted {len(df)} rows")

    # Build sequences per user
    print("  Building user histories...")
    user_histories: dict[str, list[int]] = defaultdict(list)

    parquet_path = output_dir / "amazon.parquet"
    writer = None

    # Process in chunks to avoid memory issues
    chunk_size = args.chunk_size
    for start in range(0, len(df), chunk_size):
        end = min(start + chunk_size, len(df))
        chunk = df.iloc[start:end]

        seq_item_ids = []
        seq_timestamps = []

        for _, row in chunk.iterrows():
            user = row["user"]
            current_item = item_vocab.get(row["item"], 0)

            history = user_histories[user][-args.max_seq_len:]
            seq_item_ids.append(list(history))
            seq_timestamps.append([0] * len(history))

            user_histories[user].append(current_item)

        # Encode features
        user_ids_enc = chunk["user"].map(user_vocab).fillna(0).astype(np.int64).values
        item_ids_enc = chunk["item"].map(item_vocab).fillna(0).astype(np.int64).values
        timestamps = chunk["ts"].astype(np.int64).values
        # Rating >= 4 -> label_type=2 (positive), else 1 (negative)
        label_type = (chunk["rating"] >= 4).astype(int) + 1
        label_type = label_type.astype(np.int64).values

        # Build PyArrow arrays
        arrays = []
        fields = []

        scalar_cols = [
            ("timestamp", timestamps),
            ("label_type", label_type),
            ("user_id", user_ids_enc),
            ("user_int_feats_1", user_ids_enc),
            ("item_int_feats_1", item_ids_enc),
        ]

        for name, data in scalar_cols:
            arrays.append(pa.array(data, type=pa.int64()))
            fields.append(pa.field(name, pa.int64()))

        # Sequence columns
        offsets = [0]
        flat_items = []
        for lst in seq_item_ids:
            flat_items.extend(lst)
            offsets.append(len(flat_items))

        item_arr = pa.ListArray.from_arrays(
            pa.array(offsets, type=pa.int32()),
            pa.array(flat_items, type=pa.int64()),
        )
        arrays.append(item_arr)
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

        if end % 1_000_000 == 0:
            print(f"  Written {end}/{len(df)} rows...")

    if writer is not None:
        writer.close()

    print(f"Parquet written: {parquet_path}")

    # Generate schema.json
    schema = {
        "user_int": [[1, len(user_vocab) + 1, 1]],
        "item_int": [[1, len(item_vocab) + 1, 1]],
        "user_dense": [],
        "item_dense": [],
        "seq": {
            "a": {
                "prefix": "domain_a_seq",
                "ts_fid": 0,
                "features": [[0, 1], [38, len(item_vocab) + 1]],
            }
        },
    }

    schema_path = output_dir / "schema.json"
    schema_path.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"Schema written: {schema_path}")

    # Verify
    verify_table = pq.read_table(parquet_path)
    print(f"Verified: {verify_table.num_rows} rows, {verify_table.num_columns} columns")

    # Print label distribution
    pos_rate = (df["rating"] >= 4).mean()
    print(f"Positive rate (rating>=4): {pos_rate:.4f}")

    print("Done!")


if __name__ == "__main__":
    main()
