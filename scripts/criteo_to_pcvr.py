"""Convert Criteo train.txt to PCVR Parquet format.

Criteo has no sequence structure. All features are NS features.
- Col0: label (0/1) -> label_type (1/2)
- Col1-13: continuous features -> log2 bucketed to discrete
- Col14-39: categorical features -> filter low freq (<10) to OOV
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Criteo to PCVR Parquet")
    parser.add_argument("--input", required=True, help="Path to train.txt")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--chunk-size", type=int, default=1_000_000, help="Rows per chunk")
    parser.add_argument("--sample", type=int, default=0, help="If > 0, only process N rows")
    parser.add_argument("--min-freq", type=int, default=10, help="Min freq for categorical features")
    return parser.parse_args()


def log2_bucket(value: float | None) -> int:
    """Log2 bucketing for continuous features."""
    if value is None or pd.isna(value) or value <= 0:
        return 0
    return int(np.log2(float(value))) + 1


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading Criteo: {input_path}")

    nrows = args.sample if args.sample > 0 else None

    # First pass: compute continuous feature bucket boundaries + categorical vocabularies
    print("First pass: computing bucket boundaries and categorical vocabularies...")

    cont_mins = [float('inf')] * 13
    cont_maxs = [0.0] * 13
    cat_counters: list[Counter] = [Counter() for _ in range(26)]
    total_rows = 0

    for chunk in pd.read_csv(input_path, sep='\t', header=None, chunksize=args.chunk_size,
                             nrows=nrows, dtype=str):
        # Continuous features (cols 1-13)
        for i in range(13):
            col_idx = i + 1
            vals = pd.to_numeric(chunk[col_idx], errors='coerce').dropna()
            if len(vals) > 0:
                cont_mins[i] = min(cont_mins[i], vals.min())
                cont_maxs[i] = max(cont_maxs[i], vals.max())

        # Categorical features (cols 14-39)
        for i in range(26):
            col_idx = i + 14
            cat_counters[i].update(chunk[col_idx].dropna().astype(str).values)

        total_rows += len(chunk)
        if total_rows % 1_000_000 == 0:
            print(f"  Processed {total_rows} rows...")

    print(f"Total rows: {total_rows}")

    # Build vocabularies for categorical features (filter low freq)
    print("Building categorical vocabularies...")
    cat_vocabs: list[dict[str, int]] = []
    for i, counter in enumerate(cat_counters):
        # Keep only values with freq >= min_freq
        filtered = {val: count for val, count in counter.items() if count >= args.min_freq}
        vocab = {val: idx + 1 for idx, val in enumerate(sorted(filtered.keys()))}
        cat_vocabs.append(vocab)
        print(f"  Cat{i}: {len(counter)} unique -> {len(vocab)} after freq>={args.min_freq} filter")

    # Build bucket boundaries for continuous features
    print("Continuous feature stats:")
    cont_bucket_counts: list[int] = []
    for i in range(13):
        if cont_maxs[i] > 0:
            max_bucket = log2_bucket(cont_maxs[i])
            cont_bucket_counts.append(max_bucket + 1)
            print(f"  Cont{i}: min={cont_mins[i]:.2f}, max={cont_maxs[i]:.2f}, buckets={max_bucket + 1}")
        else:
            cont_bucket_counts.append(1)
            print(f"  Cont{i}: all missing, buckets=1")

    # Second pass: transform and write
    print("Second pass: transforming and writing...")

    parquet_path = output_dir / "criteo.parquet"
    writer = None
    processed = 0

    for chunk in pd.read_csv(input_path, sep='\t', header=None, chunksize=args.chunk_size,
                             nrows=nrows, dtype=str):
        row_count = len(chunk)

        # Label: 0->1, 1->2
        label_type = chunk[0].astype(int) + 1

        # Continuous features -> log2 buckets
        cont_encoded = []
        for i in range(13):
            col_idx = i + 1
            vals = pd.to_numeric(chunk[col_idx], errors='coerce')
            buckets = vals.apply(log2_bucket).fillna(0).astype(np.int64).values
            cont_encoded.append(buckets)

        # Categorical features -> vocab index (OOV -> 0)
        cat_encoded = []
        for i in range(26):
            col_idx = i + 14
            vocab = cat_vocabs[i]
            encoded = chunk[col_idx].astype(str).map(vocab).fillna(0).astype(np.int64).values
            cat_encoded.append(encoded)

        # Build PyArrow arrays
        arrays = []
        fields = []

        # timestamp (dummy)
        arrays.append(pa.array(np.zeros(row_count, dtype=np.int64), type=pa.int64()))
        fields.append(pa.field("timestamp", pa.int64()))

        # label_type
        arrays.append(pa.array(label_type.values, type=pa.int64()))
        fields.append(pa.field("label_type", pa.int64()))

        # user_id (dummy, use row index)
        arrays.append(pa.array(np.arange(processed, processed + row_count, dtype=np.int64), type=pa.int64()))
        fields.append(pa.field("user_id", pa.int64()))

        # user_int_feats: categorical features (26) + continuous bucketed (13) = 39
        # Map: user_int = cat features, item_int = cont features
        for i in range(26):
            arrays.append(pa.array(cat_encoded[i], type=pa.int64()))
            fields.append(pa.field(f"user_int_feats_{i+1}", pa.int64()))

        for i in range(13):
            arrays.append(pa.array(cont_encoded[i], type=pa.int64()))
            fields.append(pa.field(f"item_int_feats_{i+1}", pa.int64()))

        table_chunk = pa.Table.from_arrays(arrays, schema=pa.schema(fields))

        if writer is None:
            writer = pq.ParquetWriter(parquet_path, table_chunk.schema)
        writer.write_table(table_chunk)

        processed += row_count
        if processed % 1_000_000 == 0:
            print(f"  Written {processed}/{total_rows} rows...")

    if writer is not None:
        writer.close()

    print(f"Parquet written: {parquet_path}")

    # Generate schema.json
    user_int = []
    for i in range(26):
        vocab_size = len(cat_vocabs[i]) + 1
        user_int.append([i + 1, vocab_size, 1])

    item_int = []
    for i in range(13):
        item_int.append([i + 1, cont_bucket_counts[i], 1])

    schema = {
        "user_int": user_int,
        "item_int": item_int,
        "user_dense": [],
        "item_dense": [],
        "seq": {},
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
