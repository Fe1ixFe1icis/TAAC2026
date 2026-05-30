"""
Merge Amazon, Avazu, and Criteo datasets into a unified schema for multi-dataset training.

Strategy:
- Create a unified schema with non-overlapping feature IDs
- Remap each dataset's features to the unified space
- Convert all datasets to the unified Parquet format
- Missing features are filled with null/empty lists
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm import tqdm


# =============================================================================
# Unified Schema Definition
# =============================================================================

# We design a unified schema that can accommodate all three datasets.
# Feature IDs are allocated in non-overlapping ranges:
#
# User Int Features: fid 1-31 (31 dims total)
#   - Amazon: fid 1 → unified fid 1
#   - Avazu: fid 1-5 → unified fid 1-5
#   - Criteo: fid 1-26 → unified fid 1-26
#   (All fit in 1-26, with room for future expansion)
#
# Item Int Features: fid 1-15 (15 dims total)
#   - Amazon: fid 1 → unified fid 1
#   - Avazu: fid 1-15 → unified fid 1-15
#   - Criteo: fid 1-13 → unified fid 1-13
#
# Sequence Domain 'a': features [0, 38] (timestamp + sideinfo)
#   - All datasets with seq use the same domain_a_seq_0 and domain_a_seq_38
#
# Note: sample_1000_raw has a very different schema with many more features.
# It will be used as validation set with its own schema, NOT merged.

UNIFIED_SCHEMA = {
    "user_int": [
        [1, 4201697, 1],   # Amazon user (also Avazu fid 1, Criteo fid 1)
        [2, 8252, 1],      # Avazu user fid 2
        [3, 6, 1],         # Avazu user fid 3
        [4, 5, 1],         # Avazu user fid 4
        [5, 8, 1],         # Avazu user fid 5
        [6, 1458, 1],      # Criteo user fid 2
        [7, 555, 1],       # Criteo user fid 3
        [8, 193948, 1],    # Criteo user fid 4
        [9, 138800, 1],    # Criteo user fid 5
        [10, 306, 1],      # Criteo user fid 6
        [11, 18, 1],       # Criteo user fid 7
        [12, 11970, 1],    # Criteo user fid 8
        [13, 634, 1],      # Criteo user fid 9
        [14, 4, 1],        # Criteo user fid 10
        [15, 42646, 1],    # Criteo user fid 11
        [16, 5178, 1],     # Criteo user fid 12
        [17, 192772, 1],   # Criteo user fid 13
        [18, 3175, 1],     # Criteo user fid 14
        [19, 27, 1],       # Criteo user fid 15
        [20, 11422, 1],    # Criteo user fid 16
        [21, 181074, 1],   # Criteo user fid 17
        [22, 11, 1],       # Criteo user fid 18
        [23, 4654, 1],     # Criteo user fid 19
        [24, 2031, 1],     # Criteo user fid 20
        [25, 4, 1],        # Criteo user fid 21
        [26, 189656, 1],   # Criteo user fid 22
        [27, 17, 1],       # Criteo user fid 23
        [28, 16, 1],       # Criteo user fid 24
        [29, 59696, 1],    # Criteo user fid 25
        [30, 85, 1],       # Criteo user fid 26
        [31, 6729487, 1],  # Avazu user fid 1 (max vocab, use Avazu's)
    ],
    "item_int": [
        [1, 476003, 1],    # Amazon item (also Avazu fid 1)
        [2, 7746, 1],      # Avazu item fid 2
        [3, 27, 1],        # Avazu item fid 3
        [4, 8553, 1],      # Avazu item fid 4
        [5, 560, 1],       # Avazu item fid 5
        [6, 37, 1],        # Avazu item fid 6
        [7, 8, 1],         # Avazu item fid 7
        [8, 2627, 1],      # Avazu item fid 8
        [9, 9, 1],         # Avazu item fid 9
        [10, 10, 1],       # Avazu item fid 10
        [11, 436, 1],      # Avazu item fid 11
        [12, 5, 1],        # Avazu item fid 12
        [13, 69, 1],       # Avazu item fid 13
        [14, 173, 1],      # Avazu item fid 14
        [15, 61, 1],       # Avazu item fid 15
    ],
    "user_dense": [],
    "item_dense": [],
    "seq": {
        "a": {
            "prefix": "domain_a_seq",
            "ts_fid": 0,
            "features": [
                [0, 1],
                [38, 476003]   # Use Amazon's item vocab size for seq sideinfo
            ]
        }
    }
}

# Dataset-specific feature mappings to unified schema
# Format: {dataset_name: {unified_fid: source_fid}}
# If source_fid is None, the feature is missing in this dataset

DATASET_MAPPINGS = {
    "amazon": {
        "user_int": {1: 1},  # unified fid 1 → source fid 1
        "item_int": {1: 1},
        "seq_a": {0: 0, 38: 38},
    },
    "avazu": {
        "user_int": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5},
        "item_int": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13, 14: 14, 15: 15},
        "seq_a": {0: 0, 38: 38},
    },
    "criteo": {
        "user_int": {
            1: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6, 11: 7, 12: 8, 13: 9,
            14: 10, 15: 11, 16: 12, 17: 13, 18: 14, 19: 15, 20: 16,
            21: 17, 22: 18, 23: 19, 24: 20, 25: 21, 26: 22, 27: 23,
            28: 24, 29: 25, 30: 26,
        },
        "item_int": {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 13: 13},
        "seq_a": {},  # No sequence in Criteo
    },
}


def get_unified_columns() -> list[str]:
    """Get all column names in the unified schema."""
    columns = ["timestamp", "label_type", "user_id"]
    for fid, _vocab, _dim in UNIFIED_SCHEMA["user_int"]:
        columns.append(f"user_int_feats_{fid}")
    for fid, _vocab, _dim in UNIFIED_SCHEMA["item_int"]:
        columns.append(f"item_int_feats_{fid}")
    for domain, config in UNIFIED_SCHEMA["seq"].items():
        for fid, _vocab in config["features"]:
            columns.append(f"{config['prefix']}_{fid}")
    return columns


def create_empty_list_array(target_length: int, value_type: pa.DataType = pa.int64()) -> pa.ListArray:
    """Create an empty list array of given length."""
    # All lists are empty, so offsets are all 0
    offsets = pa.array([0] * (target_length + 1), type=pa.int32())
    values = pa.array([], type=value_type)
    return pa.ListArray.from_arrays(offsets, values)


def remap_parquet_dataset(
    source_parquet_path: str,
    output_parquet_path: str,
    dataset_name: str,
    batch_size: int = 65536,
) -> None:
    """Remap a source dataset to the unified schema format."""
    mapping = DATASET_MAPPINGS[dataset_name]
    unified_columns = get_unified_columns()

    source_file = pq.ParquetFile(source_parquet_path)
    source_schema = source_file.schema_arrow
    source_columns = set(source_schema.names)

    print(f"\nRemapping {dataset_name}:")
    print(f"  Source: {source_parquet_path}")
    print(f"  Output: {output_parquet_path}")
    print(f"  Source rows: {source_file.metadata.num_rows}")
    print(f"  Unified columns: {len(unified_columns)}")

    # Prepare writers
    output_path = Path(output_parquet_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    writer = None
    total_rows = 0

    for batch in tqdm(source_file.iter_batches(batch_size=batch_size), desc=f"Processing {dataset_name}"):
        row_count = batch.num_rows
        total_rows += row_count

        # Build unified batch column by column
        unified_arrays = []

        for col_name in unified_columns:
            if col_name in ["timestamp", "label_type", "user_id"]:
                # Core columns - copy directly
                if col_name in source_columns:
                    unified_arrays.append(batch.column(batch.schema.names.index(col_name)))
                else:
                    raise ValueError(f"Missing required column: {col_name}")

            elif col_name.startswith("user_int_feats_"):
                fid = int(col_name.split("_")[-1])
                source_fid = mapping["user_int"].get(fid)
                if source_fid is not None:
                    source_col = f"user_int_feats_{source_fid}"
                    if source_col in source_columns:
                        unified_arrays.append(batch.column(batch.schema.names.index(source_col)))
                    else:
                        unified_arrays.append(pa.array([0] * row_count, type=pa.int64()))
                else:
                    unified_arrays.append(pa.array([0] * row_count, type=pa.int64()))

            elif col_name.startswith("item_int_feats_"):
                fid = int(col_name.split("_")[-1])
                source_fid = mapping["item_int"].get(fid)
                if source_fid is not None:
                    source_col = f"item_int_feats_{source_fid}"
                    if source_col in source_columns:
                        unified_arrays.append(batch.column(batch.schema.names.index(source_col)))
                    else:
                        unified_arrays.append(pa.array([0] * row_count, type=pa.int64()))
                else:
                    unified_arrays.append(pa.array([0] * row_count, type=pa.int64()))

            elif col_name.startswith("domain_a_seq_"):
                fid = int(col_name.split("_")[-1])
                source_fid = mapping.get("seq_a", {}).get(fid)
                if source_fid is not None:
                    source_col = f"domain_a_seq_{source_fid}"
                    if source_col in source_columns:
                        unified_arrays.append(batch.column(batch.schema.names.index(source_col)))
                    else:
                        unified_arrays.append(create_empty_list_array(row_count))
                else:
                    unified_arrays.append(create_empty_list_array(row_count))

            else:
                raise ValueError(f"Unknown unified column: {col_name}")

        unified_batch = pa.RecordBatch.from_arrays(unified_arrays, unified_columns)

        if writer is None:
            writer = pq.ParquetWriter(output_parquet_path, unified_batch.schema)
        writer.write_batch(unified_batch)

    if writer:
        writer.close()

    print(f"  Written {total_rows} rows to unified format")


def main():
    base_dir = Path("D:/trae_proj/taac2026/win_version/data")
    output_dir = base_dir / "merged"
    output_dir.mkdir(exist_ok=True)

    # Save unified schema
    schema_path = output_dir / "schema.json"
    with open(schema_path, "w") as f:
        json.dump(UNIFIED_SCHEMA, f, indent=2)
    print(f"Saved unified schema to {schema_path}")

    # Process each dataset
    datasets = [
        ("amazon", base_dir / "amazon-ratings-electronics/pcvr/amazon.parquet", output_dir / "amazon_unified.parquet"),
        ("avazu", base_dir / "avazu-ctr-prediction/pcvr/avazu.parquet", output_dir / "avazu_unified.parquet"),
        ("criteo", base_dir / "criteo/pcvr/criteo.parquet", output_dir / "criteo_unified.parquet"),
    ]

    for dataset_name, source_path, output_path in datasets:
        remap_parquet_dataset(str(source_path), str(output_path), dataset_name)

    print("\n" + "=" * 60)
    print("All datasets remapped successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
