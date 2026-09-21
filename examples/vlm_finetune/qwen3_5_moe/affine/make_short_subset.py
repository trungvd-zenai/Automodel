# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Carve a short-row subset out of the pre-filtered corpus for the fast packing rung.

`pack_size` cannot be lowered below the pre-filter cap on the full corpus. The VLM neat
packer books an over-long row into a bin at a CLAMPED planning length
(``est_len = min(est_len, knapsack_capacity)``) and only discovers the real length in
``__getitem__``, where it drops the sample with a bare ``logger.warning``; a bin that held
only that sample then yields a padding-only pack. Unlike the LLM packer, which raises,
nothing fails. So a small `pack_size` against 40k rows silently trains on padding and makes
``num_label_tokens`` meaningless -- which is the one number the smoke rung reads.

This keeps rows short enough that no row can be dropped at ``pack_size``, while staying long
enough that several documents share a pack -- cross-document attention and the GatedDeltaNet
state reset are the whole point of the rung, and one-document packs would not exercise them.

Usage::

    python make_short_subset.py --src data/v5_130k_filtered --out data/v5_130k_short \
        --max-tokens 1000
"""

import argparse
import os

import pyarrow.parquet as pq


def main() -> None:
    """Write train/val Parquet files containing only rows at or below the token cap."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default="data/v5_130k_filtered")
    parser.add_argument("--out", default="data/v5_130k_short")
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1000,
        help="Keep rows whose rendered length is at or below this. At pack_size 4096 a cap of "
        "1000 puts about four documents in every pack with no possibility of a drop.",
    )
    parser.add_argument("--train-rows", type=int, default=4096, help="Cap on kept train rows.")
    parser.add_argument("--val-rows", type=int, default=64, help="Cap on kept validation rows.")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for split, cap in (("train", args.train_rows), ("val", args.val_rows)):
        table = pq.read_table(os.path.join(args.src, f"{split}.parquet"))
        if "n_tokens" not in table.column_names:
            raise SystemExit(f"{split}.parquet has no n_tokens column; re-run prefilter.py")
        lengths = table.column("n_tokens").to_pylist()
        keep = [i for i, n in enumerate(lengths) if n <= args.max_tokens][:cap]
        if not keep:
            raise SystemExit(f"no {split} row is <= {args.max_tokens} tokens")
        subset = table.take(keep)
        kept_lengths = subset.column("n_tokens").to_pylist()
        path = os.path.join(args.out, f"{split}.parquet")
        pq.write_table(subset, path)
        print(
            f"{split}: {len(keep):,} of {table.num_rows:,} rows <= {args.max_tokens} tokens "
            f"(max kept {max(kept_lengths):,}, mean {sum(kept_lengths) / len(kept_lengths):.0f}) -> {path}"
        )


if __name__ == "__main__":
    main()
