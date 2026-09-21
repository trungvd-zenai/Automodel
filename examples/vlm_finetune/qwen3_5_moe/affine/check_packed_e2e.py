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
"""End-to-end packed vs unpacked per-document loss on the real model and the real packer.

`check_packed_parity.py` proves the layers are document-isolated on a small stack.
This check closes the remaining gap: it drives the recipe's own dataloader (pre-filtered
corpus -> knapsack -> `neat_packed_vlm_collater` -> `last_turn_label_hook`), forwards
each pack once through the real 35B checkpoint, and compares the per-document
cross-entropy against the same document forwarded ALONE with no ``position_ids`` -- the
model derives its own positions exactly as unpacked training does. A control run drops the
pack's ``position_ids`` to show the test can see a positions bug at all.

Read the result as a spread, not a threshold: document 1 of every pack sees exactly the
solo context, so its relative difference (0.3-0.5% in fp32 on 2 x B300) is the numerical
floor -- a 4096-token forward tiles GEMMs differently from a 999-token one, and MoE
routing amplifies that. Later documents must sit in the same band with no consistent
sign. Measured 2026-09-21: worst 1.07% fp32, signed mean -0.0009.

Usage (single GPU, from the repo root with the training venv)::

    torchrun --nproc-per-node 1 examples/vlm_finetune/qwen3_5_moe/affine/check_packed_e2e.py \\
        --attn te --dtype float32 --packs 2
"""

import argparse
import os
import tempfile

import torch
import torch.distributed as dist
import torch.nn.functional as F
import yaml


def main() -> None:
    """Forward real packs and their documents alone; print per-document CE differences."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attn", default="te", choices=["te", "sdpa"])
    parser.add_argument("--dtype", default="float32", choices=["bfloat16", "float32"])
    parser.add_argument("--packs", type=int, default=2, help="Packs to check.")
    parser.add_argument("--config", default="examples/vlm_finetune/qwen3_5_moe/qwen3_6_35b_4node_ep8_packed.yaml")
    parser.add_argument("--model", default="Qwen/Qwen3.6-35B-A3B")
    parser.add_argument(
        "--data",
        default="data/v5_130k_short",
        help="Directory with train/val.parquet. The short subset keeps every row under the "
        "4096 pack used here, so nothing is dropped and each pack holds ~4 documents.",
    )
    parser.add_argument("--pack-size", type=int, default=4096)
    args = parser.parse_args()
    dtype = getattr(torch, args.dtype)

    dist.init_process_group("nccl")
    torch.cuda.set_device(0)

    from nemo_automodel import NeMoAutoModelForImageTextToText
    from nemo_automodel._transformers.utils import resolve_get_rope_index
    from nemo_automodel.components.config.loader import load_yaml_config
    from nemo_automodel.components.models.common import BackendConfig
    from nemo_automodel.recipes._typed_config import RecipeConfig

    # The recipe config, narrowed to one GPU and the short corpus.
    raw = yaml.safe_load(open(args.config))
    raw["dataset"]["path_or_dataset"] = os.path.join(args.data, "train.parquet")
    raw["validation_dataset"]["path_or_dataset"] = os.path.join(args.data, "val.parquet")
    raw["packed_sequence"].update(pack_size=args.pack_size, collate_max_length=args.pack_size, drop_long_samples=True)
    raw["model"]["backend"]["attn"] = args.attn
    raw["dataloader"].update(num_workers=0, persistent_workers=False, shuffle=False)
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(raw, tmp)

    # torch_mm/torch: DeepEP needs an EP group, and this is one process.
    backend = BackendConfig(
        attn=args.attn, linear="torch", rms_norm="torch_fp32", rope_fusion=False, experts="torch_mm", dispatcher="torch"
    )
    model = (
        NeMoAutoModelForImageTextToText.from_pretrained(
            args.model,
            torch_dtype=dtype,
            backend=backend,
            text_config={"output_hidden_states": True, "use_cache": False},
        )
        .cuda()
        .eval()
    )
    print("model loaded:", type(model).__name__, flush=True)

    dl_cfg = RecipeConfig(load_yaml_config(tmp.name)).vlm_dataloader
    built = dl_cfg.build(
        pretrained_model_name_or_path=args.model,
        dp_rank=0,
        dp_world_size=1,
        batch_size=1,
        get_rope_index=resolve_get_rope_index(model),
        packing_attn_implementation=args.attn,
    )
    loader = built[0] if isinstance(built, tuple) else getattr(built, "dataloader", built)

    def per_token_ce(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-token CE in fp32 with -100 ignored; returns ``(ce, valid)`` over the sequence."""
        valid = labels != -100
        ce = F.cross_entropy(logits.float(), labels.clamp_min(0), reduction="none")
        return ce * valid, valid

    def forward_logits(batch: dict) -> torch.Tensor:
        with torch.no_grad():
            return model(**batch).logits[0]

    worst = 0.0
    worst_ctrl = 0.0
    signed: list[float] = []
    for pack_idx, batch in enumerate(loader):
        if pack_idx >= args.packs:
            break
        batch = {k: (v.cuda() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        labels = batch.pop("labels")[0]
        doc = batch["_packed_seq_ids"][0]
        docs = sorted(set(doc.tolist()) - {0})
        if pack_idx == 0:
            pos = batch.get("position_ids")
            starts = [int((doc == d).nonzero()[0]) for d in docs]
            axis0 = [int(pos[0, 0, s]) if pos.ndim == 3 else int(pos[0, s]) for s in starts]
            print("position_ids shape:", tuple(pos.shape), "| at document starts:", axis0)

        p_ce, _ = per_token_ce(forward_logits(batch), labels)
        c_ce, _ = per_token_ce(forward_logits({k: v for k, v in batch.items() if k != "position_ids"}), labels)

        print(f"\n=== pack {pack_idx}: {int((doc > 0).sum())} real tokens, docs {docs} ===")
        for d in docs:
            m = doc == d
            ids = batch["input_ids"][0][m][None]
            s_ce, s_valid = per_token_ce(
                forward_logits({"input_ids": ids, "attention_mask": torch.ones_like(ids)}), labels[m]
            )
            n = int(s_valid.sum())
            if n == 0:
                continue
            solo, packed, control = (x.sum().item() / n for x in (s_ce, p_ce[m], c_ce[m]))
            rel, rel_c = abs(packed - solo) / solo, abs(control - solo) / solo
            worst, worst_ctrl = max(worst, rel), max(worst_ctrl, rel_c)
            signed.append(packed - solo)
            print(
                f"doc {d:2d} len {int(m.sum()):5d} sup {n:4d} | solo {solo:.4f} packed {packed:.4f} "
                f"rel {rel:.2e} | no-pos control {control:.4f} rel {rel_c:.2e}"
            )

    print(f"\nWORST packed-vs-solo relative CE diff: {worst:.3e}")
    print(f"WORST no-position_ids control:          {worst_ctrl:.3e}")
    print(f"signed mean (packed - solo):            {sum(signed) / len(signed):+.4f} over {len(signed)} documents")
    os.unlink(tmp.name)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
