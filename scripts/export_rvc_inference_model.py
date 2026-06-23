from __future__ import annotations
from pathlib import Path
import json
import sys
import torch


def build_config(config_json: dict) -> list:
    data = config_json["data"]
    model = config_json["model"]
    return [
        data["filter_length"] // 2 + 1,
        32,
        model["inter_channels"],
        model["hidden_channels"],
        model["filter_channels"],
        model["n_heads"],
        model["n_layers"],
        model["kernel_size"],
        model["p_dropout"],
        model["resblock"],
        model["resblock_kernel_sizes"],
        model["resblock_dilation_sizes"],
        model["upsample_rates"],
        model["upsample_initial_channel"],
        model["upsample_kernel_sizes"],
        model["spk_embed_dim"],
        model["gin_channels"],
        data["sampling_rate"],
    ]


def main() -> None:
    if len(sys.argv) != 4:
        print("usage: python scripts/export_rvc_inference_model.py <training_ckpt> <config_json> <output_pth>")
        raise SystemExit(1)

    ckpt_path = Path(sys.argv[1])
    config_path = Path(sys.argv[2])
    output_path = Path(sys.argv[3])

    ckpt = torch.load(ckpt_path, map_location="cpu")
    weights = ckpt["model"] if "model" in ckpt else ckpt
    config_json = json.loads(config_path.read_text())

    exported = {
        "weight": {k: v.half() for k, v in weights.items() if "enc_q" not in k},
        "config": build_config(config_json),
        "info": f"Exported from {ckpt_path.name}",
        "name": output_path.stem,
        "sr": str(config_json["data"]["sampling_rate"] // 1000) + "k",
        "f0": 1 if config_json["train"].get("f0", True) else 0,
        "version": config_json["train"].get("version", "v2"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(exported, output_path)
    print(output_path)


if __name__ == "__main__":
    main()
