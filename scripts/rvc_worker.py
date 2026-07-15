#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from scipy.io import wavfile


def _write(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def main() -> int:
    repo = Path(os.environ["BEAM_RVC_REPO"]).resolve()
    os.chdir(repo)
    sys.path.append(str(repo))
    os.environ.setdefault("rmvpe_root", str(repo / "assets" / "rmvpe"))
    os.environ.setdefault("index_root", str(repo / "logs"))
    os.environ.setdefault("outside_index_root", str(repo / "assets" / "indices"))
    load_dotenv()

    from configs.config import Config, CPUConfig
    from infer.modules.vc import VC

    device = os.environ.get("BEAM_RVC_DEVICE", "cpu")
    is_half = os.environ.get("BEAM_RVC_IS_HALF", "false").lower() == "true"
    if device == "cpu":
        config = CPUConfig()
    else:
        config = Config()
        config.device = device
        config.is_half = is_half
    Config.use_insecure_load()

    vc = VC(config)
    current_model = ""
    _write({"ready": True, "device": config.device, "isHalf": config.is_half})

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            if request.get("command") == "shutdown":
                _write({"ok": True, "shutdown": True})
                return 0

            model_name = request["modelName"]
            os.environ["weight_root"] = request["weightRoot"]
            if model_name != current_model:
                vc.get_vc(model_name)
                current_model = model_name

            info, wav_opt = vc.vc_single(
                0,
                request["inputPath"],
                int(request.get("pitchShift", 0)),
                None,
                request.get("f0Method", "pm"),
                request.get("indexPath") or "",
                None,
                float(request.get("indexRate", 0.0)),
                int(request.get("filterRadius", 3)),
                int(request.get("resampleSr", 0)),
                float(request.get("rmsMixRate", 1.0)),
                float(request.get("protect", 0.33)),
            )
            if wav_opt is None:
                raise RuntimeError(info)
            wavfile.write(request["outputPath"], wav_opt[0], wav_opt[1])
            _write({"ok": True, "info": info})
        except Exception as error:
            _write({"ok": False, "error": str(error), "traceback": traceback.format_exc()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
