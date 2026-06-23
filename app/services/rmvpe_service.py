from pathlib import Path


class RMVPEService:
    def extract_f0(self, vocals_wav: Path, workdir: Path) -> Path:
        # TODO: replace with actual RMVPE invocation
        # expected output: frame-level F0 file such as .npy
        output_path = workdir / "f0.npy"
        output_path.write_bytes(b"")
        return output_path
