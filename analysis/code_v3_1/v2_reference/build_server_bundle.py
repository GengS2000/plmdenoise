"""Build a small, checksummed server archive without data/checkpoint bulk."""
import argparse
import hashlib
import io
from pathlib import Path
import tarfile


def main():
    root = Path(__file__).resolve().parent
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=root.parent / "server_bundle/microbiome_denoise_v2.tar.gz")
    p.add_argument("--include-local-validation", action="store_true")
    args = p.parse_args()
    files = {"code/" + f.name: f for f in root.iterdir()
             if f.is_file() and f.suffix in {".py", ".md", ".json", ".txt"}}
    for folder in ("slurm", "tests"):
        for f in (root / folder).iterdir():
            if f.is_file() and f.suffix in {".py", ".sh", ".sbatch"}:
                files["code/" + f.relative_to(root).as_posix()] = f
    if args.include_local_validation:
        for name in ("validation_smoke_v1", "validation_full_v1", "validation_extended_v1", "validation_holdout_v2"):
            base = root.parent / "tmp" / name
            if base.exists():
                for f in base.rglob("*"):
                    if f.is_file() and f.suffix in {".py", ".json", ".md"} and "__pycache__" not in f.parts:
                        files["local_validation/" + name + "/" + f.relative_to(base).as_posix()] = f
    args.out.parent.mkdir(parents=True, exist_ok=True)
    checksums = []
    with tarfile.open(args.out, "w:gz") as archive:
        for name, path in sorted(files.items()):
            data = path.read_bytes()
            checksums.append(hashlib.sha256(data).hexdigest() + "  " + name)
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mode = 0o755 if path.suffix in {".sh", ".sbatch"} else 0o644
            archive.addfile(info, io.BytesIO(data))
        data = ("\n".join(checksums) + "\n").encode()
        info = tarfile.TarInfo("SHA256SUMS")
        info.size = len(data)
        archive.addfile(info, io.BytesIO(data))
    checksum = hashlib.sha256(args.out.read_bytes()).hexdigest()
    args.out.with_suffix(args.out.suffix + ".sha256").write_text(checksum + "  " + args.out.name + "\n", encoding="utf-8")
    # Read the written archive back and verify every payload, not just file names.
    with tarfile.open(args.out, "r:gz") as archive:
        for line in archive.extractfile("SHA256SUMS").read().decode().splitlines():
            expected, name = line.split("  ", 1)
            actual = hashlib.sha256(archive.extractfile(name).read()).hexdigest()
            if actual != expected:
                raise RuntimeError(f"Archive integrity check failed: {name}")
    print(f"Verified {len(files)} files; {args.out.stat().st_size / 1024:.1f} KiB -> {args.out}")


if __name__ == "__main__":
    main()
