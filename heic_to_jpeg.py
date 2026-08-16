from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pillow_heif
from PIL import Image

pillow_heif.register_heif_opener()

HEIC_EXTS = {".heic", ".heif"}


def convert(src: Path, dst: Path, quality: int) -> None:
    img = Image.open(src)
    save_kwargs: dict = {"quality": quality, "optimize": True}
    exif = img.info.get("exif")
    if exif:
        save_kwargs["exif"] = exif
    img.convert("RGB").save(dst, "JPEG", **save_kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description="Convert HEIC/HEIF images to JPEG.")
    ap.add_argument("--input", default="face_images",
                    help="folder to scan (default: face_images)")
    ap.add_argument("--quality", type=int, default=95,
                    help="JPEG quality 1-100 (default: 95)")
    ap.add_argument("--overwrite", action="store_true",
                    help="overwrite existing JPEG files")
    ap.add_argument("--delete-originals", action="store_true",
                    help="remove HEIC files after successful conversion")
    args = ap.parse_args()

    src_dir = Path(args.input)
    if not src_dir.is_dir():
        print(f"input folder not found: {src_dir}", file=sys.stderr)
        return 1

    heics = sorted(p for p in src_dir.iterdir() if p.suffix.lower() in HEIC_EXTS)
    if not heics:
        print(f"no HEIC files in {src_dir}")
        return 0

    converted = skipped = failed = 0
    for i, src in enumerate(heics, 1):
        dst = src.with_suffix(".jpg")
        if dst.exists() and not args.overwrite:
            print(f"  [{i}/{len(heics)}] SKIP exists: {dst.name}")
            skipped += 1
            continue
        try:
            convert(src, dst, args.quality)
        except Exception as e:
            print(f"  [{i}/{len(heics)}] FAIL {src.name}: {e}", file=sys.stderr)
            failed += 1
            continue
        if args.delete_originals:
            src.unlink()
        print(f"  [{i}/{len(heics)}] OK  {src.name} → {dst.name}")
        converted += 1

    print(f"done: {converted} converted, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
