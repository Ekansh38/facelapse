from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker, FaceLandmarkerOptions, RunningMode,
)
from PIL import ExifTags, Image

RIGHT_IRIS_CENTER = 468  # right eye
LEFT_IRIS_CENTER = 473   # left eye
LEFT_EYE_CORNERS = (362, 263)
RIGHT_EYE_CORNERS = (33, 133)

DEFAULT_MODEL = Path(__file__).with_name("face_landmarker.task")

# get the date and time for chronological order
_EXIF_TAGS = {v: k for k, v in ExifTags.TAGS.items()}
_DT_ORIG = _EXIF_TAGS["DateTimeOriginal"]
_DT_DIG = _EXIF_TAGS["DateTimeDigitized"]
_DT = _EXIF_TAGS["DateTime"]


def natural_key(name: str) -> tuple:
    return tuple(int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name))


def read_capture_date(path: Path) -> datetime:
    try:
        exif = Image.open(path).getexif()
        raw = exif.get(_DT_ORIG) or exif.get(_DT_DIG) or exif.get(_DT)
        if raw:
            return datetime.strptime(raw, "%Y:%m:%d %H:%M:%S")
    except Exception:
        pass
    return datetime.fromtimestamp(path.stat().st_mtime)


def detect_eyes(landmarker: FaceLandmarker, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    h, w = img_bgr.shape[:2]
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB,
                      data=cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    result = landmarker.detect(mp_img)
    if not result.face_landmarks:
        return None
    lms = result.face_landmarks[0]

    def pt(i: int) -> np.ndarray:
        return np.array([lms[i].x * w, lms[i].y * h], dtype=np.float64)

    if len(lms) > max(LEFT_IRIS_CENTER, RIGHT_IRIS_CENTER):
        return pt(LEFT_IRIS_CENTER), pt(RIGHT_IRIS_CENTER)
    left = (pt(LEFT_EYE_CORNERS[0]) + pt(LEFT_EYE_CORNERS[1])) / 2
    right = (pt(RIGHT_EYE_CORNERS[0]) + pt(RIGHT_EYE_CORNERS[1])) / 2
    return left, right


def align(img_bgr: np.ndarray, left_eye: np.ndarray, right_eye: np.ndarray,
          out_w: int, out_h: int, eye_y_frac: float, eye_dist_frac: float) -> np.ndarray | None:
    target_dist = out_w * eye_dist_frac
    target_y = out_h * eye_y_frac
    cx = out_w / 2
    dst = np.array([[cx + target_dist / 2, target_y],   # left eye  → image-right
                    [cx - target_dist / 2, target_y]],  # right eye → image-left
                   dtype=np.float64)
    src = np.stack([left_eye, right_eye])
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)
    if M is None:
        return None
    return cv2.warpAffine(img_bgr, M, (out_w, out_h),
                          flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_CONSTANT,
                          borderValue=(0, 0, 0))


def caption_layers(dates: list[datetime], fade_frames: int) -> list[list[tuple[str, float]]]:
    labels = [d.strftime("%B %Y") for d in dates]
    n = len(labels)
    prev_run = [0] * n
    next_run = [0] * n
    for i in range(1, n):
        prev_run[i] = 0 if labels[i] != labels[i - 1] else prev_run[i - 1] + 1
    for i in range(n - 2, -1, -1):
        next_run[i] = 0 if labels[i] != labels[i + 1] else next_run[i + 1] + 1

    half = fade_frames / 2
    L = max(fade_frames, 1)
    layers: list[list[tuple[str, float]]] = []
    for i in range(n):
        pairs: list[tuple[str, float]] = []
        cur = 1.0
        # Near the start of this run and a real previous run exists.
        if prev_run[i] < half and i - prev_run[i] - 1 >= 0:
            weight_cur = 0.5 + (prev_run[i] + 0.5) / L
            pairs.append((labels[i - prev_run[i] - 1], 1.0 - weight_cur))
            cur = min(cur, weight_cur)
        # Near the end of this run and a real next run exists.
        if next_run[i] < half and i + next_run[i] + 1 < n:
            weight_cur = 0.5 + (next_run[i] + 0.5) / L
            pairs.append((labels[i + next_run[i] + 1], 1.0 - weight_cur))
            cur = min(cur, weight_cur)
        pairs.append((labels[i], cur))
        layers.append(pairs)
    return layers


def draw_caption(frame: np.ndarray, text: str, alpha: float) -> None:
    if alpha <= 0:
        return
    h, w = frame.shape[:2]
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = h / 700  # ~46px cap height at 1080p
    thick = max(1, int(round(h / 540)))
    (tw, _), _ = cv2.getTextSize(text, font, scale, thick)
    x = (w - tw) // 2
    y = int(h * 0.94)

    overlay = frame.copy()
    cv2.putText(overlay, text, (x + 2, y + 2), font, scale, (0, 0, 0),
                thick + 2, cv2.LINE_AA)
    cv2.putText(overlay, text, (x, y), font, scale, (255, 255, 255),
                thick, cv2.LINE_AA)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="face_images", help="folder of source images")
    ap.add_argument("--output", default="output.mp4", help="output video path")
    ap.add_argument("--width", type=int, default=1920, help="output width in pixels")
    ap.add_argument("--height", type=int, default=1080, help="output height in pixels")
    timing = ap.add_mutually_exclusive_group(required=True)
    timing.add_argument("--length", type=float,
                        help="target total video length in seconds "
                             "(per-image time = length / number of images)")
    timing.add_argument("--per-image", type=float,
                        help="seconds each image should stay on screen")
    ap.add_argument("--eye-y", type=float, default=0.45,
                    help="target eye y position as fraction of frame height")
    ap.add_argument("--eye-dist", type=float, default=0.12,
                    help="target interocular distance as fraction of frame width "
                         "(smaller = more surrounding background visible)")
    ap.add_argument("--order", choices=("date", "name"), default="date",
                    help="frame order: EXIF date (default) or filename")
    ap.add_argument("--caption", action="store_true",
                    help="overlay a Month Year caption at the bottom "
                         "that crossfades between months")
    ap.add_argument("--fade-frames", type=int, default=2,
                    help="length of each caption crossfade in frames")
    ap.add_argument("--save-frames", default=None,
                    help="optional dir to also dump aligned still frames")
    ap.add_argument("--model", default=str(DEFAULT_MODEL),
                    help="path to face_landmarker.task model file")
    args = ap.parse_args()

    src_dir = Path(args.input)
    if not src_dir.is_dir():
        print(f"input folder not found: {src_dir}", file=sys.stderr)
        return 1

    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    entries = [(p, read_capture_date(p)) for p in src_dir.iterdir()
               if p.suffix.lower() in exts]
    if not entries:
        print(f"no images found in {src_dir}", file=sys.stderr)
        return 1
    if args.order == "date":
        entries.sort(key=lambda e: (e[1], natural_key(e[0].name)))
    else:
        entries.sort(key=lambda e: natural_key(e[0].name))
    print(f"found {len(entries)} images (ordered by {args.order})")

    frames_dir = Path(args.save_frames) if args.save_frames else None
    if frames_dir:
        frames_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model)
    if not model_path.is_file():
        print(f"model file not found: {model_path}\n"
              f"download with: curl -o {model_path} "
              f"https://storage.googleapis.com/mediapipe-models/face_landmarker/"
              f"face_landmarker/float16/latest/face_landmarker.task",
              file=sys.stderr)
        return 1
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.3,
        min_face_presence_confidence=0.3,
    )
    landmarker = FaceLandmarker.create_from_options(options)

    aligned: list[tuple[np.ndarray, datetime, str]] = []
    for i, (path, when) in enumerate(entries, 1):
        img = cv2.imread(str(path))
        if img is None:
            print(f"  [{i}/{len(entries)}] SKIP unreadable: {path.name}")
            continue
        eyes = detect_eyes(landmarker, img)
        if eyes is None:
            print(f"  [{i}/{len(entries)}] SKIP no face: {path.name}")
            continue
        frame = align(img, *eyes, args.width, args.height, args.eye_y, args.eye_dist)
        if frame is None:
            print(f"  [{i}/{len(entries)}] SKIP bad transform: {path.name}")
            continue
        aligned.append((frame, when, path.name))
        print(f"  [{i}/{len(entries)}] OK  {when:%Y-%m-%d}  {path.name}")
    landmarker.close()

    if not aligned:
        print("no frames aligned; nothing to write", file=sys.stderr)
        return 1

    dates = [w for _, w, _ in aligned]
    layers = (caption_layers(dates, args.fade_frames)
              if args.caption else [[] for _ in aligned])

    per_image = (args.length / len(aligned)) if args.length else args.per_image
    if per_image <= 0:
        print(f"per-image duration must be positive (got {per_image:.4f}s)",
              file=sys.stderr)
        return 1
    fps = 1.0 / per_image
    total = per_image * len(aligned)

    out_path = Path(args.output)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (args.width, args.height))
    for i, ((frame, _when, name), frame_layers) in enumerate(zip(aligned, layers), 1):
        for label, alpha in frame_layers:
            draw_caption(frame, label, alpha)
        writer.write(frame)
        if frames_dir:
            cv2.imwrite(str(frames_dir / f"{i:03d}_{Path(name).stem}.jpg"), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
    writer.release()
    print(f"wrote {len(aligned)} frames × {per_image:.3f}s "
          f"= {total:.2f}s @ {fps:.2f} fps → {out_path}"
          f"{' (caption on)' if args.caption else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
