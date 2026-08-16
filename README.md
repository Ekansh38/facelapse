# facelapse

Turns a folder of face photos into a video where the eyes stay in the same spot every frame, so the
face looks locked while everything around it changes.

## Demo

I have a really cool video with my face pictures over time but ill upload it here when I am a little
more comfortable showing my face 😢. (trust me its really good)

## What you need

- Python 3.11 or newer
- A folder of photos of the same person, roughly facing the camera

## Setup

Run these once:

```bash 
python3 -m venv venv 
source venv/bin/activate 
pip3 install -r requirements.txt 
curl -o face_landmarker.task \ https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

Every new terminal, activate the venv again:

```bash source venv/bin/activate ```

## Use

Put your photos in a folder called `face_images` next to `stitch.py`.

If your photos are HEIC (e.g. straight from an iPhone), convert them first:

```bash python heic_to_jpeg.py ```

That drops a `.jpg` next to each `.heic` in `face_images`, preserving EXIF so
capture dates still work. Add `--delete-originals` to remove the HEICs after.

Then run `stitch.py`. With no flags it uses a sensible default of 0.15s per
photo:

```bash python stitch.py ```

You can override the timing with one of:

- `--length SECONDS`: total video length. Each photo gets `length / number of photos` on screen.
- `--per-image SECONDS`: how long each photo stays on screen.

For example, to make a 5 second video with 25 photos (0.2 seconds each):

```bash python stitch.py --length 5 ```

Or to give each photo exactly 0.1 seconds:

```bash python stitch.py --per-image 0.1 ```

That writes `output.mp4` at 1920x1080.

## Options

| flag | default | what it does |
|---|---|---|
| `--length SECONDS` | (optional) | total video length. Splits it evenly across your photos |
| `--per-image SECONDS` | `0.15` when neither is set | seconds each photo stays on screen |
| `--input FOLDER` | `FACE IMAGES` | folder to read photos from |
| `--output FILE` | `output.mp4` | video file to write |
| `--width N` | `1920` | output width in pixels |
| `--height N` | `1080` | output height in pixels |
| `--eye-dist F` | `0.12` | face size as a fraction of frame width. Smaller shows more background |
| `--eye-y F` | `0.45` | eye vertical position as a fraction of frame height |
| `--order date\|name` | `date` | play photos by EXIF date, or by filename |
| `--caption` | off | show a Month Year caption at the bottom that crossfades between months |
| `--fade-frames N` | `2` | frames used to crossfade between month labels. `0` means hard cut |
| `--save-frames DIR` | off | also save each aligned photo as a JPG in this folder |

## How it works

1. Reads every photo in the input folder.
2. Finds the two eyes in each photo using MediaPipe FaceLandmarker.
3. Rotates, scales, and shifts each photo so both eye centers land at the same pixel positions on
   the output canvas.
4. Pulls the EXIF capture date from each photo and orders the frames by that.
5. Writes them into an mp4, one frame per photo, with an optional month and year caption that
   crossfades at every month change.

## Common tweaks

Slower playback (longer total video): ```bash python stitch.py --length 10 ```

Bigger face, less background: ```bash python stitch.py --length 3 --eye-dist 0.2 ```

Turn on the month caption: ```bash python stitch.py --length 3 --caption ```

Play in filename order instead of date: ```bash python stitch.py --length 3 --order name ```
