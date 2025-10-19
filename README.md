# video_lag_fix

Automatically finds and fixes lags in video recordings by detecting duplicate frames and interpolating only the laggy sections using RIFE.

## Usage

Basic example:

```bash
./target/release/video_lag_fix -i input.mp4 -o output.mp4
```

## Processing pipeline

1. Perceptual hashing to quickly filter obviously unique frames.
2. `tiny_duplicate_net` classifier for robust duplicate detection (more resistant to artifacts).
3. `tiny_motion_net` regressor to decide the interpolation window by estimating global absolute motion magnitude.
4. RIFE interpolates intermediate frames.
5. Patch interpolated frames back into the video.

## Key points

* Naive full-video interpolation amplifies lags. This tool detects laggy regions and interpolates only those, which both reduces amplified lag and is faster than naive RIFE.
* Motion compensation interpolates more frames than were detected as duplicates to avoid the interpolated segment appearing slower.
* Uses RIFE-NCNN-Vulkan for interpolation.

## Key options

* `-i <path>` input file
* `-o <path>` output file
* `-m <path>` RIFE model path relative to the rife binary
* `--find-only` detect duplicates only, do not patch video
* `--render-hwaccel` ffmpeg hardware accel backend (default `cuda`)
* `--render-args` ffmpeg output args (must be changed for non-CUDA hardware or different codecs)

## Build

```bash
git clone --recursive https://github.com/redstonemedia/video_lag_fix.git
cd video_lag_fix
cargo build --release --bins
```

## Additional tools
- **validate** : used to validate duplicate detection performance from **insert_lags**.
- **insert_lags** : intentionally inserts artificial lags into videos for testing.
