# video_lag_fix

Automatically finds and fixes lags in video recordings by detecting duplicate frames and interpolating only the laggy sections using RIFE.

> **Note:** his tool is not a magic bullet. While interpolating large freeze frames in low-motion areas works, high-motion segments with long freezes are difficult to restore.

## Key points

* Naive full-video interpolation amplifies lags. This tool detects laggy regions and interpolates only those, which both reduces amplified lag and is faster than naive RIFE.
* Motion compensation interpolates more frames than were detected as duplicates to avoid the interpolated segment appearing slower.
* Uses RIFE-NCNN-Vulkan for interpolation.

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


## Video Example

https://github.com/user-attachments/assets/1a966c1d-e2ae-4f3d-a6c5-81ca7646f238

*Left: Laggy video, Right: Fixed video*
*(both slowed by 50%)*

## Key options

* `-i <path>` input file
* `-o <path>` output file
* `-m <path>` RIFE model path relative to the rife binary
* `--find-only` detect duplicates only, do not patch video
* `--render-hwaccel` ffmpeg hardware accel backend (default `cuda`)
* `--render-args` ffmpeg output args (must be changed for non-CUDA hardware or different codecs)
* `--max-motion-mul` maximum multiple of the background average motion allowed to still interpolate (default 7.3). Higher values allow more motion to be interpolated; lower values restrict high-motion interpolation. Range: 1.0..

## Build

```bash
git clone --recursive https://github.com/redstonemedia/video_lag_fix.git
cd video_lag_fix
cargo build --release --bins
```

You will also need to follow the build instructions for [rife-ncnn-vulkan](https://github.com/RedstoneMedia/rife-ncnn-vulkan/?tab=readme-ov-file#build-from-source). \
The rife binary is expected in `rife-ncnn-vulkan/build`.


## Additional tools
- **validate** : used to validate duplicate detection performance from **insert_lags**.
- **insert_lags** : intentionally inserts artificial lags into videos for testing.
- **log_viewer** : visualizes the produced logs for debugging. Best used with -v when generating logs: <img width="812" height="464" alt="Screenshot_20251020_144143" src="https://github.com/user-attachments/assets/7ceb7af0-12ed-4b5c-8150-b70021b08be6" />


