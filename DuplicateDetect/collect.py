import os
import subprocess
import math
from pathlib import Path

DATA_ROOT = r"./data"

def get_video_duration(file_path):
    result = subprocess.run([
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration",
        "-of", "csv=p=0",
        file_path
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    try:
        return float(result.stdout.strip())
    except ValueError:
        raise RuntimeError("Could not extract video duration.")


def extract_segments(video_path: Path | str, output_dir: Path | str, segment_duration: float, max_n: int, segment_every: float):
    if isinstance(video_path, str):
        video_path = Path(video_path)
    if isinstance(output_dir, str):
        output_dir = Path(output_dir)

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input file '{video_path}' not found.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' not found.")

    total_duration = get_video_duration(video_path)

    n = math.floor((total_duration - segment_duration) / segment_every) + 1
    n = min(max_n, n)
    if n <= 0:
        raise ValueError("Video too short for any segments.")
    gap = (total_duration - n * segment_duration) / (n + 1)
    start_times = [gap * (i + 1) + segment_duration * i for i in range(n)]

    for i, start in enumerate(start_times):
        j = i
        segment_dir = Path(output_dir).joinpath(f"{j:05d}")
        while segment_dir.exists():
            j += 1
            segment_dir = Path(output_dir).joinpath(f"{j:05d}")
        os.mkdir(segment_dir)
        cmd = [
            "target/release/extract_similar",
            "-i", str(video_path.absolute()),
            "-s", str(start),
            "-t", str(segment_duration),
            "-l", "0.008",
            "-u", "0.011",
            "-o", str(segment_dir.absolute()),
        ]
        print(f"Extracting segment {i+1}/{n} from {start} to {start + segment_duration}")
        cwd = os.getcwd()
        os.chdir('..')
        subprocess.run(cmd)
        os.chdir(cwd)


def extract_all_segments(paths: list[str], segment_duration: float, max_n: int, segment_every: float):
    out_dir = Path(DATA_ROOT).joinpath("imgs")
    for video_path in paths:
        print("Extraing video: ", video_path)
        extract_segments(video_path, out_dir, segment_duration, max_n, segment_every)


if __name__ == "__main__":
    paths = list(Path(DATA_ROOT).joinpath("vids").glob("*.*"))
    paths += [r"/mnt/D/Aufnahmen/Videos_Raw/2025-10-01 22-01-02.mkv", "/mnt/D/Aufnahmen/Videos_Raw/2025-10-02 21-29-34.mkv", "/mnt/D/Aufnahmen/Videos_Raw/2025-09-29 22-23-13.mkv", "/mnt/D/Aufnahmen/Videos_Raw/2025-08-22 20-35-43.mkv", "/mnt/D/Aufnahmen/Videos_Raw/2025-09-16 23-49-28.mkv"]
    extract_all_segments(paths, 20, 10, 2.3*60)