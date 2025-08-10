import cv2
import os
import subprocess
import math
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights

DATA_ROOT = r"./data"
FLOW_IMAGE_SIZE = (1536, 864)

# Uhh
_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_preprocess = None

def get_flow(last, current):
    global _model, _preprocess

    # Lazy‐load because charpp wanted this
    if _model is None:
        weights = Raft_Large_Weights.DEFAULT
        _model = raft_large(weights=weights, progress=False).to(_device).eval()
        _preprocess = weights.transforms()

    def _bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(img_rgb)

    img1 = _bgr_to_pil(last)
    img2 = _bgr_to_pil(current)

    # Doohickey
    tensor1, tensor2 = _preprocess(img1, img2)  # C×H×W tensor
    tensor1 = tensor1.unsqueeze(0).to(_device)  # 1×C×H×W
    tensor2 = tensor2.unsqueeze(0).to(_device)
    # Inference
    with torch.no_grad():
        flows = _model(tensor1, tensor2)
    flow_tensor = flows[-1]  # 1×2×h'×w'
    # Upsample?
    h, w = last.shape[:2]
    flow_tensor = torch.nn.functional.interpolate(
        flow_tensor, size=(h, w), mode="bilinear", align_corners=False
    )
    # To NumPy
    flow_out = flow_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.float32)
    return flow_out


def calc_motion(last, current, min_mag = 0.2) -> tuple[np.ndarray, tuple[float, float, float]]:
    flow = get_flow(last, current)
    avg_flow_x = np.mean(flow[..., 0])
    avg_flow_y = np.mean(flow[..., 1])

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    motion = np.average(np.maximum(mag, min_mag) - min_mag)
    motion = max(float(motion), 1e-4) - 1e-4
    return flow, (motion, avg_flow_x, avg_flow_y)


def get_motion(frames_path: str):
    last = None
    last_file = None
    last_last = None
    last_last_file = None
    flow = None
    for file in sorted(os.listdir(frames_path)):
        p = os.path.join(frames_path, file)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, FLOW_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        # Two frames ago
        if last_last is not None:
            _, motion = calc_motion(last_last, img)
            yield last_last_file, file, motion
        # One frames ago
        if last is not None:
            flow, motion = calc_motion(last, img)
            visul_flow(flow)
            yield last_file, file, motion

        last_last = last
        last_last_file = last_file
        last = img
        last_file = file


def visul_flow(flow: np.ndarray):
    hsv_mask = np.zeros((FLOW_IMAGE_SIZE[1], FLOW_IMAGE_SIZE[0], 3), dtype=np.uint8)
    hsv_mask[..., 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv_mask[..., 0] = ang * 180 / np.pi / 2
    hsv_mask[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb_representation = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    cv2.imshow('flow', rgb_representation)
    cv2.waitKey(1)


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
        output_pattern = f"{segment_dir}/%07d.png"
        cmd = [
            "ffmpeg",
            "-hwaccel", "cuda",
            "-ss", str(start),
            "-i", str(video_path),
            "-t", str(segment_duration),
            "-vf", "scale=1920:1080",
            "-an",
            output_pattern
        ]
        print(f"Extracting segment {i+1}/{n} from {start} to {start + segment_duration}")
        subprocess.run(cmd)


def extract_all_segments(paths: list[str], segment_duration: float, max_n: int, segment_every: float):
    out_dir = Path(DATA_ROOT).joinpath("imgs")
    for video_path in paths:
        print("Extraing video: ", video_path)
        extract_segments(video_path, out_dir, segment_duration, max_n, segment_every)


def collect(name: str):
    sections_path = os.path.join(DATA_ROOT, "imgs")
    csv_path = os.path.join(DATA_ROOT, name)
    with open(csv_path, mode="a", encoding="utf-8") as f:
        for dir in os.listdir(sections_path):
            path = os.path.join(sections_path, dir)
            if not os.path.isdir(path):
                continue
            print(f"Segment: {dir}")
            n = len(os.listdir(path)) * 2
            for i, (f0, f1, motion) in enumerate(get_motion(path)):
                print(f"{i:04d}/{n-1}")
                f.write(f"{os.path.join("imgs", dir, f0)},{os.path.join("imgs", dir, f1)},{','.join([str(v) for v in motion])}\n")


if __name__ == "__main__":
    #paths = list(Path(DATA_ROOT).joinpath("vids").glob("*.*"))
    #paths += [r"D:\Aufnahmen\Videos_Raw\2025-04-30 01-39-08.mkv", r"D:\Aufnahmen\Videos_Raw\2025-04-19 13-37-38.mkv", r"D:\Aufnahmen\Videos_Raw\2023-05-10 22-36-36.mkv", r"D:\Aufnahmen\Videos_Raw\2025-08-03 16-25-30.mkv"]
    #extract_all_segments(paths, 2.5, 10, 2.3*60)
    collect("motion.csv")