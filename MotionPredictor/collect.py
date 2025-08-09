import cv2
import os
import subprocess
import math
import numpy as np
import torch
from PIL import Image
from torchvision.models.optical_flow import raft_large, Raft_Large_Weights


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


def calc_motion(last, current, flow, min_mag = 0.2):
    flow = get_flow(last, current)
    #flow = cv2.calcOpticalFlowFarneback(last, current, flow, 0.4, 7, 31, 10, 9, 1.8, cv2.OPTFLOW_FARNEBACK_GAUSSIAN)
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    motion = np.average(np.maximum(mag, min_mag) - min_mag)
    motion = max(float(motion), 1e-4) - 1e-4
    return flow, motion


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
            _, motion = calc_motion(last_last, img, flow)
            yield last_last_file, file, motion
        # One frames ago
        if last is not None:
            flow, motion = calc_motion(last, img, flow)
            visul_flow(flow)
            yield last_file, file, float(motion)

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

def extract_segments(video_path: str, segment_duration: float, n: int, output_dir: str):
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input file '{video_path}' not found.")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory '{output_dir}' not found.")

    total_duration = get_video_duration(video_path) / n
    start_times = []
    for i in range(n):
        center = (i + 0.5) * total_duration / n
        start = center - segment_duration / 2
        start = max(0.0, min(start, total_duration - segment_duration))
        start_times.append(math.floor(start))

    for i, start in enumerate(start_times):
        segment_dir = f"{output_dir}/{i:04d}"
        if os.path.exists(segment_dir):
            raise FileExistsError(f"Segment directory '{segment_dir}' already exists")
        os.mkdir(segment_dir)
        output_pattern = f"{segment_dir}/%06d.png"
        cmd = [
            "ffmpeg",
            "-i", video_path,
            "-ss", str(start),
            "-t", str(segment_duration),
            "-vf", "scale=1920:1080",
            output_pattern
        ]
        print(f"Extracting segment {i+1}/{n} from {start} to {start + segment_duration}")
        subprocess.run(cmd)


def collect():
    sections_path = "imgs"
    with open("motion_other.csv", mode="w", encoding="utf-8") as f:
        for dir in os.listdir(sections_path):
            path = os.path.join(sections_path, dir)
            print(f"Segment: {dir}")
            n = len(os.listdir(path))
            for i, (f0, f1, motion) in enumerate(get_motion(path)):
                print(f"{i:04d}/{n}")
                f.write(f"{os.path.join(path, f0)},{os.path.join(path, f1)},{motion}\n")


if __name__ == "__main__":
    #extract_segments(r"D:\Aufnahmen\Videos_Raw\2025-07-18 19-53-12.mkv", 5.0, 10, r"D:\Coding\VideoLagFix\MotionPredictor\imgs")
    collect()