import cv2
import numpy as np
import torch
import os
from PIL import Image

from img_process_server_connect import ImageProcessServerConnect
from train import TinyMotionNet, DATA_ROOT
from dataset import INPUT_IMAGE_SIZE, MotionDataset
import torchvision.transforms.functional as TF
from collect import FLOW_IMAGE_SIZE, calc_motion


def pred_one(img_a_path: str, img_b_path: str, model: TinyMotionNet) -> float:
    #images = img_server.ask_for_images([img_a_path, img_b_path], INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1])

    img1 = cv2.imread(img_a_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img_b_path, cv2.IMREAD_GRAYSCALE)

    img1 = cv2.resize(img1, INPUT_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, INPUT_IMAGE_SIZE, interpolation=cv2.INTER_CUBIC)

    norm1 = img1.astype(np.float32) / 255.0
    norm2 = img2.astype(np.float32) / 255.0
    diff = norm1 - norm2

    cv2.imshow("diff", (diff + 1) / 2.0)
    cv2.imshow("abs", np.abs(diff))
    cv2.waitKey()

    x = TF.to_tensor(diff).to(torch.float32)
    result = model(x.unsqueeze(1))
    print(result)
    return result[0].item()


def main():
    model = TinyMotionNet(denorm=False)
    state_dict = torch.load(r".\models\v5\40598_-0.847.pth") # models/v3/80430_0.051.pth
    model.load_state_dict(state_dict)
    model.eval()
    """
    img_server = ImageProcessServerConnect(
        cache_dir=os.path.join(DATA_ROOT, "cache"),
        threaded_reads=True,
        working_dir=".",
        fill_strategy="constant 0",
        filter_type="CatmullRom",
        grayscale=True
    )"""

    dir = r"data\imgs\00083"
    last_path = None
    for file in os.listdir(dir):
        path = os.path.join(dir, file)
        if last_path is not None:
            cv_last = cv2.imread(last_path, cv2.IMREAD_GRAYSCALE)
            cv_last = cv2.resize(cv_last, FLOW_IMAGE_SIZE, interpolation=cv2.INTER_AREA)
            cv = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            cv = cv2.resize(cv, FLOW_IMAGE_SIZE, interpolation=cv2.INTER_AREA)

            _, (real_motion, ax, ay) = calc_motion(cv_last, cv)
            real_motion = MotionDataset.calc_raw_target(real_motion, ax, ay)
            pred_motion = pred_one(last_path, path, model)
            print(f"{file} real: {real_motion:.2f} pred: {pred_motion:.2f}")
        last_path = path


if __name__ == "__main__":
    main()