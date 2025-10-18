import io
import os
import random

import cv2
import tqdm
from PIL import Image
import numpy as np
from functools import partial
import re
import glob
from multiprocessing import Pool

from collect import DATA_ROOT


def generate_variation(
        img_path: str,
        i: int,
        quality_range: tuple[int, int],
        detail_noise_range: tuple[float, float],
        macro_noise_range: tuple[float, float],
        macro_resize_range: tuple[float, float],
        offset_range: tuple[float, float],
        resolutions: tuple[tuple[int, int]],
        passthrough_chance: float
):
    out_path = f"{img_path}_{i}.png"

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    new_size = random.choice(resolutions)
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_LANCZOS4).astype(dtype=np.float32)
    if random.random() < passthrough_chance:
        out = cv2.resize(resized, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LANCZOS4).astype(np.uint8)
        cv2.imwrite(out_path, out)
        return

    detail_noise_std = random.uniform(*detail_noise_range)
    macro_noise_std = random.uniform(*macro_noise_range)
    detail_noise = np.random.normal(0.0, detail_noise_std * 255.0, resized.shape)
    macro_size = random.uniform(*macro_resize_range)
    macro_shape = (round(img.shape[0] / macro_size), round(img.shape[1] / macro_size), img.shape[2])
    macro_noise = np.random.normal(0.0, macro_noise_std * 255.0, macro_shape)
    macro_noise = cv2.resize(macro_noise, new_size, interpolation=cv2.INTER_NEAREST)
    offset = (random.uniform(*offset_range), random.uniform(*offset_range), random.uniform(*offset_range))
    change = detail_noise + macro_noise + offset
    noisy = np.clip(resized + change, 0, 255).astype(np.uint8)

    noisy_pil = Image.fromarray(cv2.cvtColor(noisy, cv2.COLOR_BGR2RGB), mode="RGB")
    buffer = io.BytesIO()
    quality = random.randint(*quality_range)
    noisy_pil.save(buffer, format="JPEG", quality=quality)
    buffer.seek(0)
    jpeg_img = np.asarray(Image.open(buffer)).astype(np.float32)
    jpeg_img = cv2.cvtColor(jpeg_img, cv2.COLOR_RGB2BGR)

    undo = offset + macro_noise
    denoised_jpeg = np.clip(jpeg_img - undo, 0, 255).astype(np.uint8)
    out = cv2.resize(denoised_jpeg, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_AREA)
    out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)

    cv2.imwrite(out_path, out)


def generate_variations(
        img_path: str,
        n: int,
        quality_range=(13, 55),
        detail_noise_range=(0.0003, 0.01),
        macro_noise_range=(0.0001, 0.019),
        macro_resize_range=(1, 5),
        offset_range=(-7.0, 7.0),
        resolutions=((1280, 720), (1920, 1080)),
        passthrough_chance=0.02
):
    for i in range(n):
        generate_variation(img_path, i, quality_range, detail_noise_range, macro_noise_range, macro_resize_range, offset_range, resolutions, passthrough_chance)


def generate_all_variations(path: str, n=3, processes=6):
    ignore_pattern = re.compile(r"^.+\/[^.]+\.png_.+\.png$")
    paths_iter = glob.glob(f"{path}/**/*.png", recursive=True)
    filtered = list(filter(lambda f: not ignore_pattern.match(f), paths_iter))
    with Pool(processes) as pool:
        func = partial(generate_variations, n=n)
        _ = list(tqdm.tqdm(pool.imap(func, filtered, chunksize=10), total=len(filtered)))


if __name__ == "__main__":
    generate_all_variations(os.path.join(DATA_ROOT, "imgs"), processes=10)