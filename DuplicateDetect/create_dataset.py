import random

import cv2
import pandas
import pandas as pd
import tqdm

from collect import DATA_ROOT
import os
import itertools

def get_segment_pairs(segment_name: str) -> list[tuple[list[str], list[str]]]:
    variations = []
    current = []
    last_frame_number = -1

    segment_path = os.path.join(DATA_ROOT, "imgs", segment_name)
    for name in sorted(os.listdir(segment_path)):
        img_file = os.path.join(segment_name, name)
        split = name.split(".png")
        frame_number = int(split[0])
        if last_frame_number != frame_number and last_frame_number != -1:
            variations.append((last_frame_number, current))
            current = []
        current.append(img_file)
        last_frame_number = frame_number
    if current:
        variations.append((last_frame_number, current))

    pairs = []
    for i, (frame_number, current) in enumerate(variations):
        if i == 0:
            continue
        last_frame_number, last = variations[i-1]
        if last_frame_number + 1 == frame_number:
            pairs.append((last, current))
    return pairs


def create_from_pairs(pairs: list[tuple[list[str], list[str]]]):
    data = []
    for before, after in pairs:
        same_pairs = list(itertools.chain(
            itertools.combinations(after, 2),
            itertools.combinations(before, 2)
        ))
        same_pairs.append((before[0], before[0]))
        #same_pairs.append((after[0], after[0]))
        for pair in same_pairs:
            data.append((*pair, False))

        cross_pairs = list(itertools.product(before, after))
        if len(cross_pairs) > len(same_pairs):
            cross_pairs = random.sample(cross_pairs, len(same_pairs))
        for pair in cross_pairs:
            data.append((*pair, True))
    return data


def create_dataset():
    imgs_path = os.path.join(DATA_ROOT, "imgs")
    dataset = []
    for segment_name in tqdm.tqdm(os.listdir(imgs_path)):
        pairs = get_segment_pairs(segment_name)
        if not pairs:
            continue
        dataset += create_from_pairs(pairs)
    return dataset


def save_train_validation(dataset: list[tuple], validation_split: float):
    random.shuffle(dataset)
    num_validation_samples = int(len(dataset) * validation_split)

    validation_set = pd.DataFrame(dataset[:num_validation_samples], columns=["before", "after", "different"])
    train_set = pd.DataFrame(dataset[num_validation_samples:], columns=["before", "after", "different"])

    train_file_path = os.path.join(DATA_ROOT, "train.csv")
    validation_file_path = os.path.join(DATA_ROOT, "val.csv")

    train_set.to_csv(train_file_path, index=False)
    validation_set.to_csv(validation_file_path, index=False)

    print(f"Total rows in dataset: {len(dataset)}")
    print(f"Training set saved to: {train_file_path} with {len(train_set)} rows.")
    print(f"Validation set saved to: {validation_file_path} with {len(validation_set)} rows.")



def visul_check_segment(segment_name: str):
    pairs = get_segment_pairs(segment_name)
    for a, b in pairs:
        print("start")
        img = cv2.imread(os.path.join(DATA_ROOT, "imgs", a[0]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("a", img)
        cv2.waitKey(0)
        img = cv2.imread(os.path.join(DATA_ROOT, "imgs", b[0]), cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
        cv2.imshow("a", img)
        cv2.waitKey(0)


if __name__ == "__main__":
    seed = 1337
    random.seed(seed)
    dataset = create_dataset()
    save_train_validation(dataset, validation_split=0.25)
