import torch
import torchinfo
import argparse

from dataset import INPUT_IMAGE_SIZE
from train import TinyMotionNet

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("output_path")
    args = parser.parse_args()

    model = TinyMotionNet(probabilistic=True, output_dist=False)
    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    input_shape = (1, 1, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0])
    torchinfo.summary(model, input_shape, device="cpu")
    model.eval() # just to be sure
    dummy = torch.randn(input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        args.output_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"]
    )


if __name__ == "__main__":
    main()