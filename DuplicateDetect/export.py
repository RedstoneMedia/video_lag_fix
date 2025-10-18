import torch
import torchinfo
import argparse
import importlib
import ast

from dataset import INPUT_IMAGE_SIZE

def parse_kwargs(kwargs_list):
    kwargs = {}
    for kw in kwargs_list or []:
        key, value = kw.split("=")
        try:
            value = ast.literal_eval(value)
        except ValueError:
            pass
        kwargs[key] = value
    return kwargs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to the model's state dict")
    parser.add_argument("output_path", help="ONNX output path")
    parser.add_argument("model_class", help="Name of the model class")
    parser.add_argument("--kwargs", nargs="*", help="Model constructor kwargs (key=value)")
    args = parser.parse_args()
    model_kwargs = parse_kwargs(args.kwargs)

    models_module = importlib.import_module("models")
    if not hasattr(models_module, args.model_class):
        print(f"Model '{args.model_class}' not found")
        exit(1)
    ModelClass = getattr(models_module, args.model_class)
    model = ModelClass(**model_kwargs)

    state_dict = torch.load(args.model_path)
    model.load_state_dict(state_dict)
    model.to("cpu")
    model.eval()

    input_shape = (1, 2, INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0])
    torchinfo.summary(model, input_shape, device="cpu")
    model.eval() # just to be sure
    dummy = torch.randn(input_shape, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy,
        args.output_path,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamo=True
    )


if __name__ == "__main__":
    main()