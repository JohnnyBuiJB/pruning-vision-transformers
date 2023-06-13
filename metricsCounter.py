# Code adapted from DynamicViT: https://github.com/raoyongming/DynamicViT/blob/master/calc_flops.py

import warnings
from transformers import ViTForImageClassification
import torch
from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis

# Define number of classes (not sure if needed but added to be safe)
num_classes = 1000

def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops

def calc_metrics(model, img_size=224):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size)
        model.default_ratio = None
        fca1 = FlopCountAnalysis(model, x)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        print("\n----------------------\n")
        print("#### FLOPs: {} (B)".format(round(flops1 / 1e9, 1)))
        print("#### PARAMS: {} (M)\n".format(round(model.num_parameters() / 1e6, 1)))
    return 


def main():
    model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
    model.config.num_classes = num_classes

    calc_metrics(model) # calculate flops and params

if __name__ == "__main__":
    main()