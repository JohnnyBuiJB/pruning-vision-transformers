import argparse
import datetime
import numpy as np
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import json

from datasets import load_dataset, Image

from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma

from transformers import ViTForImageClassification
from torchvision import transforms

from engine import train_one_epoch, evaluate
from samplers import RASampler
from functools import partial
from models.dyvit import VisionTransformerDiffPruning
import utils

from numbers import Number
from typing import Any, Callable, List, Optional, Union
from numpy import prod
import numpy as np
from fvcore.nn import FlopCountAnalysis

IMGNET_MEAN = [0.485, 0.456, 0.406]
IMGNET_STD = [0.229, 0.224, 0.225]

transform = transforms.Compose([
    transforms.Resize((256,256)), # Resize image to 256x256
    transforms.CenterCrop((224, 224)), # Center crop image to 224x224
    transforms.ToTensor(), # Convert image to PyTorch tensor
    transforms.Normalize(mean=IMGNET_MEAN, std=IMGNET_STD) # Normalize image using imagenet values
])

def rfft_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the rfft/rfftn operator.
    """
    input_shape = inputs[0].type().sizes()
    B, H, W, C = input_shape
    N = H * W
    flops = N * C * np.ceil(np.log2(N))
    return flops

def calc_flops(model, img_size=224, show_details=False, ratios=None):
    with torch.no_grad():
        x = torch.randn(1, 3, img_size, img_size).cuda()
        model.default_ratio = ratios
        fca1 = FlopCountAnalysis(model, x)
        handlers = {
            'aten::fft_rfft2': rfft_flop_jit,
            'aten::fft_irfft2': rfft_flop_jit,
        }
        fca1.set_op_handle(**handlers)
        flops1 = fca1.total()
        if show_details:
            print(fca1.by_module())
        print("\n#### GFLOPs: {}".format(flops1 / 1e9))
    return flops1 / 1e9

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model', default='deit_base', type=str, help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int, help='images input size')
    parser.add_argument('--data_path', default='/datasets01/imagenet_full_size/061417/', type=str,
                        help='dataset path')
    parser.add_argument('--data_set', default='IMNET', choices=['CIFAR', 'IMNET', 'INAT', 'INAT19'],
                        type=str, help='Image Net dataset path')
    parser.add_argument('--imagenet_default_mean_and_std', type=utils.str2bool, default=True)
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--model_path', default='', help='resume from checkpoint')
    parser.add_argument('--crop_pct', type=float, default=None)
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--base_rate', type=float, default=0.7)

    return parser

def collate_fn(batch):
    images = [transform(img_path["image"].convert('RGB')) for img_path in batch] # create list of images 
    labels = [item["label"] for item in batch] # create list of labels
    return {"image": torch.stack(images), "label": labels} # return dict of images and labels

def main(args):

    cudnn.benchmark = True
    dataset_val = load_dataset('imagenet-1k', split='validation', cache_dir=args.data_path)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=False
    )

    base_rate = args.base_rate
    KEEP_RATE1 = [base_rate, base_rate ** 2, base_rate ** 3]

    print(f"Creating model: {args.model}")

    PRUNING_LOC = [3,6,9] 
    print('token_ratio =', KEEP_RATE1, 'at layer', PRUNING_LOC)
    model = VisionTransformerDiffPruning(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True, 
        pruning_loc=PRUNING_LOC, token_ratio=KEEP_RATE1
    )
    
    model_path = args.model_path
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model"])

    #model = ViTForImageClassification.from_pretrained('facebook/deit-base-patch16-224')
    #model.config.num_classes = 1000

    print('## model has been successfully loaded')

    model = model.cuda()

    n_parameters = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters)

    #validate(data_loader_val, model)
    calc_flops(model)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def validate(val_loader, model):
    batch_time = AverageMeter('Time', ':6.3f')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.eval()

    progress = ProgressMeter(
        len(val_loader),
        [batch_time, top1, top5],
        prefix='Test: ')


    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(val_loader):
            images = batch["image"].cuda()
            labels = torch.FloatTensor(batch["label"]).cuda()

            # compute output
            output = model(images)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 20 == 0:
                progress.display(i)

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Dynamic evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)