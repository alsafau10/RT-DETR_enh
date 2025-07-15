#!/usr/bin/env python
"""
by lyuwenyu

CLI evaluation for RT-DETR with per-batch image enhancement and custom COCO summarization.
"""

import os
import sys
import argparse
import random
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import torch
import datetime
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as T

import src.misc.dist as dist
from src.core import YAMLConfig
from src.solver import TASKS
from src.data.coco import CocoDetection
from src.data import get_coco_api_from_dataset
from src.transforms.transform import (
    HistogramEqualization, GPUHistogramEqualization,
    LinearStretch,
    CLAHE, GPUCLAHE,
)
from src.data.coco import CocoEvaluator


def summarize(coco_eval, catId=None):
    """
    Custom COCO summarization. Returns (stats_list, print_string).
    """
    p = coco_eval.params

    def _summarize(ap=1, iouThr=None, areaRng='all', maxDets=100):
        p.maxDets =[1, 10, 100]
        if ap == 1:
            s = coco_eval.eval['precision']
            titleStr, typeStr = 'Average Precision', '(AP)'
        else:
            s = coco_eval.eval['recall']
            titleStr, typeStr = 'Average Recall', '(AR)'
        # select IoU
        if iouThr is not None:
            t = np.where(np.isclose(p.iouThrs, iouThr))[0]
            s = s[t]
        # select cats/areas/dets
        aind = p.areaRngLbl.index(areaRng)
        mind = p.maxDets.index(maxDets)
        if catId is not None:
            if ap == 1:
                s = s[:, :, catId, aind, mind]
            else:
                s = s[:, catId, aind, mind]
        else:
            if ap == 1:
                s = s[:, :, :, aind, mind]
            else:
                s = s[:, :, aind, mind]
        # mean non-negative
        valid = s[s > -1]
        mean_s = float(valid.mean()) if valid.size else -1.0
        iouStr = f"{iouThr:.2f}" if iouThr is not None else f"{p.iouThrs[0]:.2f}:{p.iouThrs[-1]:.2f}"
        line = f"{titleStr:<18} {typeStr} @[ IoU={iouStr:<9} | area={areaRng:>6s} | maxDets={maxDets:>3d} ] = {mean_s:0.3f}"
        return mean_s, line

    lines = []
    # AP @ 50 maxdet 1, 10,  100 lebih dari ini nilai cenderung sama
    for md in [1, 10, 100]:
        _, l = _summarize(1, .5, 'all', md)
        lines.append(l)
    # AR @ IoU=.50 for various maxdet lebih dari ini nilai cenderung sama
    for md in [1, 10, 100]:
        _, l = _summarize(0, .50, 'all', md)
        lines.append(l)

    for area in p.areaRngLbl:
        print(p.areaRngLbl)
        ap50, _ = _summarize(1, 0.50, area, 100)
        lines.append(_)
    for area in p.areaRngLbl:
        print(p.areaRngLbl)
        ar50, _ = _summarize(0, 0.50, area, 100)
        lines.append(_)
    for iou in [.50, .55, .60, .65, .70, .75, .80, .85, .90, .95]:
        _, l = _summarize(1, iou, 'all', p.maxDets[-1])
        lines.append(l)
    # AR @ IoU=.50 for various areas
    for area in ['all', 'small', 'medium', 'large']:
        _, l = _summarize(0, .50, area, p.maxDets[-1])
        lines.append(l)
    return lines


def main(args):

    dist.init_distributed()

    # load config & checkpoint
    cfg = YAMLConfig(args.config, resume=args.resume, use_amp=False, tuning=None)
    checkpoint = torch.load(args.resume, map_location='cpu')
    if 'ema' in checkpoint:
        state = checkpoint['ema']['module']
    else:
        state = checkpoint['model']
    cfg.model.load_state_dict(state)

    #konfigurasi model di dalam trainable checkpoint
    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    device = solver.cfg.device
    model = solver.cfg.model
    model.to(device).eval()

    #implementasi enhancer
    use_gpu = torch.cuda.is_available()
    if args.enhance_method == 'he':
        enhancer = GPUHistogramEqualization() if use_gpu else HistogramEqualization()
    elif args.enhance_method == 'linear':
        enhancer = LinearStretch()
    elif args.enhance_method == 'clahe':
        enhancer = GPUCLAHE(args.clahe_clip, tuple(args.clahe_grid)) if use_gpu \
                   else CLAHE(args.clahe_clip, tuple(args.clahe_grid))
    else:
        enhancer = None

    # epoch bisnis logic 
    apply_enh = enhancer is not None and (random.random() <= args.he_threshold)
    print(f"[EVAL] enhance_method={args.enhance_method}, applied={apply_enh}")

    # prepare dataset & dataloader
    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToImageTensor(),
        T.ConvertImageDtype(torch.float)
    ])
    test_ds = CocoDetection(
        img_folder=args.data_root,
        ann_file=args.ann_file,
        transforms=transforms,
        return_masks=None,
        remap_mscoco_category=True
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=4,
        num_workers= 2,
        drop_last=False,
        collate_fn=solver.cfg.val_dataloader.collate_fn
    )
    base_ds = get_coco_api_from_dataset(test_ds)

    # lakukan pengujian terhadap data testing
    coco_eval = CocoEvaluator(base_ds, solver.cfg.postprocessor.iou_types)
    model.eval()
    with torch.no_grad():
        for samples, targets in test_dl:
            samples = samples.to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            # apply enhancement
            if apply_enh:
                if hasattr(samples, 'tensors'):
                    samples.tensors = enhancer(samples.tensors)
                else:
                    samples = enhancer(samples)

            outputs = model(samples)
            orig_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
            results = solver.cfg.postprocessor(outputs, orig_sizes)
            res = {t["image_id"].item(): o for t, o in zip(targets, results)}
            coco_eval.update(res)

    coco_eval.synchronize_between_processes()
    coco_eval.accumulate()
    coco_eval.summarize()

    # sumarisasi
    lines = summarize(coco_eval.coco_eval["bbox"])
    print("\nCustom COCO Summary:")
    print("\n".join(lines))

    # save to file
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    record = out_dir / f"eval_{timestamp}_max_detects.txt"
    with open(record, "w") as f:
        f.write("Custom COCO Summary:\n")
        f.write("\n".join(lines))
    print(f"\nSaved evaluation to {record}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RT-DETR eval with enhancement")
    parser.add_argument('--config',       '-c',  type=str, required=True, help='path to YAML config')
    parser.add_argument('--resume',       '-r',  type=str, required=True, help='path to checkpoint .pth')
    parser.add_argument('--data-root',    '-dr', type=str, required=True, help='COCO image folder')
    parser.add_argument('--ann-file',     '-ann',type=str, required=True, help='COCO annotation JSON')
    parser.add_argument('--output-dir',   '-o',  type=str, default='outputs/eval', help='where to save results')

    # enhancement flags
    parser.add_argument('--enhance-method',
                        type=str, default='none',
                        choices=['none','he','linear','clahe'],
                        help='Which enhancement to run')
    parser.add_argument('--he-threshold',
                        type=float, default=1.0,
                        help='Probability to apply enhancement this run')
    parser.add_argument('--clahe-clip',
                        type=float, default=2.0,
                        help='CLAHE clipLimit')
    parser.add_argument('--clahe-grid',
                        type=int, nargs=2, default=[8,8],
                        help='CLAHE tileGridSize (h w)')

    args = parser.parse_args()
    main(args)
