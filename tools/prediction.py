import torch
import torch.nn as nn 
import torchvision.transforms as T
import numpy as np 
from PIL import Image, ImageDraw
import os 
import sys 
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pycocotools.coco import COCO
from torchvision.ops import box_iou

import time
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from src.core import YAMLConfig 
from src.solver import TASKS
from src.transforms.transform import HistogramEqualization, CLAHE, LinearStretch

def tensor_to_pil_image(tensor):
    if tensor.max() <= 1.0:
        tensor = tensor * 255.0
    return Image.fromarray(tensor.permute(1, 2, 0).byte().cpu().numpy())

def save_image_with_boxes(image_tensor, gt_boxes, tp_boxes, fp_boxes, fn_boxes, filename):
    image = tensor_to_pil_image(image_tensor)
    draw = ImageDraw.Draw(image)
    for box in gt_boxes:
        draw.rectangle(box, outline="blue", width=2)
    for box in tp_boxes:
        draw.rectangle(box, outline="green", width=2)
    for box in fp_boxes:
        draw.rectangle(box, outline="red", width=2)
    for box in fn_boxes:
        draw.rectangle(box, outline="orange", width=2)
    image.save(filename)
    print(f"Image saved to: {filename}")

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)

    cfg = YAMLConfig(args.config, resume=args.resume)
    checkpoint = torch.load(args.resume, map_location='cpu') 
    state = checkpoint.get('ema', {}).get('module', checkpoint.get('model'))
    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()
        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            return self.postprocessor(outputs, orig_target_sizes)

    model = Model().to(args.device)
    model.eval()

    enhancer = None
    if args.enhance == 'he':
        enhancer = HistogramEqualization()
    elif args.enhance == 'clahe':
        enhancer = CLAHE(clip_limit=args.clahe_clip, tile_grid_size=tuple(args.clahe_grid))
    elif args.enhance == 'linear':
        enhancer = LinearStretch()

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor()
    ])

    coco = COCO(args.ann_file)
    images = coco.loadImgs(coco.getImgIds())

    total_tp = total_fp = total_fn = 0
    iou_threshold = 0.5
    conf_threshold = 0.5
    apply_enh = random.random() <= args.he_threshold
    print(f"[Inference] apply enhancement: {apply_enh} enhance method {args.enhance}")

    sum_inference_time = 0

    image_dir = Path(args.image_dir)
    image_paths = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
    print(f"Found {len(image_paths)} images in {image_dir}")

    for img_info in images:
        start = time.time()
        fname = img_info['file_name']
        img_path = os.path.join(args.image_dir, fname)
        if not os.path.exists(img_path):
            continue

        im_pil = Image.open(img_path).convert('RGB')
        w, h = im_pil.size
        orig_size = torch.tensor([[w, h]]).to(args.device)

        im_tensor = transforms(im_pil).unsqueeze(0).to(args.device)
        apply_enh = enhancer is not None and (apply_enh)
        
        if apply_enh:
            im_tensor = enhancer(im_tensor)
        
        with torch.no_grad():
            labels, boxes, scores = model(im_tensor, orig_size)

        labels = labels.squeeze(0)
        boxes = boxes.squeeze(0)
        scores = scores.squeeze(0)

        keep = scores > conf_threshold
        boxes = boxes[keep]
        labels = labels[keep].to(torch.int)
        scores = scores[keep]

        ann_ids = coco.getAnnIds(imgIds=[img_info['id']])
        anns = coco.loadAnns(ann_ids)
        gt_boxes = []
        for ann in anns:
            x, y, w_, h_ = ann['bbox']
            gt_boxes.append([x, y, x + w_, y + h_])

        gt_boxes_tensor = torch.tensor(gt_boxes, dtype=torch.float32).to(args.device)
        matched_gt = set()
        used_preds = set()
        tp_boxes = []
        fp_boxes = []
        fn_boxes = []
        tp = fp = 0

        if boxes.shape[0] > 0 and gt_boxes_tensor.shape[0] > 0:
            iou_matrix = box_iou(boxes, gt_boxes_tensor)

            for pred_idx in range(iou_matrix.shape[0]):
                if pred_idx in used_preds:
                    continue

                gt_idx = torch.argmax(iou_matrix[pred_idx]).item()
                max_iou = iou_matrix[pred_idx, gt_idx].item()

                if max_iou >= iou_threshold and gt_idx not in matched_gt:
                    tp += 1
                    matched_gt.add(gt_idx)
                    used_preds.add(pred_idx)
                    tp_boxes.append(boxes[pred_idx].cpu().tolist())

                    for other_pred_idx in range(iou_matrix.shape[0]):
                        if other_pred_idx != pred_idx and iou_matrix[other_pred_idx, gt_idx] >= iou_threshold:
                            used_preds.add(other_pred_idx)
                else:
                    if pred_idx not in used_preds:
                        fp += 1
                        fp_boxes.append(boxes[pred_idx].cpu().tolist())
                        used_preds.add(pred_idx)
        else:
            for b in boxes:
                fp_boxes.append(b.cpu().tolist())
            fp += boxes.shape[0]

        for j in range(len(gt_boxes)):
            if j not in matched_gt:
                fn_boxes.append(gt_boxes[j])
        fn = len(fn_boxes)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        end = time.time() - start
        sum_inference_time+=end
        print(f"[{fname}] TP={tp} FP={fp} FN={fn} Inference {end}")

        vis_path = os.path.join(args.output_dir, f"vis_{os.path.splitext(fname)[0]}.jpg")
        save_image_with_boxes(im_tensor.squeeze(0).cpu(), gt_boxes, tp_boxes, fp_boxes, fn_boxes, vis_path)

    with open(os.path.join(args.output_dir, "summary_metrics.txt"), "w") as f:
        f.write(f"TP: {total_tp}\n")
        f.write(f"FP: {total_fp}\n")
        f.write(f"FN: {total_fn}\n")
        f.write(f"total time : {sum_inference_time}\n")
    print(f"✓ Summary saved to {args.output_dir}/summary_metrics.txt")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, required=True)
    parser.add_argument('-r', '--resume', type=str, required=True)
    parser.add_argument('-idir', '--image-dir', type=str, required=True)
    parser.add_argument('--ann-file', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--enhance', type=str, default='none', choices=['none', 'he', 'clahe', 'linear'])
    parser.add_argument('--clahe-grid', type=int, nargs=2, default=[8, 8])
    parser.add_argument('--clahe-clip', type=float, default=2.0)
    parser.add_argument('--he-threshold', type=float, default=1.0)
    parser.add_argument('--output-dir', type=str, default='outputs')
    args = parser.parse_args()
    main(args)
