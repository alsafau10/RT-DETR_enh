import os
import argparse
from os.path import isfile
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import ToTensor
import onnxruntime as ort
import cv2
import time

# -------------------------------------------------------------------
CLASS_NAMES = ['speedbump']
VIDEO_EXTS = ('.mp4', '.avi', '.mov', '.mkv')
IMG_EXTS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
# -------------------------------------------------------------------

def run_inference(sess, frame_pil, original_size, thrh, font):
    # 1) Resize to 640x640 & batchify
    im_resized = frame_pil.resize((640, 640))
    im_data = ToTensor()(im_resized)[None]     # shape (1,3,640,640)
    size    = torch.tensor([[640, 640]])       # shape (1,2)

    # 2) Run ONNX inference
    output = sess.run(
        None,
        {'images': im_data.numpy(), 'orig_target_sizes': size.numpy()}
    )
    labels, boxes, scores = output

    # 3) Draw detections
    draw = ImageDraw.Draw(frame_pil)
    for i in range(im_data.shape[0]):
        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]

        for b, l in zip(box, lab):
            # scale back to original size
            b_scaled = [
                coord * original_size[j % 2] / 640
                for j, coord in enumerate(b)
            ]
            class_name = CLASS_NAMES[int(l)]
            draw.rectangle(b_scaled, outline='red', width=2)
            draw.text((b_scaled[0], b_scaled[1]), class_name, fill='yellow', font=font)
    return frame_pil

def draw_timing(frame_cv, inf_time_ms, fps, pos=(10, 30)):
    """Overlay inference time and FPS using OpenCV."""
    text = f"Inference: {inf_time_ms:.1f} ms | FPS: {fps:.1f}"
    cv2.putText(frame_cv, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)
    return frame_cv

def process_video(sess, video_path, output_dir, thrh, font,random_value = 0):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video file {video_path}")
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(output_dir, basename + f"_annotated_{random_value}.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(out_path, fourcc, fps, (w, h))
    frame_idx = 0

    # For reporting average FPS
    total_time = 0
    total_frames = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert to PIL Image for annotation
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        # --- Timing the inference ---
        t0 = time.time()
        annotated = run_inference(sess, frame_pil, (w, h), thrh, font)
        t1 = time.time()
        inf_time = t1 - t0
        inf_time_ms = inf_time * 1000
        curr_fps = 1.0 / inf_time if inf_time > 0 else 0
        total_time += inf_time
        total_frames += 1
        # ----------------------------

        # Convert back to OpenCV image
        annotated_cv = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        annotated_cv = draw_timing(annotated_cv, inf_time_ms, curr_fps)
        out_vid.write(annotated_cv)
        frame_idx += 1
        if frame_idx % 10 == 0:
            print(f"Processed {frame_idx} frames... (Current FPS: {curr_fps:.2f})")

    cap.release()
    out_vid.release()
    avg_fps = total_frames / total_time if total_time > 0 else 0
    print(f"✅ Saved annotated video: {out_path}")
    print(f"Average inference FPS: {avg_fps:.2f}")

def process_images(sess, input_dir, output_dir, thrh, font,random_value = 0):
    os.makedirs(output_dir, exist_ok=True)
    for fn in sorted(os.listdir(input_dir)):
        if not fn.lower().endswith(IMG_EXTS):
            continue
        path = os.path.join(input_dir, fn)
        basename = fn.split('.')[0]
        EXT_IM = fn.split('.')[1]
        original_im = Image.open(path).convert('RGB')
        original_size = original_im.size  # (width, height)
        annotated = run_inference(sess, original_im, original_size, thrh, font)
        out_path = os.path.join(output_dir, f'{fn}')
        annotated.save(out_path)
        print(f"✅ Saved {out_path}")

def process_single_image(sess, input_dir, output_dir, thrh, font,random_value = 0):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_dir))[0]
    IM_EXT  = os.path.splitext(os.path.basename(input_dir))[1]
    original_im = Image.open(input_dir).convert('RGB')
    original_size = original_im.size #(width,height)
    anotated = run_inference(sess, original_im, original_size, thrh, font)
    out_path = os.path.join(args.output_dir, f'{basename}_{random_value}.{IM_EXT}')
    anotated.save(out_path)
    print(f"✅ Saved {out_path}")

def main(args):
    # create ONNX Runtime session
    import random
    rand_value = random.randint(0,9999)
    sess = ort.InferenceSession(
        args.model,
        providers=["CUDAExecutionProvider","CPUExecutionProvider"]
    )

    # Show ONNX provider info
    print("ONNX Runtime providers:", sess.get_providers())
    print("ONNX Runtime session device:", sess.get_provider_options())

    try:
        font = ImageFont.truetype("Arial.ttf", 15)
    except IOError:
        font = ImageFont.load_default()

    if os.path.isdir(args.input):
        process_images(sess, args.input, args.output_dir, args.conf_thresh, font, rand_value)
    elif os.path.isfile(args.input) and args.input.lower().endswith(VIDEO_EXTS):
        process_video(sess, args.input, args.output_dir, args.conf_thresh, font, rand_value)
    elif os.path.isfile(args.input) and args.input.lower().endswith(IMG_EXTS):
        process_single_image(sess, args.input, args.output_dir, args.conf_thresh, font, rand_value)
    else:
        raise ValueError("Input must be an image, a folder of images, or a video file!")

if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Folder or video inference with single-class speedbump"
    )
    p.add_argument("-m", "--model",      required=True, help="Path to ONNX model (e.g. model.onnx)")
    p.add_argument("-i", "--input",      required=True, help="Input folder (images) or video file")
    p.add_argument("-o", "--output_dir", required=True, help="Where to save annotated images/video")
    p.add_argument("-t", "--conf_thresh", type=float, default=0.55, help="Confidence threshold for detections")
    args = p.parse_args()
    main(args)

