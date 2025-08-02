## TODO
<details>
  <summary>  Model Terbaik Pengujian (Pytorch)  </summary>

Model yang memberikan weight terbaik dapat diunduh pada tautan ini [``weight``](https://github.com/alsafau10/RT-DETR_enh/releases/download/weight/best_model.pth)
    
</details>
<!-- <details> -->
<!-- <summary> see details </summary>
- [x] Training
- [x] Evaluation
- [x] Export onnx
- [x] Upload source code
- [x] Upload weight convert from paddle, see [_links_](https://github.com/lyuwenyu/RT-DETR/issues/42)
- [x] Align training details with the [_paddle version_](../rtdetr_paddle/)
- [x] Tuning rtdetr based on [_pretrained weights_](https://github.com/lyuwenyu/RT-DETR/issues/42)
 -->
<!-- </details> -->

<details>
<summary>Install</summary>

```bash
bash ./install.sh
```

</details>

<details>
<summary>Data</summary>

- Download and extract COCO 2017 train and val images.

```
path/to/coco/
  annotations/  # annotation json files
  train2017/    # train images
  val2017/      # val images
```

- Modify config [`img_folder`, `ann_file`](configs/dataset/coco_detection.yml)
</details>

<details>
<summary>Training & Evaluation</summary>

- Training on a Single GPU:

```shell
# training on single-gpu
export CUDA_VISIBLE_DEVICES=0
python tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

<!--Training on Multiple GPUs:

```shell
# train on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml
```

- Evaluation on Multiple GPUs:

```shell
# val on multi-gpu
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 tools/train.py -c configs/rtdetr/rtdetr_r50vd_6x_coco.yml -r path/to/checkpoint --test-only
```
-->
</details>

<details>
<summary>Export</summary>

```shell
python tools/export_onnx.py -c configs/rtdetr/rtdetr_r18vd_6x_coco.yml -r path/to/checkpoint --check
```

</details>

<details open>
<summary>Train custom data</summary>

1. set `remap_mscoco_category: False`. This variable only works for ms-coco dataset. If you want to use `remap_mscoco_category` logic on your dataset, please modify variable [`mscoco_category2name`](https://github.com/lyuwenyu/RT-DETR/blob/main/rtdetr_pytorch/src/data/coco/coco_dataset.py#L154) based on your dataset.

2. add `-t path/to/checkpoint` (optinal) to tuning rtdetr based on pretrained checkpoint. see [training script details](./tools/README.md).
</details>

<details open> 
<summary>Penggunaan tools lainnya</summary>
  Harap gunakan --help pada tools yang tersedia untuk melihat parameter yang dibutuhkan
  
```shell
  python tools/file_name.py --help
```
</details>
