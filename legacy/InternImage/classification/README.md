# InternImage for Image Classification

This folder contains the implementation of the InternImage for image classification.

<!-- TOC -->

- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Released Models](#released-models)
- [Evaluation](#evaluation)
- [Training from Scratch on ImageNet-1K](#training-from-scratch-on-imagenet-1k)
- [Manage Jobs with Slurm](#manage-jobs-with-slurm)
- [Training with DeepSpeed](#training-with-deepspeed)
- [Extracting Intermediate Features](#extracting-intermediate-features)
- [Export](#export)

<!-- TOC -->

## Installation

- Clone this repository:

```bash
git clone https://github.com/OpenGVLab/InternImage.git
cd InternImage
```

- Create a conda virtual environment and activate it:

```bash
conda create -n internimage python=3.9
conda activate internimage
```

- Install `CUDA>=10.2` with `cudnn>=7` following
  the [official installation instructions](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)
- Install `PyTorch>=1.10.0` and `torchvision>=0.9.0` with `CUDA>=10.2`:

For examples, to install `torch==1.11` with `CUDA==11.3`:

```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html
```

- Install `timm==0.6.11` and `mmcv-full==1.5.0`:

```bash
pip install -U openmim
mim install mmcv-full==1.5.0
mim install mmsegmentation==0.27.0
pip install timm==0.6.11 mmdet==2.28.1
```

- Install other requirements:

```bash
pip install opencv-python termcolor yacs pyyaml scipy
# Please use a version of numpy lower than 2.0
pip install numpy==1.26.4
pip install pydantic==1.10.13
```

- Compile CUDA operators

Before compiling, please use the `nvcc -V` command to check whether your `nvcc` version matches the CUDA version of PyTorch.

```bash
cd ./ops_dcnv3
sh ./make.sh
# unit test (should see all checking is True)
python test.py
```

- You can also install the operator using precompiled `.whl` files
  [DCNv3-1.0-whl](https://github.com/OpenGVLab/InternImage/releases/tag/whl_files)

## Data Preparation

We provide the following ways to prepare data:

<details open>
  <summary>Standard ImageNet-1K</summary>

- We use standard ImageNet dataset, you can download it from http://image-net.org/.

- For standard folder dataset, move validation images to labeled sub-folders. The file structure should look like:

  ```bash
  $ tree data
  imagenet
  ├── train
  │   ├── class1
  │   │   ├── img1.jpeg
  │   │   ├── img2.jpeg
  │   │   └── ...
  │   ├── class2
  │   │   ├── img3.jpeg
  │   │   └── ...
  │   └── ...
  └── val
      ├── class1
      │   ├── img4.jpeg
      │   ├── img5.jpeg
      │   └── ...
      ├── class2
      │   ├── img6.jpeg
      │   └── ...
      └── ...
  ```

</details>

<details>
  <summary>Zipped ImageNet-1K</summary>

- To boost the slow speed when reading images from massive small files, we also support zipped ImageNet, which includes
  four files:

  - `train.zip`, `val.zip`: which store the zipped folder for train and validate splits.
  - `train.txt`, `val.txt`: which store the relative path in the corresponding zip file and ground truth
    label. Make sure the data folder looks like this:

  ```bash
  $ tree data
  data
  └── ImageNet-Zip
      ├── train_map.txt
      ├── train.zip
      ├── val_map.txt
      └── val.zip

  $ head -n 5 meta_data/val.txt
  ILSVRC2012_val_00000001.JPEG	65
  ILSVRC2012_val_00000002.JPEG	970
  ILSVRC2012_val_00000003.JPEG	230
  ILSVRC2012_val_00000004.JPEG	809
  ILSVRC2012_val_00000005.JPEG	516

  $ head -n 5 meta_data/train.txt
  n01440764/n01440764_10026.JPEG	0
  n01440764/n01440764_10027.JPEG	0
  n01440764/n01440764_10029.JPEG	0
  n01440764/n01440764_10040.JPEG	0
  n01440764/n01440764_10042.JPEG	0
  ```

</details>

<details>
  <summary>ImageNet-22K</summary>

- For ImageNet-22K dataset, make a folder named `fall11_whole` and move all images to labeled sub-folders in this
  folder. Then download the train-val split
  file ([ILSVRC2011fall_whole_map_train.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_train.txt)
  & [ILSVRC2011fall_whole_map_val.txt](https://github.com/SwinTransformer/storage/releases/download/v2.0.1/ILSVRC2011fall_whole_map_val.txt))
  , and put them in the parent directory of `fall11_whole`. The file structure should look like:

  ```bash
    $ tree imagenet22k/
    imagenet22k/
    └── fall11_whole
        ├── n00004475
        ├── n00005787
        ├── n00006024
        ├── n00006484
        └── ...
  ```

</details>

<details>
  <summary>iNaturalist 2018</summary>

- For the iNaturalist 2018, please download the dataset from the [official repository](https://github.com/visipedia/inat_comp/blob/master/2018/README.md).
  The file structure should look like:

  ```bash
    $ tree inat2018/
    inat2018/
    ├── categories.json
    ├── test2018
    ├── test2018.json
    ├── train2018.json
    ├── train2018_locations.json
    ├── val2018
    ├── val2018.json
    └── val2018_locations.json
  ```

</details>

## Released Models

<details open>
<summary> Open-Source Visual Pretrained Models </summary>
<br>
<div>

|      name      |       pretrain       | resolution | #param |                                                                                  download                                                                                   |
| :------------: | :------------------: | :--------: | :----: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-L  |        IN-22K        |  384x384   |  223M  |     [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22k_192to384.pth)    \| [hf](https://huggingface.co/OpenGVLab/internimage_l_22k_384)      |
| InternImage-XL |        IN-22K        |  384x384   |  335M  |     [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22k_192to384.pth)   \| [hf](https://huggingface.co/OpenGVLab/internimage_xl_22k_384)     |
| InternImage-H  | Joint 427M -> IN-22K |  384x384   | 1.08B  | [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_jointto22k_384.pth)   \| [hf](https://huggingface.co/OpenGVLab/internimage_h_jointto22k_384)  |
| InternImage-G  | Joint 427M -> IN-22K |  384x384   |   3B   | [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_g_pretrainto22k_384.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_g_jointto22k_384) |

</div>

</details>

<details open>
<summary> ImageNet-1K Image Classification </summary>
<br>
<div>

|      name      |       pretrain       | resolution | acc@1 | #param | FLOPs |                                                                                                                download                                                                                                                 |
| :------------: | :------------------: | :--------: | :---: | :----: | :---: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-T  |        IN-1K         |  224x224   | 83.5  |  30M   |  5G   |          [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_t_1k_224.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_t_1k_224) \| [cfg](configs/without_lr_decay/internimage_t_1k_224.yaml)          |
| InternImage-S  |        IN-1K         |  224x224   | 84.2  |  50M   |  8G   |          [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_s_1k_224.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_s_1k_224) \| [cfg](configs/without_lr_decay/internimage_s_1k_224.yaml)          |
| InternImage-B  |        IN-1K         |  224x224   | 84.9  |  97M   |  16G  |          [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_b_1k_224.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_b_1k_224) \| [cfg](configs/without_lr_decay/internimage_b_1k_224.yaml)          |
| InternImage-L  |        IN-22K        |  384x384   | 87.7  |  223M  | 108G  |  [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_l_22kto1k_384.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_l_22kto1k_384) \| [cfg](configs/without_lr_decay/internimage_l_22kto1k_384.yaml)   |
| InternImage-XL |        IN-22K        |  384x384   | 88.0  |  335M  | 163G  | [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_xl_22kto1k_384.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_xl_22kto1k_384) \| [cfg](configs/without_lr_decay/internimage_xl_22kto1k_384.yaml) |
| InternImage-H  | Joint 427M -> IN-22K |  640x640   | 89.6  | 1.08B  | 1478G |  [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_22kto1k_640.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_h_22kto1k_640) \| [cfg](configs/without_lr_decay/internimage_h_22kto1k_640.yaml)   |
| InternImage-G  | Joint 427M -> IN-22K |  512x512   | 90.1  |   3B   | 2700G |  [pth](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_g_22kto1k_512.pth) \| [hf](https://huggingface.co/OpenGVLab/internimage_g_22kto1k_512) \| [cfg](configs/without_lr_decay/internimage_g_22kto1k_512.yaml)   |

</div>

</details>

<details open>
<summary> iNaturalist 2018 Image Classification </summary>
<br>
<div>

|     name      |       pretrain       | resolution | acc@1 | #param |                                                                                                                                  download                                                                                                                                  |
| :-----------: | :------------------: | :--------: | :---: | :----: | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: |
| InternImage-H | Joint 427M -> IN-22K |  384x384   | 92.6  |  1.1B  | [ckpt](https://huggingface.co/OpenGVLab/InternImage/resolve/main/internimage_h_22ktoinat18_384.pth) \| [cfg](configs/inaturalist2018/internimage_h_22ktoinat18_384.yaml) \| [log](https://huggingface.co/OpenGVLab/InternImage/raw/main/internimage_h_22ktoinat18_384.log) |

</div>

</details>

## Evaluation

To evaluate a pretrained `InternImage` on ImageNet val, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345 main.py --eval \
--cfg <config-file> --resume <checkpoint> --data-path <imagenet-path>
```

For example, to evaluate the `InternImage-B` with a single GPU:

```bash
python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --eval \
--cfg configs/internimage_b_1k_224.yaml --resume internimage_b_1k_224.pth --data-path <imagenet-path>
```

## Training from Scratch on ImageNet-1K

> The paper results were obtained from models trained with configs in `configs/without_lr_decay`.

To train an `InternImage` on ImageNet from scratch, run:

```bash
python -m torch.distributed.launch --nproc_per_node <num-of-gpus-to-use> --master_port 12345  main.py \
--cfg <config-file> --data-path <imagenet-path> [--batch-size <batch-size-per-gpu> --output <output-directory> --tag <job-tag>]
```

## Manage Jobs with Slurm

For example, to train or evaluate `InternImage` with slurm cluster, run:

<details open>
<summary> InternImage-T (IN-1K) </summary>
<br>

```bash
# Train for 300 epochs with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --batch-size 512
# Train for 300 epochs with 32 GPUs
GPUS=32 sh train_in1k.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --batch-size 128
# Evaluate on ImageNet-1K with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --resume pretrained/internimage_t_1k_224.pth --eval
```

</details>

<details>
<summary> InternImage-S (IN-1K) </summary>
<br>

```bash
# Train for 300 epochs with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_s_1k_224.yaml --batch-size 512
# Train for 300 epochs with 32 GPUs
GPUS=32 sh train_in1k.sh <partition> <job-name> configs/internimage_s_1k_224.yaml --batch-size 128
# Evaluate on ImageNet-1K with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_s_1k_224.yaml --resume pretrained/internimage_s_1k_224.pth --eval
```

</details>

<details>
<summary> InternImage-B (IN-1K) </summary>
<br>

```bash
# Train for 300 epochs with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_b_1k_224.yaml --batch-size 512
# Train for 300 epochs with 32 GPUs
GPUS=32 sh train_in1k.sh <partition> <job-name> configs/internimage_b_1k_224.yaml --batch-size 128
# Evaluate on ImageNet-1K with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_b_1k_224.yaml --resume pretrained/internimage_b_1k_224.pth --eval
```

</details>

<details>
<summary> InternImage-L (IN-22K to IN-1K) </summary>
<br>

```bash
# Train for 20 epochs with 32 GPUs
GPUS=32 sh train_in1k.sh <partition> <job-name> configs/internimage_l_22kto1k_384.yaml --batch-size 16
# Evaluate on ImageNet-1K with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_l_22kto1k_384.yaml --resume pretrained/internimage_l_22kto1k_384.pth --eval
```

</details>

<details>
<summary> InternImage-XL (IN-22K to IN-1K) </summary>
<br>

```bash
# Train for 20 epochs with 32 GPUs
GPUS=32 sh train_in1k.sh <partition> <job-name> configs/internimage_xl_22kto1k_384.yaml --batch-size 16
# Evaluate on ImageNet-1K with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_xl_22kto1k_384.yaml --resume pretrained/internimage_xl_22kto1k_384.pth --eval
```

</details>

<details>
<summary> InternImage-H (IN-22K to IN-1K) </summary>
<br>

```bash
# Train for 20 epochs with 32 GPUs
GPUS=32 sh train_in1k.sh <partition> <job-name> configs/internimage_h_22kto1k_640.yaml --batch-size 16
# Evaluate on ImageNet-1K with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_h_22kto1k_640.yaml --resume pretrained/internimage_h_22kto1k_640.pth --eval
```

</details>

<details>
<summary> InternImage-G (IN-22K to IN-1K) </summary>
<br>

```bash
# Train for 20 epochs with 64 GPUs
GPUS=64 sh train_in1k.sh <partition> <job-name> configs/internimage_g_22kto1k_512.yaml --batch-size 8
# Evaluate on ImageNet-1K with 8 GPUs
GPUS=8 sh train_in1k.sh <partition> <job-name> configs/internimage_g_22kto1k_512.yaml --resume pretrained/internimage_g_22kto1k_512.pth --eval
```

</details>

## Training with DeepSpeed

We support utilizing [DeepSpeed](https://github.com/microsoft/DeepSpeed) to reduce memory costs for training large-scale models, e.g. InternImage-H with over 1 billion parameters.
To use it, first install the requirements as

```bash
pip install deepspeed==0.8.3
```

Then you could launch the training in a slurm system with 8 GPUs as follows (tiny and huge as examples).
The default zero stage is 1 and it could config via command line args `--zero-stage`.

```
GPUS=8 GPUS_PER_NODE=8 sh train_in1k_deepspeed.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --batch-size 128 --accumulation-steps 4
GPUS=8 GPUS_PER_NODE=8 sh train_in1k_deepspeed.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --batch-size 128 --accumulation-steps 4 --eval --resume ckpt.pth
GPUS=8 GPUS_PER_NODE=8 sh train_in1k_deepspeed.sh <partition> <job-name> configs/internimage_t_1k_224.yaml --batch-size 128 --accumulation-steps 4 --eval --resume deepspeed_ckpt_dir
GPUS=8 GPUS_PER_NODE=8 sh train_in1k_deepspeed.sh <partition> <job-name> configs/internimage_h_22kto1k_640.yaml --batch-size 16 --accumulation-steps 4 --pretrained pretrained/internimage_h_jointto22k_384.pth
GPUS=8 GPUS_PER_NODE=8 sh train_in1k_deepspeed.sh <partition> <job-name> configs/internimage_h_22kto1k_640.yaml --batch-size 16 --accumulation-steps 4 --pretrained pretrained/internimage_h_jointto22k_384.pth --zero-stage 3
```

🤗 **HuggingFace Accelerate Integration of DeepSpeed**

Optionally, you could use our [HuggingFace Accelerate](https://github.com/huggingface/accelerate) integration to use DeepSpeed.

```bash
pip install accelerate==0.18.0
```

```bash
accelerate launch --config_file configs/accelerate/dist_8gpus_zero3_wo_loss_scale.yaml main_accelerate.py --cfg configs/internimage_h_22kto1k_640.yaml --data-path data/imagenet --batch-size 16 --pretrained pretrained/internimage_h_jointto22k_384.pth --accumulation-steps 4
accelerate launch --config_file configs/accelerate/dist_8gpus_zero3_offload.yaml main_accelerate.py --cfg configs/internimage_t_1k_224.yaml --data-path data/imagenet --batch-size 128 --accumulation-steps 4 --output output_zero3_offload
accelerate launch --config_file configs/accelerate/dist_8gpus_zero1.yaml main_accelerate.py --cfg configs/internimage_t_1k_224.yaml --data-path data/imagenet --batch-size 128 --accumulation-steps 4
```

**Memory Costs**

Here is the reference GPU memory cost for InternImage-H with 8 GPUs.

- total batch size = 512, 16 batch size for each GPU, gradient accumulation steps = 4.

| Resolution | Zero Stage | Cpu Offloading | Memory |
| :--------: | :--------: | :------------: | :----: |
|    640     |   zero1    |     False      | 22572  |
|    640     |   zero3    |     False      | 20000  |
|    640     |   zero3    |      True      | 19144  |
|    384     |   zero1    |     False      | 16000  |
|    384     |   zero3    |      True      | 11928  |

**Convert Checkpoints**

To convert DeepSpeed checkpoints to pytorch fp32 checkpoint, you could use the following snippet.

```python
from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, 'best.pth', tag='best')
```

Then, you could use `best.pth` as usual, e.g., `model.load_state_dict(torch.load('best.pth'))`

> Due to the lack of computational resources, the deepspeed training scripts are currently only verified for the first few epochs. Please fire an issue if you have problems for reproducing the whole training.

## Extracting Intermediate Features

To extract the features of an intermediate layer, you could use `extract_feature.py`.

For example, extract features of `b.png` from layers `patch_embed` and `levels.0.downsample` and save them to 'b.pth'.

```bash
python extract_feature.py --cfg configs/internimage_t_1k_224.yaml --img b.png --keys patch_embed levels.0.downsample --save --resume internimage_t_1k_224.pth
```

## Export

Install `mmdeploy` at first:

```shell
pip install mmdeploy==0.14.0
```

To export `InternImage-T` from PyTorch to ONNX, run:

```shell
python export.py --model_name internimage_t_1k_224 --ckpt_dir /path/to/ckpt/dir --onnx
```

To export `InternImage-T` from PyTorch to TensorRT, run:

```shell
git clone https://github.com/open-mmlab/mmdeploy.git
cd mmdeploy && git checkout v0.13.0
export MMDEPLOY_DIR=$(pwd)
# prepare our custom ops, you can find it at InternImage/tensorrt/modulated_deform_conv_v3
cp -r ../../tensorrt/modulated_deform_conv_v3 csrc/mmdeploy/backend_ops/tensorrt/

# build custom ops
mkdir -p build && cd build
cmake -DCMAKE_CXX_COMPILER=g++ -DMMDEPLOY_TARGET_BACKENDS=trt -DTENSORRT_DIR=${TENSORRT_DIR} -DCUDNN_DIR=${CUDNN_DIR} ..
make -j$(nproc) && make install

# install the mmdeploy after building custom ops
pip install -e .
cd ../

python export.py --model_name internimage_t_1k_224 --ckpt_dir /path/to/ckpt/dir --trt
```
