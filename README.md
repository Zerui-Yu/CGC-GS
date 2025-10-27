# CGC-GS
Code for "CGC-GS: Cross Geometric Cues Constrained Gaussian Splatting", accepted by KBS 2025. [[Paper]](https://doi.org/10.1016/j.knosys.2025.114630) 
# Results
 ![](./images/f1.jpg "Our results on public datasets.")
 Chamfer distance (↓) results on DTU dataset:
|                | 24   | 37   | 40   | 55   | 63   | 65   | 69   | 83   | 97   | 105  | 106  | 110  | 114  | 118  | 122  | Mean |
|----------------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|------|
| **2DGS**       | 0.51 | 0.96 | 0.38 | 0.43 | 1.02 | 0.89 | 0.80 | 1.37 | 1.24 | 0.70 | 0.68 | 1.50 | 0.41 | 0.72 | 0.52 | 0.81 |
| **PGSR**       | 0.34 | 0.58 | 0.29 | 0.29 | 0.78 | 0.58 | 0.54 | 1.01 | 0.73 | 0.51 | 0.49 | 0.69 | 0.31 | 0.37 | 0.38 | 0.53 |
| **Ours**       | 0.36 | 0.65 | 0.37 | 0.32 | 0.76 | 0.52 | 0.49 | 1.11 | 0.68 | 0.61 | 0.38 | 0.51 | 0.33 | 0.40 | 0.38 | 0.52 |

## Environment

The repository contains submodules, thus please check it out with 
```shell
conda create -n cgcgs python=3.8
conda activate cgcgs

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 #replace your cuda version
pip install -r requirements.txt
pip install submodules/diff-surfel-rasterization
pip install submodules/simple-knn
```
## Dataset Preprocess
Please download the preprocessed DTU dataset from [2DGS](https://surfsplatting.github.io/). You also need to download the ground truth point clouds from the [DTU dataset](https://roboimagedata.compute.dtu.dk/?page_id=36). 

The data folder should like this (We provide some examples in "data" file):
```shell
data
├── dtu_dataset
│   ├── dtu
│   │   ├── scan24
│   │   │   ├── images
│   │   │   ├── mask
│   │   │   ├── sparse
│   │   │   ├── cameras_sphere.npz
│   │   │   ├── cameras.npz
│   │   │   ├── normal (Your normal prior(png) from StableNormal or other sources)
│   │   │   └── depth_pro 
│   │   │       └── npy (Your depth prior(npy) from DepthPro or other sources)
│   │   └── ...
├── ├── dtu_eval
        ├── Points
        │   └── stl
        └── ObsMask

```

## Training and Evaluation
```shell
# DTU dataset
python train.py -s ./data/dtu_dataset/dtu/scan55 -m ./output/DTU/scan55 --data_device cuda --quiet -r 2 --test_iterations -1
```

## Mesh:
- Adjust max_depth and voxel_size based on the dataset, smaller voxel_size and larger max_depth values require more memory.
```shell
# Rendering and Extract Mesh
python render.py -m out_path --max_depth 10.0 --voxel_size 0.01
```

## Useful Params:
- --use_depth_prior, --use_normal_prior 1/0: If you don't have depth or normal prior, set them as 0.
- --use_mask 1/0: If your images are "RGBA" and want to train with Alpha masks, set it as 1.
- --use_multi_loss 1/0: With or without multi view regularization. If only high-quality rendering effects are required without the need for extremely detailed meshes, set it as 0.
- --scale_iter0, --scale_iter1: Replace Eq. (28) in the paper with a more direct approach to control the number of rounds for scale-driven splitting. This is set to 500 by default.

## Some Suggestions:
- The accuracy of depth priors is critical. To our knowledge, DepthPRO has not performed as expected when handling outdoor scenes.
- When memory is insufficient, try switching `data_device="cpu"`, which stores the prior information on the CPU.
