<!-- 1 -->
# Robust Weight Signatures
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

The official implementation of paper [Towards Accurate and Efficient 3D Object Detection for Autonomous Driving: A Mixture of Experts Computing System on Edge](https://arxiv.org/abs/2507.04123) [ICCV2025].

Linshen Liu, Boyan Su, Junyue Jiang, Guanlin Wu, Cong Guo, Ceyu Xu, Hao Frank Yang

<!-- 2 abstract  --> 
## Abstract
This paper presents Edge-based Mixture of Experts (MoE) Collaborative Computing (EMC2), an optimal computing system designed for autonomous vehicles (AVs) that simultaneously achieves low-latency and high-accuracy 3D object detection. Unlike conventional approaches, EMC2 incorporates a scenario-aware MoE architecture specifically optimized for edge platforms. By effectively fusing LiDAR and camera data, the system leverages the complementary strengths of sparse 3D point clouds and dense 2D images to generate robust multimodal representations. To enable this, EMC2 employs an adaptive multimodal data bridge that performs multi-scale preprocessing on sensor inputs, followed by a scenario-aware routing mechanism that dynamically dispatches features to dedicated expert models based on object visibility and distance. In addition, EMC2 integrates joint hardware-software optimizations, including hardware resource utilization optimization and computational graph simplification, to ensure efficient and real-time inference on resource-constrained edge devices. Experiments on open-source benchmarks clearly show the EMC2 advancements as a end-to-end system. On the KITTI dataset, it achieves an average accuracy improvement of 3.58% and a 159.06% inference speedup compared to 15 baseline methods on Jetson platforms, with similar performance gains on the nuScenes dataset, highlighting its capability to advance reliable, real-time 3D object detection tasks for AVs.
<!-- 3 here is the figure  -->  
![avatar](framework.png)

<!-- 4 here is the installation requirement  -->   
## Requirements
Requirements are provided in ``requirements.txt``.

<!-- 5 here is training and installation code  -->   
## Model Training and Testing
The model is based on Openpcdet framework. Here is the detail process. 
<!-- Models preparation - standard models.
```
python train_corruption.py \
    --pretrained --lr 0.01 \
    --dataset <dataset> --arch <arch> \
    --data <training data> --std_data <standard data> \
    --pretrained_path <pre-trained model path> --save_dir <path for checkpoint>
```

Models preparation - robust models.
```
python train_corruption.py \
    --pretrained --lr 0.01 \
    --dataset <dataset> --arch <arch> --corruption <corruption type> \
    --data <training data> --std_data <standard data> \
    --pretrained_path <pre-trained model path> --save_dir <path for checkpoint>
```

RWS extraction and Model Patching.
```
python model_patching.py --keep_num <num of layers used> --dataset <dataset> --arch <arch> \
        --corruption <corruption type> --serverity <severity level> --data <std testing data> --corruption_data <corrupted testing data>
        --corruption_model_root <root to store all robust models> \
        --base_model <root to base model> --pretrained <root to pretrained model --save_log <path to save log>
``` -->
<!-- 6 here is the install and training code  --> 
## Citation
If you find this useful, please cite the following paper:
```
@article{liu2025EMC2,
  title={Towards Accurate and Efficient 3D Object Detection for Autonomous Driving: A Mixture of Experts Computing System on Edge},
  author={Linshen Liu, Boyan Su, Junyue Jiang, Guanlin Wu, Cong Guo, Ceyu Xu, Hao Frank Yang},
  journal={arXiv preprint arXiv:2507.04123},
  year={2025}
}
```