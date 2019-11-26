## [DBG-dense-boundary-generator](https://arxiv.org/pdf/1911.04127.pdf)

[![License](https://img.shields.io/badge/license-BSD-blue.svg)](LICENSE)

By [Chuming Lin*](https://github.com/linchuming), [Jian Li*](https://lijiannuist.github.io/), [Yabiao Wang](https://github.com/ChaunceyWang), [Ying Tai](https://tyshiwo.github.io/), Donghao Luo, Zhipeng Cui, Chengjie Wang, Jilin Li, Feiyue Huang, Rongrong Ji.

*indicates equal contributions

## Update

* 2019.11.12: Release tensorflow-version DBG inference code.
* 2019.11.11: DBG is accepted by AAAI2020.
* 2019.11.08: Our ensemble DBG ranks No.1 on [ActivityNet](http://activity-net.org/challenges/2019/evaluation.html) 

## Introduction
In this repo, we propose a novel and unified action detection framework, named DBG, with superior performance over the state-of-the-art action detectors [BSN](https://arxiv.org/abs/1806.02964) and [BMN](https://arxiv.org/abs/1907.09702). You can use the code to evaluate our DBG for action proposal generation or action detection. For more details, please refer to our paper [Fast Learning of Temporal Action Proposal via Dense Boundary Generator](https://arxiv.org/pdf/1911.04127.pdf)!

## Contents

* [Paper Introduction](#paper-introduction)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
* [Citation](#citation)
* [Contact](#contact)

## Paper Introduction

 <img src="./figures/frameworkv2.PNG" width = "1000px" alt="image" align=center />

This paper introduces a novel and unified temporal action proposal generator named Dense Boundary Generator (DBG). In this work, we propose dual stream BaseNet to generate two different level and more discriminative features. We then adopt a temporal boundary classification module to predict precise temporal boundaries, and an action-aware completeness regression module to provide reliable action completeness confidence.

### ActivityNet1.3 Results
<p align='center'>
 <img src="./figures/ActivityNet.PNG" width = "800px" alt="image" align=center />
</p>

### THUMOS14 Results
<p align='center'>
 <img src="./figures/THUMOS14v3.PNG" width = "1000px" alt="image" align=center />
</p>

### Qualitative Results
<p align='center'>
  <img src='./figures/Qualitative.PNG' width=800'/>
</p>

## Prerequisites

- Tensorflow == 1.9.0
- Python == 3.6
- NVIDIA GPU == Tesla P40 
- Linux CUDA 9.0 CuDNN

## Getting Started

### Installation

Clone the github repository. We will call the cloned directory as `$DBG_ROOT`.
```bash
git clone https://github.com/TencentYoutuResearch/ActionDetection-DBG.git
cd ActionDetection-DBG
export CUDA_VISIBLE_DEVICES=0
```

### Download Datasets

Prepare ActivityNet 1.3 dataset. You can use [official ActivityNet downloader](https://github.com/activitynet/ActivityNet/tree/master/Crawler) to download videos from the YouTube. Some videos have been deleted from YouTube，and you can also ask for the whole dataset by email.

Extract visual feature, we adopt TSN model pretrained on the training set of ActivityNet, Please refer this repo [TSN-yjxiong](https://github.com/yjxiong/temporal-segment-networks) to extract frames and optical flow and refer this repo [anet2016-cuhk](https://github.com/yjxiong/anet2016-cuhk) to find pretrained TSN model.

For convenience of training and testing, we rescale the feature length of all videos to same length 100, and we provide the 19993 rescaled feature at here [Google Cloud](https://drive.google.com/file/d/1MYzegWXgfZd-DD9gi_GPyZ_YAN5idiFV/view?usp=sharing) or [微云](https://share.weiyun.com/5FD85UY).

### Compile Proposal Feature Generation Layer
Please make sure your Tensorflow==1.9.0 and Python==3.6

```
cd custom_op/src
make
```

### Runing of DBG

Pretrained model is included in `output/pretrained_model` and set parameters on `config.yaml`.
Please check the `feat_dir` in `config.yaml` and use a script to run DBG.

```
bash auto_run.sh
```

This script contains:
#### 1. Training
```
python train.py ./config.yaml
```
#### 2. Testing
```
python test.py ./config.yaml
```
#### 3. Evaluating
```
python post_processing.py output/result output/result_proposal.json
python eval.py output/result_proposal.json
```


## Citation
If you find DBG useful in your research, please consider citing: 
```
@inproceedings{DBG2020arXiv,
  author    = {Chuming Lin*, Jian Li*, Yabiao Wang, Ying Tai, Donghao Luo, Zhipeng Cui, Chengjie Wang, Jilin Li, Feiyue Huang, Rongrong Ji},
  title     = {Fast Learning of Temporal Action Proposal via Dense Boundary Generator},
  booktitle   = {AAAI Conference on Artificial Intelligence},
  year      = {2020},
}
```

## Contact
For any question, please file an issue or contact
```
Jian Li: swordli@tencent.com
```
腾讯优图实验室提出一种新的视频动作检测算法DBG并开源，这是继今年4月人脸检测算法DSFD开源后，优图的又一次开源动作。
DBG算法在全球两大权威视频动作数据集ActivityNet-1.3和THUMOS14上均取得了第一。相关论文《Fast Learning of Temporal Action Proposal via Dense Boundary Generator》已被国际人工智能顶级会议AAAI2020接收，与此同时，算法代码已在优图研究官方Github上开源。
视频动作检测技术是精彩视频集锦、视频字幕生成、动作识别等任务的基础，随着互联网的飞速发展，在产业界中得到越来越广泛地应用。视频动作检测算法需要在给出视频中找到动作发生的位置（起点和终点）和置信度。
如下图所示，我们需要在一段体育视频场景中，找到“跳高”这个动作发生的起点时刻和终点时刻。
（1）提出一种快速的、端到端的稠密边界动作生成器(Dense Boundary Generator，DBG)。该生成器能够对所有的动作提名（proposal）估计出稠密的边界置信度图。

（2）引入额外的时序上的动作分类损失函数来监督动作概率特征，该特征能够有利于回归动作的完整度。

（3）设计一种高效的动作提名特征生成层，该层能够有效捕获动作的全局特征，便于后面的分类和回归模块。

据了解，DBG算法在全球两大权威视频动作数据集ActivityNet-1.3和THUMOS14上均取得了第一。算法的整体框架主要包括三个部分：视频特征抽取（Video Representation），稠密边界动作检测器（DBG），后处理（Post-processing）。
2017年7月，腾讯优图首个AI开源项目NCNN开源，2019年至今已有人脸检测、属性等5项技术开源。腾讯优图将不断研究和探索，通过开源更多新技术为产业发展提供助力。
