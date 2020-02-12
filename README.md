

[![License CC BY-NC-SA 4.0](https://img.shields.io/badge/license-CC4.0-blue.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode)
![Python 2.7](https://img.shields.io/badge/python-2.7-green.svg)
![Python 3.7](https://img.shields.io/badge/python-3.7-green.svg)
# NoduleNet: Decoupled False Positive Reduction for Pulmonary Nodule Detection and Segmentation
## License

Copyright (C) 2019 University of California Irvine and DEEPVOXEL Inc.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode). 

**This software is licensed for non-commerical research purpose only.**

## Paper

This paper has been accepted to MICCAI' 2019. 

If you use the code in your research, we would appreciate it if you can cite this paper.
```
@article{DBLP:journals/corr/abs-1907-11320,
  author    = {Hao Tang and
               Chupeng Zhang and
               Xiaohui Xie},
  title     = {NoduleNet: Decoupled False Positive Reductionfor Pulmonary Nodule
               Detection and Segmentation},
  journal   = {CoRR},
  volume    = {abs/1907.11320},
  year      = {2019},
  url       = {http://arxiv.org/abs/1907.11320},
  archivePrefix = {arXiv},
  eprint    = {1907.11320},
  timestamp = {Thu, 01 Aug 2019 08:59:33 +0200},
  biburl    = {https://dblp.org/rec/bib/journals/corr/abs-1907-11320},
  bibsource = {dblp computer science bibliography, https://dblp.org}
}
```

## Table of contents

<!--ts-->
* [Introduction](#introduction)
    * [Loss function](#loss-function)
    * [Training details](#training-details)
* [Install dependencies](#install-dependencies)
* [Usage](#usage)
    * [Preprocess](#preprocess)
    * [Train](#train)
    * [Evaluate](#evaluate)
    * [Cross validation](#6-fold-cross-validation)
<!--te-->

## Introduction

This is the codebase for performing pulmonary nodule detection and segmentation (end-to-end) using computed tomography (CT) images. Details about the methodolgy and results are in the aforementioned paper. Because of the limited space in the paper, we will elaborate the technical details here.

![demo](figures/demo_raw.gif) ![demo](figures/demo_pred.gif)

The detailed model architecture is demonstrated in the following figure. 
![model](figures/model.png)

### Loss function
First, we go into more details of the loss function of each branch.

1. Nodule candidate screening
The loss function is defined as:
<p align="center"><img alt="\begin{equation*}&#10;\begin{gathered}&#10;L_{ncs}=L(\{p_i\}, \{t_i\}) = \frac{1}{N_{cls}}\sum_i{L_{cls}(p_i, p_i^*)} + \lambda\frac{1}{N_{reg}}\sum_i{L_{reg}(t_i, t_i^*)}&#10;\end{gathered}&#10;\end{equation*}" src="svgs/f7289c09ca92409839bb26dc67cc40b4.png" align="middle" width="479.10059999999993pt" height="41.069655pt"/></p>

where <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.png" align="middle" width="5.642109000000004pt" height="21.602129999999985pt"/> is the index of <img alt="$i-th$" src="svgs/4467369d94a3da6705d94f709b6449e6.png" align="middle" width="41.03517pt" height="22.745910000000016pt"/> anchor in one CT image, <img alt="$p_i$" src="svgs/0d19b0a4827a28ecffa01dfedf5f5f2c.png" align="middle" width="12.873300000000002pt" height="14.102549999999994pt"/> is its predicted probability that this anchor contains nodule candidate and <img alt="$t_i$" src="svgs/02ab12d0013b89c8edc7f0f2662fa7a9.png" align="middle" width="10.547460000000003pt" height="20.14650000000001pt"/> is a vector denoting the six parameterized coordinate offsets of this anchor with respect to the ground truth box. <img alt="$\lambda$" src="svgs/fd8be73b54f5436a5cd2e73ba9b6bfa9.png" align="middle" width="9.553335pt" height="22.745910000000016pt"/> is a hyper parameter balancing the two losses and we set it to 1 in this work. <img alt="$N_{cls}$" src="svgs/16bed4a53895ad7c26042b06e5f06e79.png" align="middle" width="29.400690000000004pt" height="22.381919999999983pt"/> is the total number of anchors chosen for binary classification loss and <img alt="$N_{reg}$" src="svgs/d705c7c3026d48bcaeb62c0dde5ff9d3.png" align="middle" width="32.607135pt" height="22.381919999999983pt"/> is the total number of anchors considered for computing regression loss. <img alt="$p_i^*$" src="svgs/098a93c4be8edef77b847a44c342edd4.png" align="middle" width="14.949825pt" height="22.598730000000007pt"/> is 0 if <img alt="$i-th$" src="svgs/4467369d94a3da6705d94f709b6449e6.png" align="middle" width="41.03517pt" height="22.745910000000016pt"/> anchor does not contain any nodule and 1 otherwise. <img alt="$t_i^*$" src="svgs/84a5c5e9f2d2684cfd9f419d821bc97d.png" align="middle" width="12.623985000000003pt" height="22.598730000000007pt"/> is the ground truth vector for six regression terms and is formally defined as (we ignore subscript <img alt="$i$" src="svgs/77a3b857d53fb44e33b53e4c8b68351a.png" align="middle" width="5.642109000000004pt" height="21.602129999999985pt"/> for notational convenience):
<p align="center"><img alt="\begin{equation*}&#10;\begin{gathered}&#10;t^*= (t_z, t_y, t_x, t_d, t_h, t_w) \\&#10;\textrm{More specifically, }t^*=(\frac{z^*-z_a}{d_a}, \frac{y^*-y_a}{d_a}, \frac{x^*-x_a}{d_a}, \log\frac{h^*}{d_a}, \log\frac{w^*}{h_a}, \log\frac{d^*}{w_a})&#10;\end{gathered}&#10;\end{equation*}" src="svgs/4545390fdd240147e22a16f0d95a3493.png" align="middle" width="523.8453000000001pt" height="61.053794999999994pt"/></p>

<img alt="$z^*, y^*, x^*, d^*, h^*, w^*$" src="svgs/9fcf2369205da2f9a4f42d1badaca993.png" align="middle" width="137.38428pt" height="22.745910000000016pt"/> represent the center coordinates, depth, height and width of the ground truth box. <img alt="$z_a, y_a, x_a, d_a, h_a, w_a$" src="svgs/f715487b34d7ff3a88e60aabad96ba47.png" align="middle" width="137.99890499999998pt" height="22.745910000000016pt"/> denote those for the anchor box. We use weighted binary cross entropy loss with hard negative example mining (OHEM) and smooth <img alt="$L1$" src="svgs/a7713fc6174847c3f6c7785c3f1493fa.png" align="middle" width="19.33404pt" height="22.381919999999983pt"/> loss for <img alt="$L_{reg}$" src="svgs/e4d3b5da54a31ce1acbe209cc2b12229.png" align="middle" width="30.594135pt" height="22.381919999999983pt"/>. The foreground and background ratio of OHEM is set to 1:3 is used in this work.

2. Decoupled false positive reduction
The false positive reduction network minimizes the same multi-task in loss function as the NCS above, we call it <img alt="$L_{fps}$" src="svgs/62cfb29a42f36974211b9b372493d047.png" align="middle" width="31.753755pt" height="22.381919999999983pt"/>, where the <img alt="$L_{cls}$" src="svgs/d2afdea5d182323a600d40a197589665.png" align="middle" width="27.38769pt" height="22.381919999999983pt"/> is a weighted binary cross entropy loss and <img alt="$L_{reg}$" src="svgs/e4d3b5da54a31ce1acbe209cc2b12229.png" align="middle" width="30.594135pt" height="22.381919999999983pt"/> remains the same. (We actually do not need to predict the regression terms in the false positive reduction branch. We can only include the <img alt="$L_{cls}$" src="svgs/d2afdea5d182323a600d40a197589665.png" align="middle" width="27.38769pt" height="22.381919999999983pt"/> part.)

3. Segmentation refinement
We then transform the detected bounding box representation from <img alt="$\hat{z}, \hat{y}, \hat{x}, \hat{d}, \hat{h}, \hat{w}$" src="svgs/1b6f9723b21220eb899dfed2b975eeb0.png" align="middle" width="93.048285pt" height="31.421609999999983pt"/> to <img alt="$\hat{z_0}, \hat{y_0}, \hat{x_0}, \hat{z_1}, \hat{y_1}, \hat{x_1}$" src="svgs/9f62da41ae4da91c6e968cc5cb67a7ac.png" align="middle" width="130.007625pt" height="22.745910000000016pt"/>. These two representations are exactly the same. To solve the misalignment caused by the rounding of floating points, we round the bounding box coordinate to the nearest integer that is a multiple of four (because the smallest feature map for SR is feature_map_4, which is downsampled by a factor of 4). Also, to ensure the rounding operation would not make the detected region smaller (which may cut off some part of the nodule), we round <img alt="$\hat{z_0}, \hat{y_0}, \hat{x_0}$" src="svgs/a43d367b9352139ada4d8da9eb9f91c6.png" align="middle" width="60.918659999999996pt" height="22.745910000000016pt"/> to the biggest integer (that is a multiple of four) that is smaller than <img alt="$\hat{z_0}, \hat{y_0}, \hat{x_0}$" src="svgs/a43d367b9352139ada4d8da9eb9f91c6.png" align="middle" width="60.918659999999996pt" height="22.745910000000016pt"/>, and <img alt="$\hat{z_1}, \hat{y_1}, \hat{x_1}$" src="svgs/96d677fab4b4524891c3f42a771d3529.png" align="middle" width="60.918659999999996pt" height="22.745910000000016pt"/> to the smallest integer (that is a multiple of four) that is larger than <img alt="$\hat{z_1}, \hat{y_1}, \hat{x_1}$" src="svgs/96d677fab4b4524891c3f42a771d3529.png" align="middle" width="60.918659999999996pt" height="22.745910000000016pt"/>.

The segmentation refinement network minimizes the soft dice loss of the predicted mask sets <img alt="$\{m\}$" src="svgs/b5ee3381344a6dd7679fa361f4b3f3e2.png" align="middle" width="30.756165pt" height="24.56552999999997pt"/> and the ground truth mask sets <img alt="$\{g\}$" src="svgs/3d27770ce584b04d85d3680b733e4a8d.png" align="middle" width="24.778050000000004pt" height="24.56552999999997pt"/> of the input image.
<p align="center"><img alt="\begin{equation*}&#10;\begin{gathered}&#10;L_{sr}=L_{seg}(\{m\}, \{g\}) = \sum_n^{N_m} 1 - \frac{\sum_{i=1}^{N_{np}}m_{ni} g_{ni}}{\sum_{i=1}^{N_{np}}m_{ni}g_{ni} + \alpha \sum_{i=1}^{N_{np}}m_{ni}(1-g_{ni}) + \beta \sum_{i=1}^{N_{np}}(1-m_{ni})g_{ni} + \epsilon}&#10;\end{gathered}&#10;\end{equation*}" src="svgs/ef9fde738db6177fe4f78f7ac7b54419.png" align="middle" width="689.28585pt" height="48.10509pt"/></p>

where <img alt="$N_m$" src="svgs/f096527a4fbdda599925441329c0be49.png" align="middle" width="24.7797pt" height="22.381919999999983pt"/> is the total number of nodules in the input CT scan, <img alt="$N_{np}$" src="svgs/7c4a86eff93d7149bf58a1b30ab572e3.png" align="middle" width="28.005285000000004pt" height="22.381919999999983pt"/> is the number of pixels in the <img alt="$n-th$" src="svgs/042fbae12f6e472a11cde531985aca4e.png" align="middle" width="45.2232pt" height="22.745910000000016pt"/> nodule mask. <img alt="$m_{ni}$" src="svgs/a18b03c37cd939b719cc26a11719344d.png" align="middle" width="27.108345pt" height="14.102549999999994pt"/> and <img alt="$g_{ni}$" src="svgs/a36ca09d0c888f940bd2ec5314dc75da.png" align="middle" width="20.54052pt" height="14.102549999999994pt"/> denote the predicted probability of the <img alt="$i-th$" src="svgs/4467369d94a3da6705d94f709b6449e6.png" align="middle" width="41.03517pt" height="22.745910000000016pt"/> voxel of the <img alt="$n-th$" src="svgs/042fbae12f6e472a11cde531985aca4e.png" align="middle" width="45.2232pt" height="22.745910000000016pt"/> mask being a foreground, and the ground truth of that voxel respectively. <img alt="$\alpha$" src="svgs/c745b9b57c145ec5577b82542b2df546.png" align="middle" width="10.537065000000004pt" height="14.102549999999994pt"/> and <img alt="$\beta$" src="svgs/8217ed3c32a785f0b5aad4055f432ad8.png" align="middle" width="10.127700000000003pt" height="22.745910000000016pt"/> are parameters controlling the trade-off between false positives and false negatives, and we set them both to 0.5 in this work.

The whole framework is end-to-end. So the final loss is the sum of the losses from three branches:
<p align="center"><img alt="\begin{equation*}&#10;\begin{gathered}&#10;L = L_{ncs} + L_{fps} + L_{sr}&#10;\end{gathered}&#10;\end{equation*}" src="svgs/84781a981ecd0cf4bdbd070ba35158f4.png" align="middle" width="161.68251pt" height="15.885705pt"/></p>

There could be a term for each of the three losses to control the contribution of different branches. We just used a straightforward add here, because we thought the framework is not very sensible to this.

### Training details
The whole network was trained fully end-to-end. A gold standard nodule box is a rectangle cuboid that contains the nodule with a margin <img alt="$M=8$" src="svgs/a7acc6937b1a7e14cb6d4ce3bf7c082c.png" align="middle" width="47.738625000000006pt" height="22.381919999999983pt"/> to its border voxels in all three axes. For NCS an anchor that has intersection over union (IoU) equal to or greater than 0.5 with any gold standard nodule box is considered as positive, whereas an anchor that has an IoU less than 0.1 is considered as negative. 

For each gold standard nodule box, if there does not exist any anchor that has an IoU equal or greater than 0.5, then the anchor that has the largest IoU is considered as positive.  

Non-maximum suppression (NMS) is performed among all predicted proposals to reduce redundant predictions pointing at the same nodule.

nodule proposals that have IoUs equal to or greater than 0.5 are chosen as positive samples for training decoupled false positive reduction (DFPS) branch, whereas IoUs less than 0.2 are considered as negative. 

In order to mitigate the effect of failing to detect the regions that contain nodule by NCS during training, we add the gold standard nodule box when training the DFPS branch.

Since NMS may require a lot of time if the number of positive nodule proposals is large, as is often the case during the first several iterations especially using 3D images, NCS is trained first for a few epochs and then DFPS and segmentation are added to be trained jointly for more computational efficiency.

We trained NoduleNet for a total of 200 epochs in all experiment. We used SGD as optimizer, with initial learning rate set to 0.01 and <img alt="$L2$" src="svgs/8861562cc288b56bd011868d7cc5f7c3.png" align="middle" width="19.33404pt" height="22.381919999999983pt"/> penalty set to 0.0001. The learning rate decreases to 0.001 after 100 epochs and to 0.0001 after 160 epochs. Batch size was set to 16. NCS branch was first trained for 65 epochs. Next, DFPS branch was added for training for 15 more epochs. Lastly, nodule segmentation network was added for training for the rest 120 epochs. Training was done using 4 1080ti GPUs.

## Install dependencies

1. We recommend using Python >= 3.6 (Python 2.7 may also work), cude >= 9.0 and PyTorch 1.1 (https://pytorch.org). We highly recommend using conda for managing all the packages.
```
conda install -c conda-forge pydicom
conda install opencv
conda install tqdm
```

**Note: The packages listed here may not be complete. If you run into missing packages, you may want to google and install it.**

3. Install a custom module for bounding box NMS and overlap calculation.

```
cd build/box
python setup.py install
```

3. In order to use Tensorboard for visualizing the losses during training, we need to install tensorboard.

```
pip install tb-nightly  # Until 1.14 moves to the release channel
```

### Data

There are two types of annotations we will encounter: one that only has the location of the nodule, and another that has segmentation mask for the nodule. The first one is more common in practice, since contouring the nodule on images takes radiologists more time and effort. Our framework can handle both cases. 

The dataset that has nodule segmentation is the public LIDC dataset. More common dataset would be the ones like LUNA16 that only has nodule locations.

You may have noticed in the dataset folder, there are two readers for loading the two types of data we have: mask_reader.py and bbox_reader.py.


## Usage

### Preprocess

You can download the preprocessed data from [here](<https://drive.google.com/open?id=1UqcIn2NsdOYbmCCwhjmwSbCmpAJCQmSB>)

Or, you can run through the following data preprocessing pipeline.

First, we need to preprocess the dicom images. The preprocessing includes: segmenting lung regions from the CT image, resampling the image into 1x1x1 mm spacing and converting world coordinates to voxel coordinates. All the results will be saved as .npy files. (You would better to equip youself with the notion of world coordinate and voxel coordinate, and how they are converted to each other. Tutorials on this can be found in LUNA16 Challenge.)

Then, you will need to specify which samples to be used for training, validing and testing. This can be done by generating a csv files containing patient ids for each phase and specify their paths in the config.py

To be more specific about this paper, we hereby walk you through the configuration to generate the preprocessed data for training and evaluation. 

First, we will need to download those files: 
1. Download the LIDC-IDRI Radiologist Annotations/Segmentations (XML format) from https://wiki.cancerimagingarchive.net/download/attachments/1966254/LIDC-XML-only.zip?version=1&modificationDate=1530215018015&api=v2

    And change the config.py line 24 'annos_dir' to your downloaded path

2. Download the LIDC-IDRI CT images and the corresponding lung segmentation mask from LUNA16 challenge https://luna16.grand-challenge.org/Download/

    Move all downloaded CT images from the 10 folders to one folder, and change the config.py line 18 'data_dir' to the CT images directory (combining sub folders into one folder)

    And change the config.py line 27 'lung_mask_dir' to the segmentation mask.

    Explanantions on some intermediate results saved: ctr_arr_save_dir will be the place to save the parsed intermediate nodule masks for each annotator, and mask_save_dir is the folder to save the merged nodule mask. In mask_save_dir, there will be 1 - 4 four folders, containing nodule masks that are annotated by at least 1 - 4 annotators respectively. 

Then run 
```
cd utils/LIDC
python cvrt_annos_to_npy.py
```

Finally, we will resample the CT image to 1x1x1, as well as the nodule masks. All our training and evaluations are based on the resampled images. NoduleNet also works (training and testing) using CT images that are not resampled.

Go to utils/LIDC/preprocess.py, change lung_mask_dir to the lung segmentation mask you downloaded from step 2, nod_mask_dir to the mask_save_dir you specified in the previous step. Change save_dir to the place you want to save the preprocessed data. Then run
```
cd utils/LIDC
python preprocess.py
```

### Training

Change training configuration and data configuration in config.py, especially the path to your preprocessed data.

You can change network configuration in config.py, then run training script:

```
python train.py
```

We can visualize the losses during training using Tensorboard. E.g.
```
tensorboard --logdir=results/test_py37/runs --port=11001
```

### Pretrained weight
We have uploaded a model pretrained on 0 fold (https://drive.google.com/file/d/1B2aB_4HTw5OEe8jH6XPoQYRJw4hGuDXD/view?usp=sharing).

### Evaluation

Once training is done, you will need to change the 'initial_checkpoint' in config.py to the file path of the checkpoint you want to use. And then run the following command.

```
python test.py eval
```

You will see the results of FROC analysis both saved to files and printed on the screen.

### 6-fold cross validation
We performed 6-fold cross validation and following script can be used to run the 6-fold cross validation
```
cd scripts
bash cross_val_6fold.sh
```
We then manually concatenate the predictions of each fold into one csv, and then run the evaluation script.


## Acknowledgement
We used the code from [this repo](<https://github.com/zhwhong/lidc_nodule_detection>) to parse the lidc XML annotation.

Part of the code was adpated from [DSB2017 winning solution](<https://github.com/lfz/DSB2017>)
