Gaze estimation using Caffe
===========================

A Caffe model to estimate gaze for the freeplay sandbox dataset

Generating the training and testing dataset
-------------------------------------------

Tools for generate the training and testing datasets for the [freeplay sandbox
dataset](https://freeplay-sandbox.github.io/) are available in the
[freeplay-sandbox-analysis](https://github.com/freeplay-sandbox/analysis)
repository. Make sure you have installed this repository first, including the
support for OpenPose if you need to extract the skeleton poses from the dataset.

**Steps** (using tools from the [freeplay-sandbox-analysis](https://github.com/freeplay-sandbox/analysis)
repo):

0. [if you need to re-extract the children' poses -- otherwise, if available, just use the
   `visual_tracking.poses.json` files provided with the dataset] Extract the children' poses from each frame:

```
$ ./scripts/extract_poses_visual_tracking.sh <dataset root>
```

This step relies on OpenPose and is very GPU intensive. Might take several days!

1. Match the 2D location of the visual target with the pose:

```
$ ./scripts/prepare_visual_tracking_dataset.sh <dataset root>
```

2. Collate the different `visual_tracking_dataset.json` into one
   `visual_tracking_full_dataset.json` and split it into into `visual_tracking_full_dataset.train.json` and `visual_tracking_full_dataset.test.json`

```
$ ./scripts/assemble_visual_tracking_dataset.py --multisets <dataset root>
```

This should finally result into 2 files: `visual_tracking_full_dataset.train.json` and `visual_tracking_full_dataset.test.json` that can be used to train and test the network.


Training
--------

Train with:
```
$ caffe train -log_dir logs -solver train.prototxt
```

Estimation
----------

`estimate_gaze` can display in real-time the gaze estimation.

Body and facial features from the freeplay-sandbox dataset need to be provided with the `replay_with_poses` utility:

```
$ replay_with_poses --continuousgaze=on --topics camera_yellow/rgb/image_raw/compressed camera_purple/rgb/image_raw/compressed --path <path to the dataset root>/data/2017-06-12-143746652201/ | ./estimate_gaze
```


