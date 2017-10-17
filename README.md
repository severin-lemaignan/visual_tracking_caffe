Gaze estimation using Caffe
===========================

A Caffe model to estimate gaze for the freeplay sandbox dataset

Training
--------

Train with:
```
$ caffe train -log_dir logs -solver train.prototxt
```

Estimation
----------

`estimate_gaze` can display in real-time the gaze estimation.

Body and facial features from the freeplay-sandbox dataset need to be provided with the `reaply_with_poses` utility:

```
$ replay_with_poses --continuousgaze=on --topics camera_yellow/rgb/image_raw/compressed camera_purple/rgb/image_raw/compressed --path <path to the dataset root>/data/2017-06-12-143746652201/ | ./estimate_gaze
```


