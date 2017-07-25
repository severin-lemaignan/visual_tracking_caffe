import numpy as np
import caffe
import cv2
import sys
import json
from collections import deque

class valuefilter:

    MAX_LENGTH=10

    def __init__(self, maxlen = MAX_LENGTH):
        self._vals = deque(maxlen = maxlen)

        self.lastval = 0.
        self.dirty = True

    def append(self, val):
        self.dirty = True
        self._vals.append(val)

    def get(self):

        if self.dirty:
            self.lastval = sum(self._vals) / len(self._vals)
            self.dirty = False

        return self.lastval


SANDTRAY_LENGTH=600
SANDTRAY_WIDTH=340

data = [
        [0.329,0.4009,0.3362,0.3907,0.347,0.3894,0.357,0.3945,0.347,0.4009,0.3376,0.4047,0.3075,0.369,0.3175,0.3498,0.3319,0.3447,0.3477,0.3421,0.3635,0.3485,0.4015,0.3843,0.4102,0.3715,0.4217,0.3702,0.4303,0.3792,0.4224,0.3881,0.4109,0.3894,0.4015,0.3498,0.4094,0.3409,0.4181,0.3294,0.4317,0.3268,0.4403,0.3332,0.3419,0.3958,0.4159,0.3779,0.384,0.4383,0.4085,0.6452,0.2861,0.7024,0.5172,0.5935,0.3397,0.3975,0.4147,0.3783,0.2984,0.4329,0.4605,0.3893],
        [0.2942,0.469,0.3028,0.4655,0.3101,0.4596,0.3195,0.4584,0.3101,0.4631,0.3035,0.4679,0.2762,0.4323,0.2835,0.4205,0.2962,0.4158,0.3088,0.4134,0.3188,0.4205,0.3614,0.4394,0.368,0.43,0.3787,0.4276,0.388,0.4264,0.3793,0.4347,0.3693,0.4394,0.3461,0.4004,0.3567,0.3838,0.37,0.3744,0.384,0.3767,0.3953,0.3815,0.3068,0.4655,0.3733,0.4312,0.3458,0.4928,0.3948,0.6507,0.2754,0.6997,0.4958,0.6044,0.3075,0.4601,0.3733,0.4247,0.2738,0.4655,0.4331,0.4056],
        [0.4037,0.5212,0.4066,0.5151,0.4111,0.5099,0.4155,0.5142,0.412,0.5195,0.4076,0.5221,0.3933,0.4871,0.3958,0.481,0.3973,0.4783,0.4017,0.4775,0.4047,0.4783,0.4396,0.495,0.445,0.488,0.4514,0.4871,0.4588,0.4888,0.4539,0.4967,0.447,0.5002,0.4298,0.4652,0.4386,0.4573,0.4475,0.4521,0.4588,0.453,0.4667,0.4591,0.4096,0.5151,0.4494,0.4932,0.4193,0.5309,0.4774,0.716,0.3825,0.7241,0.5707,0.7078,0.407,0.5092,0.4468,0.4899,0.0,0.0,0.5096,0.5118]
        ]

base = cv2.imread('res/map.png')
base = cv2.resize(base,(600, 340))
pts1 = np.float32([[0,0],[0,340],[600,340],[600,0]])
#pts2 = np.float32([[100,600],[600,500],[400,100],[50,100]])
pts2 = np.float32([[50,550],[550,550],[450,50],[150,50]])
M = cv2.getPerspectiveTransform(pts1,pts2)

caffe.set_device(0)
caffe.set_mode_gpu()
#caffe.set_mode_cpu()

net = caffe.Net('deploy.prototxt', 'snapshots/visual_tracking_iter_15000000.caffemodel', caffe.TEST)

Xfilters = {}
Yfilters = {}

print("Waiting for data on stdin...")
#for d in data:
while True:
    l = sys.stdin.readline()
    d = json.loads(l)
    topic = d["topic"]
    mirror = d["mirror"]
    features = d["features"]

    net.blobs['bodyfeatures'].data[...] = np.array(features).reshape(1,len(features))

    net.forward()
    result = net.blobs["result"].data


    Xfilters.setdefault(topic, valuefilter()).append(result[0][0] * SANDTRAY_LENGTH)
    if mirror:
        Yfilters.setdefault(topic, valuefilter()).append((1-result[0][1]) * SANDTRAY_WIDTH)
    else:
        Yfilters.setdefault(topic, valuefilter()).append(result[0][1] * SANDTRAY_WIDTH)

    #print("Gaze at (%dcm, %dcm)" % (result[0][0] * SANDTRAY_LENGTH, result[0][1] * SANDTRAY_WIDTH))

    img = base.copy()

    # draw all the gaze targets with their specific colors
    for t in Xfilters.keys():
        color = (255,255,255)
        if "purple" in t:
            color = (250,0,150)
        if "yellow" in t:
            color = (0,250,250)
        cv2.circle(img,(int(Xfilters[t].get()), int(Yfilters[t].get())), 4, color, -1, cv2.LINE_AA)
        cv2.circle(img,(int(Xfilters[t].get()), int(Yfilters[t].get())), 20, color, 2, cv2.LINE_AA)

    dst = cv2.warpPerspective(img,M,(600,600))
    cv2.imshow('image',dst)
    cv2.waitKey(10)

cv2.destroyAllWindows()
