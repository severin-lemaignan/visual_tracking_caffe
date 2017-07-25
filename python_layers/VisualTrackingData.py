import caffe
import json
import yaml
import numpy as np

class VisualTrackingLayer(caffe.Layer):

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._layer_params = layer_params
        # default batch_size = 256
        self._batch_size = int(layer_params.get('batch_size', 256))
        self._resize = layer_params.get('resize', -1)
        self._mean_file = layer_params.get('mean_file', None)
        self._source = layer_params.get('source')

        print("Loading %s..." % self._source)
        try:
            with open(self._source, 'r') as f:
                raw = json.load(f)
                self._data = np.array([d[0] for d in raw])
                self._targets = np.array([d[1] for d in raw])
        except Exception as e:
            print(str(e))

        print("loading done.")

        assert len(bottom) == 0,            'requires no layer.bottom'
        assert len(top) == 2,               'requires a two layer.top'

        self._current_idx = 0



    def get_next_minibatch(self):
        """Generate next mini-batch
        The return value is array of numpy array: [data, label]
        Reshape funcion will be called based on resutls of this function
        Needs to implement in each class
        """


        if (self._current_idx + self._batch_size) > len(self._data):
            self._current_idx = 0

        res = [   self._data[self._current_idx:self._current_idx + self._batch_size],
               self._targets[self._current_idx:self._current_idx + self._batch_size]]

        self._current_idx += self._batch_size

        return res


    def forward(self, bottom, top):
        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].reshape(*(blob[i].shape))
            top[i].data[...] = blob[i].astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].reshape(*(blob[i].shape))
