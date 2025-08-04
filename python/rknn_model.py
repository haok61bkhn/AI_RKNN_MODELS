import numpy as np
from rknn.api import RKNN


class RKNN_Model:
    def __init__(self, model_path):
        self.model_path = model_path
        self.rknn = None
        self.init()

    def init(self):
        self.rknn = RKNN()
        ret = self.rknn.load_rknn(self.model_path)
        if ret != 0:
            print("Load RKNN model failed")
            exit(ret)
        ret = self.rknn.init_runtime(
            target="rk3588",
            device_id=None,
            perf_debug=False,
            eval_mem=False,
            async_mode=False,
        )
        if ret != 0:
            print("Init runtime environment failed")
            exit(ret)

    def inference(self, inputs, data_format="NCHW"):
        if self.rknn is None:
            print("RKNN model not initialized")
            return None
        return self.rknn.inference(inputs=inputs, data_format=data_format) 