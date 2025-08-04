from rknn.api import RKNN
rknn = RKNN()
rknn.config(
    target_platform='rk3588',
    mean_values=[[127.5, 127.5, 127.5]],
    std_values=[[255, 255, 255]],
    quantized_dtype='w8a8'
)

ret = rknn.load_onnx(model='iresnet124_1.onnx')
if ret != 0:
    print('Load ONNX failed.')
    exit(ret)

ret = rknn.build(do_quantization=True, dataset='aligned_datasets.txt')
if ret != 0:
    print('Build failed.')
    exit(ret)

ret = rknn.export_rknn('iresnet124_1_quantized.rknn')
if ret != 0:
    print('Export failed.')
    exit(ret)

rknn.release()
