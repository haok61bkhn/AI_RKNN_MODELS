from rknn.api import RKNN
import os
import glob


def convert_onnx_to_rknn(
    onnx_path,
    output_path,
    mean_values,
    std_values,
    dataset_path="dataset.txt",
    target_platform="rk3588",
    quantized_dtype="w8a8",
    using_int8=True,
):
    rknn = RKNN()

    rknn.config(
        target_platform=target_platform,
        mean_values=mean_values,
        std_values=std_values,
        quantized_dtype=quantized_dtype,
    )

    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f"Load ONNX failed for {onnx_path}")
        rknn.release()
        return False

    ret = rknn.build(do_quantization=using_int8, dataset=dataset_path)
    if ret != 0:
        print(f"Build failed for {onnx_path}")
        rknn.release()
        return False

    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print(f"Export failed for {onnx_path}")
        rknn.release()
        return False

    rknn.release()
    print(f"Successfully converted {onnx_path} to {output_path}")
    return True


if __name__ == "__main__":

    # #scrfd_500m_320_1.onnx
    # dataset_path = 'dataset.txt'
    # onnx_path = 'scrfd_500m_320_1.onnx'
    # output_path = 'scrfd_500m_320_1_quantized.rknn'
    # mean_values = [[0, 0, 0]]
    # std_values = [[128, 128, 128]]
    # convert_onnx_to_rknn(onnx_path, output_path, mean_values, std_values, dataset_path)

    # iresnet124_1.onnx
    # dataset_path = 'aligned_datasets.txt'
    # onnx_path = 'iresnet124_1.onnx'
    # output_path = 'iresnet124_1_quantized.rknn'
    # mean_values = [[127.5, 127.5, 127.5]]
    # std_values = [[255, 255, 255]]
    # convert_onnx_to_rknn(onnx_path, output_path, mean_values, std_values, dataset_path,"rk3588","w8a8")

    # #AntiSpoofing_print-replay_1.5_128.onnx
    # dataset_path = 'aligned_datasets.txt'
    # onnx_path = 'AntiSpoofing_print-replay_1.5_128.onnx'
    # output_path = 'AntiSpoofing_print-replay_1.5_128_quantized.rknn'
    # mean_values = [[0, 0, 0]]
    # std_values = [[255, 255, 255]]
    # convert_onnx_to_rknn(onnx_path, output_path, mean_values, std_values, dataset_path)

    # #2.7_80x80_MiniFASNetV2.onnx
    # dataset_path = 'aligned_datasets.txt'
    # onnx_path = '2.7_80x80_MiniFASNetV2.onnx'
    # output_path = '2.7_80x80_MiniFASNetV2_quantized.rknn'
    # mean_values = [[0, 0, 0]]
    # std_values = [[1, 1, 1]]
    # convert_onnx_to_rknn(onnx_path, output_path, mean_values, std_values, dataset_path)

    # #4_0_0_80x80_MiniFASNetV1SE_1.onnx
    # dataset_path = 'aligned_datasets.txt'
    # onnx_path = '4_0_0_80x80_MiniFASNetV1SE.onnx'
    # output_path = '4_0_0_80x80_MiniFASNetV1SE_quantized.rknn'
    # mean_values = [[0, 0, 0]]
    # std_values = [[1, 1, 1]]
    # convert_onnx_to_rknn(onnx_path, output_path, mean_values, std_values, dataset_path)

    # obb_detection_yolov11.onnx
    # dataset_path = "obb_datasets.txt"
    # onnx_path = "text_obb.onnx"
    # output_path = "text_obb_quantized.rknn"
    # mean_values = [[0, 0, 0]]
    # std_values = [[255, 255, 255]]
    # convert_onnx_to_rknn(
    #     onnx_path,
    #     output_path,
    #     mean_values,
    #     std_values,
    #     dataset_path,
    #     using_int8=False,
    # )

    # text_recognition_yolov11.onnx
    dataset_path = "text_crop.txt"

    onnx_path = "text_recognition_yolov11.onnx"
    output_path = "text_recognition_yolov11_quantized.rknn"
    mean_values = [[0, 0, 0]]
    std_values = [[255, 255, 255]]
    convert_onnx_to_rknn(
        onnx_path,
        output_path,
        mean_values,
        std_values,
        dataset_path,
        using_int8=False,
    )
