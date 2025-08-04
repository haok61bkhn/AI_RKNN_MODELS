import cv2
import numpy as np
from rknn_model import RKNN_Model
import os
import shutil
import glob
import time


def letterbox(
    im,
    new_shape,
    color=(114, 114, 114),
    auto=False,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    shape = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return im, ratio, (dw, dh)


def clip_boxes(boxes, shape):
    boxes[..., [0, 2]] = boxes[..., [0, 2]].clip(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clip(0, shape[0])


def scale_boxes(
    img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False
):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1), round(
            (img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1
        )
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]
    if padding:
        boxes[..., [0]] -= pad[0]
        boxes[..., [1]] -= pad[1]
        if not xywh:
            boxes[..., 2] -= pad[0]
            boxes[..., 3] -= pad[1]
    boxes[..., :4] /= gain
    clip_boxes(boxes, img0_shape)
    return boxes


def xywhr2xyxyxyxy(center):
    cos, sin = np.cos, np.sin
    ctr = center[..., :2]
    w, h, angle = (center[..., i : i + 1] for i in range(2, 5))
    cos_value, sin_value = cos(angle), sin(angle)
    vec1 = [w / 2 * cos_value, w / 2 * sin_value]
    vec2 = [-h / 2 * sin_value, h / 2 * cos_value]
    vec1 = np.concatenate(vec1, axis=-1)
    vec2 = np.concatenate(vec2, axis=-1)
    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2
    return np.stack([pt1, pt2, pt3, pt4], axis=-2)


def get_covariance_matrix(boxes):
    gbbs = np.concatenate((np.power(boxes[:, 2:4], 2) / 12, boxes[:, 4:]), axis=-1)
    a, b, c = np.split(gbbs, [1, 2], axis=-1)
    return (
        a * np.cos(c) ** 2 + b * np.sin(c) ** 2,
        a * np.sin(c) ** 2 + b * np.cos(c) ** 2,
        a * np.cos(c) * np.sin(c) - b * np.sin(c) * np.cos(c),
    )


def batch_probiou(obb1, obb2, eps=1e-7):
    x1, y1 = np.split(obb1[..., :2], 2, axis=-1)
    x2, y2 = (x.squeeze(-1)[None] for x in np.split(obb2[..., :2], 2, axis=-1))
    a1, b1, c1 = get_covariance_matrix(obb1)
    a2, b2, c2 = (x.squeeze(-1)[None] for x in get_covariance_matrix(obb2))
    t1 = (
        ((a1 + a2) * (np.power(y1 - y2, 2)) + (b1 + b2) * (np.power(x1 - x2, 2)))
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.25
    t2 = (
        ((c1 + c2) * (x2 - x1) * (y1 - y2))
        / ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)) + eps)
    ) * 0.5
    t3 = (
        np.log(
            ((a1 + a2) * (b1 + b2) - (np.power(c1 + c2, 2)))
            / (
                4
                * np.sqrt(
                    (a1 * b1 - np.power(c1, 2)).clip(0)
                    * (a2 * b2 - np.power(c2, 2)).clip(0)
                )
                + eps
            )
            + eps
        )
        * 0.5
    )
    bd = t1 + t2 + t3
    bd = np.clip(bd, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1 - hd


def numpy_nms_rotated(boxes, scores, iou_threshold):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.int8)
    sorted_idx = np.argsort(scores)[::-1]
    boxes = boxes[sorted_idx]
    ious = batch_probiou(boxes, boxes)
    ious = np.triu(ious, k=1)
    pick = np.nonzero(np.max(ious, axis=0) < iou_threshold)[0]
    return sorted_idx[pick]


def non_max_suppression(
    prediction,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=300,
    nc=0,
    max_nms=30000,
    max_wh=7680,
):
    bs = prediction.shape[0]
    nm = prediction.shape[1] - nc - 4
    mi = 4 + nc
    xc = np.amax(prediction[:, 4:mi], axis=1) > conf_thres
    multi_label = nc > 1
    prediction = np.transpose(prediction, (0, 2, 1))
    output = [np.zeros((0, 6 + nm))] * bs

    for xi, x in enumerate(prediction):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box = x[:, :4]
        cls = x[:, 4 : 4 + nc]
        mask = x[:, 4 + nc : 4 + nc + nm]
        if multi_label:
            i, j = np.where(cls > conf_thres)
            x = np.concatenate(
                (box[i], x[i, 4 + j, None], j[:, None].astype(float), mask[i]), axis=1
            )
        else:
            conf = np.max(cls, axis=1, keepdims=True)
            j = np.argmax(cls, axis=1, keepdims=True)
            x = np.concatenate((box, conf, j.astype(float), mask), axis=1)[
                conf.flatten() > conf_thres
            ]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[np.argsort(x[:, 4])[::-1][:max_nms]]
        c = max_wh
        scores = x[:, 4]
        boxes = np.concatenate((x[:, :2] + c, x[:, 2:4], x[:, -1:]), axis=-1)
        i = numpy_nms_rotated(boxes, scores, iou_thres)
        i = i[:max_det]
        output[xi] = x[i]
    return output


class TextDetector:
    def __init__(
        self,
        model_file,
        score_threshold=0.5,
        iou_threshold=0.5,
        class_names=["date", "nsx"],
    ):
        self.model_file = model_file
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or ["object"]
        self.input_shape = (320, 320)
        self.rknn_model = RKNN_Model(model_file)

    def preprocess(self, image):
        input_img, ratio, pad = letterbox(image, self.input_shape)
        input_img = input_img.transpose(2, 0, 1)
        input_img = input_img[np.newaxis, :, :, :].astype(np.float32)
        input_img = np.ascontiguousarray(input_img)
        return input_img, ratio, pad

    def postprocess(self, outputs, image, ratio, pad):
        pred = non_max_suppression(
            outputs,
            conf_thres=self.score_threshold,
            iou_thres=self.iou_threshold,
            nc=len(self.class_names),
        )[0]
        if len(pred) > 0:  
            pred[:, :4] = scale_boxes(
                self.input_shape, pred[:, :4], image.shape, xywh=True
            )
            pred = np.concatenate([pred[:, :4], pred[:, -1:], pred[:, 4:6]], axis=-1)
            confs = pred[:, -2]
            classes = pred[:, -1]
            polygons = xywhr2xyxyxyxy(pred[:, :5])
            class_names = [self.class_names[int(cls)] for cls in classes]
            return polygons, confs, class_names
        else:
            return np.array([]), np.array([]), []

    def detect(self, image):
        preprocessed_image, ratio, pad = self.preprocess(image)
        outputs = self.rknn_model.inference(inputs=[preprocessed_image], data_format="NCHW")
        return self.postprocess(outputs[0], image, ratio, pad)

    def get_tl_tr_br_bl(self, points, padding_x, padding_y, image_width, image_height):
        points = np.asarray(points, dtype=np.float32)
        idx = np.argsort(points[:, 1])
        points = points[idx]
        tl, tr = (
            (points[0], points[1])
            if points[0][0] < points[1][0]
            else (points[1], points[0])
        )
        bl, br = (
            (points[2], points[3])
            if points[2][0] < points[3][0]
            else (points[3], points[2])
        )
        tl_x = np.clip(tl[0] - padding_x, 0, image_width)
        tl_y = np.clip(tl[1] - padding_y, 0, image_height)
        tr_x = np.clip(tr[0] + padding_x, 0, image_width)
        tr_y = np.clip(tr[1] - padding_y, 0, image_height)
        br_x = np.clip(br[0] + padding_x, 0, image_width)
        br_y = np.clip(br[1] + padding_y, 0, image_height)
        bl_x = np.clip(bl[0] - padding_x, 0, image_width)
        bl_y = np.clip(bl[1] + padding_y, 0, image_height)
        return (tl_x, tl_y), (tr_x, tr_y), (br_x, br_y), (bl_x, bl_y)

    def crop_image(self, image, points, padding_x=0, padding_y=0):
        shape = image.shape
        tl, tr, br, bl = self.get_tl_tr_br_bl(
            points, padding_x, padding_y, shape[1], shape[0]
        )
        points = np.array([tl, tr, br, bl], dtype=np.float32)
        crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3]),
            )
        )
        crop_height = int(
            max(
                np.linalg.norm(points[1] - points[2]),
                np.linalg.norm(points[0] - points[3]),
            )
        )
        pts_std = np.array(
            [
                [0, 0],
                [crop_width - 1, 0],
                [crop_width - 1, crop_height - 1],
                [0, crop_height - 1],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(points, pts_std)
        image = cv2.warpPerspective(
            image,
            matrix,
            (crop_width, crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC,
        )
        return image

    def draw(self, image, polygons, confs, classes):
        for i in range(polygons.shape[0]):
            p1, p2, p3, p4 = polygons[i]
            cv2.polylines(
                image,
                [
                    np.array(
                        [
                            [int(p1[0]), int(p1[1])],
                            [int(p2[0]), int(p2[1])],
                            [int(p3[0]), int(p3[1])],
                            [int(p4[0]), int(p4[1])],
                        ]
                    )
                ],
                True,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                image,
                f"{confs[i]:.3f}",
                (int(p1[0]), int(p1[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )
        return image


if __name__ == "__main__":
    text_detector = TextDetector(
        model_file="models/text_obb_quantized.rknn",
        score_threshold=0.5,
        iou_threshold=0.5,
        class_names=["date", "nsx"],
    )
    DATA_DIR = "models/obb_datasets"
    # DATA_OUTPUT = "image_examples_output_rknn"
    # if os.path.exists(DATA_OUTPUT):
    #     shutil.rmtree(DATA_OUTPUT)
    # os.makedirs(DATA_OUTPUT)
    for image_path in glob.glob(os.path.join(DATA_DIR, "*.jpg")):
        image = cv2.imread(image_path)
        t1 = time.time()
        polygons, confs, classes = text_detector.detect(image)
        t2 = time.time()
        # print(polygons, confs, classes)
        print("number of detections: ", len(polygons))
        print(f"Time taken: {t2 - t1} seconds")
        draw_image = text_detector.draw(image, polygons, confs, classes)
        # cv2.imwrite(os.path.join(DATA_OUTPUT, os.path.basename(image_path)), draw_image)
