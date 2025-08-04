import cv2
import onnxruntime as ort
import numpy as np
import glob
import os
import time


class Yolov11ONNX:
    def __init__(
        self,
        model_file,
        score_threshold=0.5,
        iou_threshold=0.5,
        class_names=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"],
    ):
        self.model_file = model_file
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self.class_names = class_names or []
        self.image_size = (320, 320)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        self.session = ort.InferenceSession(model_file, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        input_shape = self.session.get_inputs()[0].shape
        if len(input_shape) == 4:
            self.image_size = (input_shape[3], input_shape[2])

    def preprocess(self, input_img):
        img_height, img_width = input_img.shape[:2]
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        r = min(self.image_size[0] / img_width, self.image_size[1] / img_height)
        new_unpad = int(round(img_width * r)), int(round(img_height * r))
        dw, dh = self.image_size[0] - new_unpad[0], self.image_size[1] - new_unpad[1]
        dw, dh = dw / 2, dh / 2

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        input_img = cv2.resize(input_img, new_unpad, interpolation=cv2.INTER_LINEAR)
        input_img = cv2.copyMakeBorder(
            input_img,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        input_img = input_img / 255.0
        input_img = input_img.transpose(2, 0, 1)
        return input_img[np.newaxis, :, :, :].astype(np.float32)

    def postprocess(self, outputs, image_height, image_width):
        boxes_output = outputs[0][0]
        scores_output = outputs[1][0]
        classes_output = outputs[2][0]
        scores = scores_output.flatten()
        classes = classes_output.flatten()
        valid_indices = scores >= self.score_threshold
        valid_boxes = boxes_output[valid_indices]
        valid_scores = scores[valid_indices]
        valid_classes = classes[valid_indices]

        if len(valid_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        r = min(self.image_size[0] / image_width, self.image_size[1] / image_height)
        new_unpad = int(round(image_width * r)), int(round(image_height * r))
        dw, dh = self.image_size[0] - new_unpad[0], self.image_size[1] - new_unpad[1]
        dw, dh = dw / 2, dh / 2

        final_boxes = []
        final_scores = []
        final_classes = []

        for box, score, class_id in zip(valid_boxes, valid_scores, valid_classes):
            cx, cy, w, h = box[0], box[1], box[2], box[3]

            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            x1 = (x1 - dw) / r
            y1 = (y1 - dh) / r
            x2 = (x2 - dw) / r
            y2 = (y2 - dh) / r

            x1 = max(0, min(x1, image_width))
            y1 = max(0, min(y1, image_height))
            x2 = max(0, min(x2, image_width))
            y2 = max(0, min(y2, image_height))

            if x2 <= x1 or y2 <= y1:
                continue

            final_boxes.append([x1, y1, x2, y2])
            final_scores.append(float(score))
            final_classes.append(int(class_id))

        if len(final_boxes) == 0:
            return np.array([]), np.array([]), np.array([])

        final_boxes = np.array(final_boxes)
        final_scores = np.array(final_scores)
        final_classes = np.array(final_classes)

        boxes_for_nms = []
        for box in final_boxes:
            x1, y1, x2, y2 = box
            boxes_for_nms.append([x1, y1, x2 - x1, y2 - y1])

        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms, final_scores, self.score_threshold, self.iou_threshold
        )

        if len(indices) > 0:
            indices = indices.flatten()
            return final_boxes[indices], final_scores[indices], final_classes[indices]
        else:
            return np.array([]), np.array([]), np.array([])

    def detect(self, image):
        if image is None:
            raise ValueError("Input image is None")
        preprocessed_image = self.preprocess(image)
        outputs = self.session.run(None, {self.input_name: preprocessed_image})
        return self.postprocess(outputs, image.shape[0], image.shape[1])

    def predict(self, image):
        boxes, scores, classes = self.detect(image)
        if len(boxes) == 0:
            return "", []
        detections = list(zip(boxes, scores, classes))
        detections.sort(key=lambda x: x[0][0])
        sorted_boxes, sorted_scores, sorted_classes = zip(*detections)
        text = ""
        confidences = []

        for score, class_id in zip(sorted_scores, sorted_classes):
            class_name = (
                self.class_names[int(class_id)]
                if int(class_id) < len(self.class_names)
                else str(int(class_id))
            )
            text += class_name
            confidences.append(float(score))

        return text, confidences

    def draw(self, image, boxes, scores, classes):
        result_image = image.copy()
        for box, score, class_id in zip(boxes, scores, classes):
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            class_name = (
                self.class_names[int(class_id)]
                if int(class_id) < len(self.class_names)
                else str(int(class_id))
            )
            label = f"{class_name}: {score:.2f}"

            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            cv2.rectangle(
                result_image,
                (x1, y1 - text_size[1] - 4),
                (x1 + text_size[0], y1),
                (0, 255, 0),
                -1,
            )
            cv2.putText(
                result_image,
                label,
                (x1, y1 - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )

        return result_image


def main():
    model_path = "models/text_recognition_yolov11.onnx"
    classes_path = "models/text_recognition_yolov11_classes.txt"
    with open(classes_path, "r") as f:
        class_names = [line.strip() for line in f.readlines()]
    detector = Yolov11ONNX(
        model_path, score_threshold=0.5, iou_threshold=0.5, class_names=class_names
    )
    DATA_DIR = "models/text_crop"
    OUTPUT_DIR = "results"
    for image_path in glob.glob(os.path.join(DATA_DIR, "*.jpg")):
        image = cv2.imread(image_path)
        t1 = time.time()
        boxes, scores, classes = detector.detect(image)
        t2 = time.time()
        print(f"Detection time: {t2 - t1} seconds")
        if len(boxes) > 0:
            result_image = detector.draw(image, boxes, scores, classes)
            cv2.imwrite(
                os.path.join(OUTPUT_DIR, os.path.basename(image_path)), result_image
            )
            print(
                f"Result saved as {os.path.join(OUTPUT_DIR, os.path.basename(image_path))}"
            )
        else:
            print("No objects detected.")


if __name__ == "__main__":
    main()
