# This file is part of OpenCV Zoo project.
# It is subject to the license terms in the LICENSE file found in the same directory.
#
# Copyright (C) 2021, Shenzhen Institute of Artificial Intelligence and Robotics for Society, all rights reserved.
# Third party copyrights are property of their respective owners.

import argparse

import numpy as np
import cv2 as cv
import time

from yunet import YuNet

# Check OpenCV version
assert cv.__version__ >= "4.7.0", \
       "Please install latest opencv-python to try this demo: python3 -m pip install --upgrade opencv-python"

# Valid combinations of backends and targets
backend_target_pairs = [
    [cv.dnn.DNN_BACKEND_OPENCV, cv.dnn.DNN_TARGET_CPU],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA],
    [cv.dnn.DNN_BACKEND_CUDA,   cv.dnn.DNN_TARGET_CUDA_FP16],
    [cv.dnn.DNN_BACKEND_TIMVX,  cv.dnn.DNN_TARGET_NPU],
    [cv.dnn.DNN_BACKEND_CANN,   cv.dnn.DNN_TARGET_NPU]
]

parser = argparse.ArgumentParser(description='YuNet: A Fast and Accurate CNN-based Face Detector (https://github.com/ShiqiYu/libfacedetection).')
parser.add_argument('--input', '-i', type=str,
                    help='Usage: Set input to a certain image, omit if using camera.')
parser.add_argument('--video', '-d', type=str, help='Usage: Set input to a certain video.')
parser.add_argument('--model', '-m', type=str, default='face_detection_yunet_2022mar.onnx',
                    help="Usage: Set model type, defaults to 'face_detection_yunet_2022mar.onnx'.")
parser.add_argument('--backend_target', '-bt', type=int, default=0,
                    help='''Choose one of the backend-target pair to run this demo:
                        {:d}: (default) OpenCV implementation + CPU,
                        {:d}: CUDA + GPU (CUDA),
                        {:d}: CUDA + GPU (CUDA FP16),
                        {:d}: TIM-VX + NPU,
                        {:d}: CANN + NPU
                    '''.format(*[x for x in range(len(backend_target_pairs))]))
parser.add_argument('--conf_threshold', type=float, default=0.9,
                    help='Usage: Set the minimum needed confidence for the model to identify a face, defauts to 0.9. Smaller values may result in faster detection, but will limit accuracy. Filter out faces of confidence < conf_threshold.')
parser.add_argument('--nms_threshold', type=float, default=0.3,
                    help='Usage: Suppress bounding boxes of iou >= nms_threshold. Default = 0.3.')
parser.add_argument('--top_k', type=int, default=5000,
                    help='Usage: Keep top_k bounding boxes before NMS.')
parser.add_argument('--save', '-s', action='store_true',
                    help='Usage: Specify to save file with results (i.e. bounding box, confidence level). Invalid in case of camera input.')
parser.add_argument('--vis', '-v', action='store_true',
                    help='Usage: Specify to open a new window to show results. Invalid in case of camera input.')
args = parser.parse_args()


def detectAndDisplay(frame, min_confidence):
    img = cv.resize(frame, None, fx=0.8, fy=0.8)
    height, width, channels = img.shape
    blob = cv.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(output_layers)

    # -- 탐지한 객체의 클래스 예측
    class_ids = []
    confidences = []
    boxes = []

    for see in outs:
        for detection in see:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if class_id == 0 and confidence > min_confidence:
                # -- 탐지한 객체 박싱
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv.dnn.NMSBoxes(boxes, confidences, min_confidence, 0.4)
    font = cv.FONT_HERSHEY_DUPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = "{}: {:.2f}".format(classes[class_ids[i]], confidences[i] * 100)
            print(i, label)
            color = colors[i]  # -- 경계 상자 컬러 설정 / 단일 생상 사용시 (255,255,255)사용(B,G,R)
            cv.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv.putText(img, label, (x, y - 5), font, 1, color, 1)
    return img

def visualize(image, results, box_color=(0, 255, 0), text_color=(0, 0, 255), fps=None):
    output = image.copy()
    landmark_color = [
        (255,   0,   0), # right eye
        (  0,   0, 255), # left eye
        (  0, 255,   0), # nose tip
        (255,   0, 255), # right mouth corner
        (  0, 255, 255)  # left mouth corner
    ]

    if fps is not None:
        cv.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    for det in (results if results is not None else []):
        bbox = det[0:4].astype(np.int32)
        cv.rectangle(output, (bbox[0], bbox[1]), (bbox[0]+bbox[2], bbox[1]+bbox[3]), box_color, -1)

        conf = det[-1]
        cv.putText(output, '{:.4f}'.format(conf), (bbox[0], bbox[1]+12), cv.FONT_HERSHEY_DUPLEX, 0.5, text_color)

        landmarks = det[4:14].astype(np.int32).reshape((5,2))
        for idx, landmark in enumerate(landmarks):
            cv.circle(output, landmark, 2, landmark_color[idx], 2)

    return output


if __name__ == '__main__':
    backend_id = backend_target_pairs[args.backend_target][0]
    target_id = backend_target_pairs[args.backend_target][1]

    # -- yolo 포맷 및 클래스명 불러오기
    model_file = './yolov3.weights'  # -- 본인 개발 환경에 맞게 변경할 것
    config_file = './yolov3.cfg'  # -- 본인 개발 환경에 맞게 변경할 것
    net = cv.dnn.readNet(model_file, config_file)

    # -- GPU 사용
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # -- 클래스(names파일) 오픈 / 본인 개발 환경에 맞게 변경할 것
    classes = []
    with open("./coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))

    # Instantiate YuNet
    model = YuNet(modelPath=args.model,
                  inputSize=[320, 320],
                  confThreshold=args.conf_threshold,
                  nmsThreshold=args.nms_threshold,
                  topK=args.top_k,
                  backendId=backend_id,
                  targetId=target_id)

    if args.video is not None:
        cap = cv.VideoCapture(args.video)
        row = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        col = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 'v')
        model.setInputSize([row, col])
        out = cv.VideoWriter('result.mp4', fourcc, 30.0, (row, col))

        while(True):
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            results = model.infer(frame)  # results is a tuple
            frame = visualize(frame, results)
            out.write(frame)

        cap.release()
        out.release()
    else:
        deviceId = 0
        cap = cv.VideoCapture(deviceId)
        row = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        col = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        writer = cv.VideoWriter('video.mp4', fourcc, 30, (int(row), int(col)))
        model.setInputSize([row, col])

        tm = cv.TickMeter()
        while cv.waitKey(1) < 0:
            # hasFrame, frame = cap.read()
            hasFrame, frame = cap.read()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # Inference
            tm.start()
            results = model.infer(frame) # results is a tuple
            tm.stop()

            # Draw results on the input image
            frame = detectAndDisplay(frame, 0.5)
            # Default fps = tm.getFPS()
            frame = visualize(frame, results, fps=tm.getFPS())

            # Visualize results in a new Window
            cv.imshow('YuNet Demo', frame)

            tm.reset()
