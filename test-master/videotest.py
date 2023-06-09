# 230414 Testing Code
import onnx
import numpy as np
import time
from cv2 import *  
import cv2
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import threading
import subprocess
from ultralytics import YOLO

def mosaic_area(img, box, ratio=0.1):
    # 모자이크를 적용할 영역을 구합니다.

    for i in range(len(box)):
        if box[i] < 0: 
            box[i] = 0
    start_x, start_y, end_x, end_y = box

    print(f'start_x, start_y, end_x, end_y = {start_x, start_y, end_x, end_y}')
    
    region = img[start_y:start_y + end_y, start_x:start_x + end_x]

        # 모자이크를 적용할 영역의 크기를 구합니다.
    height, width = region.shape[:2]
    w = int(width * ratio)
    h = int(height * ratio)
    if w <= 0:
        w = 1
    if h <= 0:
        h = 1
    print(f'height, width = {height, width}')
    print(f'h, w = {h, w}')

    # 영역을 축소합니다.
    small = cv2.resize(region, (w, h), interpolation=cv2.INTER_AREA)
    
    # 축소한 영역을 다시 확대합니다.
    mosaic = cv2.resize(small, (width, height), interpolation=cv2.INTER_NEAREST)
    
    # 모자이크를 적용한 이미지를 생성합니다.
    img_mosaic = img.copy()
    img_mosaic[start_y:start_y + end_y, start_x:start_x + end_x] = mosaic
    # cv2.imshow("img_mosaic", img_mosaic)
    return img_mosaic

'''
def detectbox(frame):
    img = frame
    # print('detectbox')
    outs = yolo_model(img, classes = 0, conf=0.3)
    # annotated_frame = outs[0].plot()

    # cv2.imwrite(f'camera/original/{timeData}.jpg', annotated_frame)
    
    # -- 탐지한 객체의 클래스 예측
    # class_ids = []
    boxes_buf = []

    # outs == 객체 개수
    # detection == box x1,y1, x2, y2
    for see in outs:
        for detection in see: # see == outs[0,1,2,3 ..., n]
            boxes = detection.boxes # 
            box = boxes[0] # 가장 높은 conf 값을 갖은 것
            x, y, w, h = box.xywh[0].tolist()
            boxes_buf.append([x, y, w, h])
    return boxes_buf
'''

# results = bbox ==? yunet
# boxes = yolo
def visualize(frame, results, boxes, box_color=(255, 0, 0), text_color=(0, 0, 255), fps=None):
    output = frame.copy()
    if fps is not None:
        cv2.putText(output, 'FPS: {:.2f}'.format(fps), (0, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color)

    if boxes == 0:
        for det in (results if results is not None else []):
            bbox = list(map(int, det[:4]))

            output = mosaic_area(output, bbox)

            # cv2.rectangle(output, bbox , box_color, -1)
            
    '''
    else:
        # print(f'len(boxes) = {len(boxes)}')
        for i in range(len(boxes)):
            if i < len(boxes):
                # print(f'boxes[{i}] = {boxes[i]}')
                x, y, w, h = boxes[i]
                x, y, w, h = int(x), int(y), int(w), int(h)
                x1 = x - w//2 
                x2 = x + w//2
                y1 = y - h//2
                y2 = y + h//2
                # reduce the y2 coordinate by 30% of the original height
                y2_new = int(y1 + (y2-y1)*0.3)
                # reduce the x2 coordinate by 50% of the original width
                x2_new = int(x1 + (x2-x1)/4 * 3)
                # increase the x1 coordinate by 50% of the original width
                x1_new = int(x1 + (x2-x1)/4)
                detected = False
                # print(x1_new, y1, x2_new, y2_new)
                for det in (results if results is not None else []):
                    bbox = list(map(int, det[:4]))
                    # print(f'99: {bbox}')
                    output = mosaic_area(output, bbox)
                    # print(f'bbox = {bbox}')
                    # cv2.rectangle(output, bbox , box_color, -1)
                    if x1_new < bbox[0] and y1 < bbox[1] and x2_new > bbox[2] and y2_new > bbox[3]:
                        detected = True

                if detected:
                    continue
                else:
                    # print(f'x = {x}, {type(x)}, y = {y}, {type(y)}, w = {w}, {type(w)}, h = {h}, {type(h)}')
                    region = output[y1:y2_new, x1_new:x2_new]
                    # region = output[y1:y2, x1:x2]

                    if region.shape[0] > 0 and region.shape[1] > 0:
                        col = region.shape[0]
                        row = region.shape[1]
                        region = cv2.resize(region, (0, 0), fx=0.10, fy=0.15, interpolation=cv2.INTER_AREA)
                        region = cv2.resize(region, (row, col), interpolation=cv2.INTER_AREA)
                        output[y1:y2_new, x1_new:x2_new] = region
                        # output[y1:y2, x1:x2] = region
                    else:
                        continue
    '''
    return output

def send_frame(url, frame, timeData):
    _, img_encoded = cv2.imencode(".jpg", frame)
    requests.post(url, data={'time': timeData, 'cameraID': cameraID}, files={'frame': ('image.jpg', img_encoded, 'image/jpeg')})

if __name__ == '__main__':

    OriginURL = "http://bangwol08.iptime.org:20002/camera/original"
    ProURL = "http://bangwol08.iptime.org:20002/camera/process"

    cameraID = 'esqure_01'
    face_detector = cv2.FaceDetectorYN.create("new1_jodory.onnx", "", (320, 320))
    yolo_model = YOLO('yolov8n.onnx')
    person = 0
    frame_num = 0
    videoPath = 'lab.mp4'
    # print('loading')
    
    cap = cv2.VideoCapture(videoPath)
    print(videoPath)
    row = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    col = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    face_detector.setInputSize([row, col])
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter('final_resultest.mp4', fourcc, fps, (row, col))

    count = 0
    tick = 0
    constancy = 0
    boxes = None
    num = 0
    while True:
        tm = cv2.TickMeter()
        while cv2.waitKey(1) < 0:
            tm.start()
            frame_num += 1
            hasFrame, frame = cap.read()
            OriginFrame = frame.copy()
            # timeData = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            timeData = time.time()
            if not hasFrame:
                print('No frames grabbed!')
                break

            # If the frame is not 3 channels, convert it to 3 channels
            channels = 1 if len(frame.shape) == 2 else frame.shape[2]
            if channels == 1:
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            if channels == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            height, width, _ = frame.shape
            face_detector.setInputSize((width, height))

            i = 0

            # Inference

            _, results = face_detector.detect(frame)  # results is a tuple

            for det in (results if results is not None else []):
                i = i + 1

            print(f'frame_num: {frame_num}, i = {i}, person = {person}, count = {count}, tick = {tick}')
            if person != i:
                print(f'different!! frame_num = {frame_num}, i = {i}, person = {person}')
                if tick != 1:
                    count = 6
                if person > i:
                    tick = 1
                    constancy = person
                    face_detector.setScoreThreshold(0.5)
                    _, results = face_detector.detect(frame)
            person = i

            if count == 6:
                print(f'frame_num: {frame_num}')
                # boxes = detectbox(frame)
                count = count - 1
            elif count > 0:
                count = count - 1
            else:
                constancy = 0

            if i >= constancy and tick == 1:
                tick = 0
                face_detector.setScoreThreshold(0.8)

            # Default fps = tm.getFPS()
            if tick == 0:
                if boxes == None:
                    frame = visualize(frame, results, 0, fps=tm.getFPS())
                else:
                    frame = visualize(frame, results, boxes, fps=tm.getFPS())
            else:
                frame = visualize(frame, results, 0, fps=tm.getFPS())

                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)  # 시계방향으로 90도 회전
                face_detector.setInputSize((height, width))
                _, results = face_detector.detect(frame)
                frame = visualize(frame, results, 0, fps=tm.getFPS())

                frame = cv2.rotate(frame, cv2.ROTATE_180)
                _, results = face_detector.detect(frame)
                frame = visualize(frame, results, 0, fps=tm.getFPS())

                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            # Visualize results in a new Window
            tm.stop()
            out.write(frame)

            t1 = threading.Thread(target=send_frame, args=(OriginURL, OriginFrame, timeData))
            t2 = threading.Thread(target=send_frame, args=(ProURL, frame, timeData))
            t1.start()
            t2.start()

            # cv2.imshow('YuNet Demo', frame)
            # cv2.imwrite(f'camera/process/{timeData}.jpg', frame)
            # cv2.imwrite(f'camera/original/{timeData}.jpg', OriginFrame)

        tm.reset()
        out.release()
        cap.release()
        break