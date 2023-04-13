import cv2
import requests
import threading

import time

# 서버 URL
OriginURL = "http://127.0.0.1:5000/camera/original"
ProURL = "http://127.0.0.1:5000/camera/process"

#카메라ID 설정
cameraID = 'esqure_01'

# 웹캠 설정
cap = cv2.VideoCapture(0)

# 프레임 처리 및 서버 전송 함수
def send_frame(url, frame, timeData):
    _, img_encoded = cv2.imencode(".jpg", frame)
    requests.post(url, data={'time': timeData, 'cameraID': cameraID}, files={'frame': ('image.jpg', img_encoded, 'image/jpeg')})

while True:
    # 원본영상 프레임 읽기
    OriginRet, OriginFrame = cap.read()
    timeData = time.time()

    #이코드는 영상처리된 프레임을 전송하기 위한 더미 데이터입니다.
    ProRet, ProFrame = cap.read()
    #영상처리 알고리즘 적용할 때, 더미 데이터를 지우고 영상처리된 프레임으로 대체해주세요


    # 프레임 처리 및 서버 전송 병렬 처리
    t1 = threading.Thread(target=send_frame, args=(OriginURL, OriginFrame, timeData))
    t2 = threading.Thread(target=send_frame, args=(ProURL, ProFrame, timeData))
    t1.start()
    t2.start()

# 웹캠 종료
cap.release()
