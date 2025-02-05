import asyncio
import websockets
import base64
import cv2
import numpy as np
import tensorflow as tf

from tensorflow import keras
from keras import models
from keras.models import load_model

# 미리 훈련된 모델 로드
model = load_model('./modelname.h5')  # 모델 파일 경로 주의

# 이미지를 전처리하고 모델에 입력하여 예측하는 함수
def predict_image(img):
    preprocessed_img = cv2.resize(img, (224, 224))  # 크기 조정
    prediction = model.predict(preprocessed_img)
    preprocessed_img = np.expand_dims(preprocessed_img, axis=0)
    
    
    ########### 여기에 코드 추가 필요




    
    return predicted_result #예측 결과 반환



# 웹 소켓 클라이언트가 접속이 되면 호출
async def accept(websocket, path):
    while True:
        cmd = await websocket.recv()
        if cmd == 'START':
            await websocket.send("FILENAME")
        elif cmd == 'FILENAME':
            filename = await websocket.recv()
            await websocket.send("FILESIZE")
        elif cmd == 'FILESIZE':
            filesize = await websocket.recv()
            await websocket.send("DATA")
        elif cmd == 'DATA': #### 데이터를 추가해야함
            data = await websocket.recv()
            # string을 byte로 변환(base64는 아스키코드로 구성) base64를 binary로 디코딩
            byte = base64.b64decode(data.encode("ASCII"))
            #byte데이터를 uint8타입 binary 읽어  3d-array로 만들어준다
            img = cv2.imdecode(np.frombuffer(byte, np.uint8), cv2.IMREAD_COLOR) ##
            
            predicted_result = predict_image(img)
            await websocket.send(predicted_result) #예측 결과 클라이언트(웹)으로 전송
            
            # 파일 전송 완료 -> 연결 종료하도록
            await websocket.close()
            break

# 웹 소켓 서버 생성. 호스트는 localhost에 port는 9998로 생성
start_server = websockets.serve(accept, "localhost", 9998)

# 비동기로 서버를 대기
asyncio.get_event_loop().run_until_complete(start_server)
asyncio.get_event_loop().run_forever()



