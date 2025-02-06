"""
# 파일명 : "modle_chili.py"
# 프로그램의 목적 및 기본 기능:
# 고추 사진을 비교하기 위한 데이터셋, 데이터 로더, 모델 정의
# 프로그램 작성자: 배재규 (2023년 9월 26일)
# 최종 Update : Version 1.0, 2023년 9월 26일 (배재규) 
========================================================
#프로그램 수정/보완 계획 및 이력
#훈련 700장 검증 300장으로 설정. 각 라벨별로 dict분류 후 비중 설정
#라벨별 밸런스가 안맞다. 질병쪽은 한 부위에 몰려있다. 따라서 질병에 맞는 부위만 남기고 다 버리는걸로.
========================================================
# 프로그램 수정/보완작업자 일자 수정/보완 내용
# 배재규 2023/09/26 v1.0 프로그램 작성
# 배재규 2023/09/30 v1.1 프로그램 작성
# 배재규 2023/11/29 v1.2 opencv는 한글 경로 못읽음. 
참고 : https://bskyvision.com/entry/python-cv2imread-%ED%95%9C%EA%B8%80-%ED%8C%8C%EC%9D%BC-%EA%B2%BD%EB%A1%9C-%EC%9D%B8%EC%8B%9D%EC%9D%84-%EB%AA%BB%ED%95%98%EB%8A%94-%EB%AC%B8%EC%A0%9C-%ED%95%B4%EA%B2%B0-%EB%B0%A9%EB%B2%95
# Test 데이터셋 구성시 참고 https://tfdream3662.tistory.com/75
========================================================
"""

# isort: off
from tensorflow.python.util.tf_export import keras_export
from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras import backend
from keras.applications import imagenet_utils
from keras.utils import Sequence
from sklearn.model_selection import train_test_split
from config import config as cfg

import make_dataFold as mdf
import json
import cv2
import os, re, glob
import numpy as np
import pickle
import random

#이미 그린 그래프가 있을때 clear
keras.backend.clear_session()

# Dataset 생성을 위한 makeDataset()선언
class makeDataset():
    # 전체 그룹의 폴더 경로 및 각 세부 카테고리 리스트를 입력받아 초기화
    def __init__(self):
        self.X_train = []
        self.Y_train = []
        self.X_valid = []
        self.Y_valid = []
        #reshape메소드 호출. 지우면 호출안함
        self.reshape()

    def makeDataLabel(self,label_path,index,isT):
        tlabels = []
        for label_name in os.listdir(label_path):
            m = "" #임시 str
            c = 0 #_단위로 나눈 수 저장
            namedict = {} #저장용 dict

            for i in label_name: #파일 이름을 char단위로 받은 후 _ 찾아서 나눔
                if i == "_":
                    namedict[c] = m #m구문 키c에 저장
                    c += 1 # 키 1증가
                    m = "" # m 초기화
                    continue
                m += i

            ##_79[1]_질병유무[2]_질병코드[3]_작물코드_작물부위_생육단계_피해정도_작업자ID_
            if int(namedict[2]) == 2 or int(namedict[2]) == 3 : continue
            if int(namedict[3]) not in cfg.categories: continue
            
            #데이터셋 부위에 맞게 분류
            """if index == 0:
                if int(namedict[5]) == 1:
                    tlabels.append(f'{label_path}/{label_name}')
                elif int(namedict[5]) == 3: 
                    tlabels2.append(f'{label_path}/{label_name}')
            elif index == 1:
                if int(namedict[5]) == 1:
                    tlabels.append(f'{label_path}/{label_name}')
            else:
                if int(namedict[5]) == 3:
                    tlabels.append(f'{label_path}/{label_name}')"""
            #테스트용 세트에 맞게 증강된 사진은 제외한다.
            #https://adnoctum.tistory.com/461 참고, dict.keys의 타입이 결정되지 않은 상태.
            #또한 ()를 써야 호출. not built - in 어쩌구 나오면 생각하기
            if index % 2 == 1:
                if 10 in namedict.keys() : continue

            #테스트셋 추출용으로 만듦
            if index == 0 or index == 1:
                if int(namedict[5]) == 1:
                    tlabels.append(f'{label_path}/{label_name}')
            else:
                if int(namedict[5]) == 3:
                    tlabels.append(f'{label_path}/{label_name}')

        #train700장, val300장
        #데이터셋 문제로 고추_정상 레이블만 열매:잎 = 5:5로 구성
        """if isT == 'T':
            if index == 0: #append쓰면 이중 리스트로 반환
                labels = random.sample(tlabels,350)+random.sample(tlabels2,350)
            else:
                labels = random.sample(tlabels,700)
        elif isT == 'V':
            if index == 0: 
                labels = random.sample(tlabels,150)+random.sample(tlabels2,150)
            else:
                labels = random.sample(tlabels,300)"""
        
        if isT == 'T':
            labels = random.sample(tlabels,3)
        
        print(labels)

        return labels

    def resize_image(self,label,img_path):
        
        with open(label) as json_file:
            json_decoded = json.load(json_file)
        
        #points는 dict list
        box = list(json_decoded["annotations"]["points"][0].values())
        name = json_decoded["description"]["image"]

        #opencv는 한글 경로 못읽음. np.fromfile로 변형
        img_array = np.fromfile(f'{img_path}/{name}',np.uint8)
        img = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
        #bgr->rgb는 생략. 코드 내부에선 문제없다.
        #img[ytl:ybr, xtl:xbr] #label annotation에 음수가 있어서 절댓값.
        cut_image = img[abs(box[1]):abs(box[3]), abs(box[0]):abs(box[2])]

        resized_image = cv2.resize(cut_image, cfg.train_size, interpolation = cv2.INTER_LANCZOS4) # INTER_AREA
        
        return resized_image 

    def reshape(self):
        pathlist_label_t = []
        pathlist_img_t = []
        #pathlist_label_v = []
        #pathlist_img_v = []
        #디렉토리 이름 별로 리스트 작성
        for b in cfg.crop_name2:
            for c in cfg.crop_name3:
                pathlist_label_t.append(f'{cfg.path}/{cfg.phase[0]}/{cfg.crop_name1[0]}{b}{c}')
                #pathlist_label_v.append(f'{cfg.path}/{cfg.phase[1]}/{cfg.crop_name1[0]}{b}{c}')
                pathlist_img_t.append(f'{cfg.path}/{cfg.phase[0]}/{cfg.crop_name1[1]}{b}{c}')
                #pathlist_img_v.append(f'{cfg.path}/{cfg.phase[1]}/{cfg.crop_name1[1]}{b}{c}')
        
        # 카테고리에 대해 인덱스를 설정(라벨링)
        # 라벨이 0부터 시작해야 sparse categorical이 적용된다.
        #패킹이 되어있어야 zip과 같이 사용
        for index,(label_path,img_path) in enumerate(zip(pathlist_label_t,pathlist_img_t)):
            labels = self.makeDataLabel(label_path,index,'T')
            print(f'Train/{index} : {len(labels)}')
            for i in labels:
                img = self.resize_image(i,img_path)
                self.X_train.append(img)
                #self.X_train.append(img/255.) rescale을 먼저 하지 않는다.
                self.Y_train.append(index)

        """for index,(label_path,img_path) in enumerate(zip(pathlist_label_v,pathlist_img_v)):
            labels = self.makeDataLabel(label_path,index,'V')
            print(f'Val/{index} : {len(labels)}')
            for i in labels:
                img = self.resize_image(i,img_path)
                self.X_valid.append(img)
                #self.X_valid.append(img/255.) rescale을 먼저 하지 않는다.
                self.Y_valid.append(index)"""
        
        X_train = np.array(self.X_train)
        Y_train = np.array(self.Y_train)
        #X_valid = np.array(self.X_valid)
        #Y_valid = np.array(self.Y_valid)
        
        #p_len = lambda x,y : print(f'{x} : {len(y)}')
        #p_len('X_train',X_train)
                
        #데이터셋 저장
        #xy = (X_train, Y_train, X_valid, Y_valid)
        xy = (X_train, Y_train)
        with open(file = f'{cfg.path}/test_data.pickle', mode = 'wb') as f:
            pickle.dump(xy, f,protocol=pickle.HIGHEST_PROTOCOL)

# Dataloader 선언
class CustomDataloader(Sequence):
    def __init__(self, x, y, batch_size, shuffle=True):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.x))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __len__(self):
        return int(np.ceil(len(self.x) / self.batch_size))

    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        x_batch = self.x[batch_indexes]
        y_batch = self.y[batch_indexes]
        return x_batch, y_batch

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

def data_normalization(dataset):
  meanRGB = [np.mean(image, axis=(0,1)) for image in dataset]
  stdRGB = [np.std(image, axis=(0,1)) for image in dataset]
  #numpy라서 BGR
  meanB = np.mean([m[0] for m in meanRGB])
  meanG = np.mean([m[1] for m in meanRGB])
  meanR = np.mean([m[2] for m in meanRGB])

  stdB = np.mean([s[0] for s in stdRGB])
  stdG = np.mean([s[1] for s in stdRGB])
  stdR = np.mean([s[2] for s in stdRGB])

  result_meanRGB = [meanB,meanG,meanR]
  result_stdRGB = [stdB,stdG,stdR]
  for i in dataset:
    for c in range(3):
      for a in range(224):
        for b in range(224):
          i[a][b][c] = (i[a][b][c]-result_meanRGB[c])/result_stdRGB[c]

def maketestset():
    """고추01 : 고추 탄저병         오이 12 : 오이 흰가루병
    무04 : 무 노균병             토마토 15 : 토마토 잎마름병
    배추05 : 배추 검은썩음병      콩 14 : 콩 점무늬병
    애호박08 : 애호박 흰가루병    파17 : 파 노균병
    양배추10 : 양배추 무름병      호박20 : 호박 흰가루병"""

    from PIL import Image

    with open(file = f'{cfg.path}/test_data.pickle', mode = 'rb') as f:
        X_trian, Y_train = pickle.load(f)

    indexname = ['고추_정상','고추_탄저병',
                 '무_정상','무_노균병',
                 '배추_정상','배추_검은썩음병',
                 '애호박_정상','애호박_흰가루병',
                 '양배추_정상','양배추_무름병',
                 '오이_정상','오이_흰가루병',
                 '토마토_정상','토마토_잎마름병',
                 '콩_정상','콩_점무늬병',
                 '파_정상','파_노균병',
                 '호박_정상','호박_흰가루병']
    #cv는 한글경로 못읽는다. imwrite도 마찬가지. bgr -> rgb 후 저장해준다
    for j,(h,i) in enumerate(zip(X_trian,Y_train)):
        img = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        im.save(f"{cfg.path}/Test/{j+1}_{indexname[i]}.jpeg")


if __name__ == "__main__" :
    with open(file = f'{cfg.path}/test_data.pickle', mode = 'rb') as f:
        X_train, Y_train = pickle.load(f)

    #recale은 필요없다. layer에 포함되어있다.
    data_normalization(X_train)
    xy = (X_train, Y_train)
    with open(file = f'{cfg.path}/nrmed_test_data.pickle', mode = 'wb') as f:
        pickle.dump(xy, f,protocol=pickle.HIGHEST_PROTOCOL)
