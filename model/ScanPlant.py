"""
# 파일명 : "Compare_chili.py"
# 프로그램의 목적 및 기본 기능:
# 고추 사진의 질병 유무 확인을 위한 모델의 평가
# 프로그램 작성자: 배재규 (2023년 9월 26일)
# 최종 Update : Version 1.1, 2023년 9월 30일 (배재규) 
========================================================
#프로그램 수정/보완 이력
========================================================
# 프로그램 수정/보완작업자 일자 수정/보완 내용
# 배재규 2023/09/26 v1.0 프로그램 작성
# 배재규 2023/09/30 v1.1 프로그램 작성
========================================================
"""

from tensorflow import keras
from keras.models import Model
from keras.layers import Input, Dense
from keras.callbacks import EarlyStopping
import numpy as np
import pickle
import matplotlib.pyplot as plt
import tensorflow as tf
import keras.backend
from model_ScanPlant import CustomDataloader
from model_ScanPlant import makeDataset
from code_Resnet50 import *
from config import config as cfg

# main()으로 사용
def model_ScanPlant(X_train, Y_train, X_valid, Y_valid,
                  batch_size=32,num_classes = 20):

    #데이터로더
    train_loader = CustomDataloader(X_train,Y_train,batch_size, shuffle=True)
    valid_loader = CustomDataloader(X_valid,Y_valid,batch_size)
    #test_loader = CustomDataloader(X_test,Y_test,batch_size)
    
    K = keras.backend.backend()
    if K == 'tensorflow':
        keras.backend.set_image_data_format("channels_last")

    print("trian_loader shape", train_loader[0][0].shape)
    print("valid_loader shape", valid_loader[0][0].shape)

    input_tensor = Input(shape=(224, 224, 3), dtype='float32', name='input')
    x = conv1_layer(input_tensor)
    x = conv2_layer(x)
    x = conv3_layer(x)
    x = conv4_layer(x)
    x = conv5_layer(x)
    
    x = GlobalAveragePooling2D()(x)
    output_tensor = Dense(num_classes, activation='softmax')(x)
    
    resnet50 = Model(input_tensor, output_tensor)
    resnet50.summary()

    early_stopping = EarlyStopping(monitor="val_loss", patience = 10, restore_best_weights=True)

    #https://inhovation97.tistory.com/32 - 학습률 설정 기준
    opt = keras.optimizers.Adam(learning_rate=0.0002)
    resnet50.compile(loss='sparse_categorical_crossentropy',
                               optimizer = opt, metrics=["accuracy"])
    
    hist = resnet50.fit(train_loader, validation_data = valid_loader,
                           epochs = 100, verbose =1,workers = 4, callbacks=[early_stopping])
                                       
    resnet50.save('model_ScanPlant')

    #resnet50.evaluate(test_loader)

    with open(f'{cfg.save_path}/trainHistory','wb') as file_pi:
        pickle.dump(hist.history, file_pi)
    
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()

    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    loss_ax.legend(loc='upper left')

    acc_ax.plot(hist.history['accuracy'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_accuracy'], 'g', label='val acc')
    acc_ax.set_ylabel('accuracy')
    acc_ax.legend(loc='upper left')

    plt.show()

def load_model():
    model = tf.keras.models.load_model("model_ScanPlant")
    history = pickle.load(open(f'{cfg.save_path}/trainHistory','rb'))

    return model, history

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

  print(f'result_meanRGB: {result_meanRGB}')
  print(f'result_stdRGB: {result_stdRGB}')
"""  for i in dataset:
    for c in range(3):
      for a in range(224):
        for b in range(224):
          i[a][b][c] = (i[a][b][c]-result_meanRGB[c])/result_stdRGB[c]"""

if __name__ == "__main__" :
    #이미지 데이터셋을 생성해 pickle로 저장    
    #with open(file = f'{cfg.path}/img_data.pickle', mode = 'rb') as f:    
    #    X_train, Y_train, X_valid, Y_valid = pickle.load(f)

    #print(X_train.shape)
    #print(X_valid.shape)
    X_train = []
    q = np.ones((244,244,3))
    X_train.append(q) #np타입 list에 append가능
    X_train=np.array(X_train)
    print(X_train.shape)
   
    
    #data_normalization(X_valid)

    #model_ScanPlant(X_train, Y_train, X_valid, Y_valid)
