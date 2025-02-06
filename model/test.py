import tensorflow as tf
from tensorflow import keras
from keras import Model, Input, backend,layers
from keras.layers import Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.applications import xception

import os
import numpy as np
import pickle
import cv2
import matplotlib.pyplot as plt
from config import config as cfg #파일명, 클래스

#tf.data.AUTOTUNE is used, then the buffer size is dynamically tuned.
def configure_for_performance(ds,batch_size,autotune):
      ds = ds.cache()
      ds = ds.shuffle(buffer_size=256,reshuffle_each_iteration=True)
      ds = ds.batch(batch_size)
      ds = ds.prefetch(buffer_size=autotune)
      return ds

def showplot(hist):

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

if __name__ == "__main__" :
    #batch_size=16,num_classes = 20
    
    tf.keras.backend.clear_session()
    keras.backend.set_image_data_format("channels_last")

    AUTOTUNE = tf.data.AUTOTUNE
    with open(file = 'D:/ScanPlant_21913686/모델/nrmed_test_data.pickle', mode = 'rb') as f:
        x,y = pickle.load(f)
    test_dataset = tf.data.Dataset.from_tensor_slices((x,y))
    #test_dataset = configure_for_performance(test_dataset,1,AUTOTUNE)
    efficientnet = tf.keras.models.load_model("D:/ScanPlant_21913686/efficientnet.h5")
    templist =[]
    print("Evaluate on test data")
    img_array = np.fromfile(f'{cfg.path}/Test/6_고추_탄저병.jpeg',np.uint8)
    image = cv2.imdecode(img_array,cv2.IMREAD_COLOR)
    templist.append(image)
    img = np.array(templist)
    print(img.shape)
    #resized_image = cv2.resize(image, (244,244), interpolation = cv2.INTER_LANCZOS4) # INTER_AREA
    #results = efficientnet.evaluate(test_dataset, verbose =1 )
    #cv2.imshow('testimg',image)
    #cv2.waitKey(10000)
    #cv2.destroyAllWindows()

    #predict_step은 tensor, predict_on_batch는 numpy
    results = efficientnet.predict_on_batch(img)
    print("test value:", results.argmax())
    
    # Evaluate the model on the test data using `evaluate`



    