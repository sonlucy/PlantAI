import json
import os

class config():
  #00:작물없음 #01:고추 #02:토마토 #03:오이 #04:양배추 #05:배추
  #06:애호박 #07:콩 #08:무 #09:파 #10:호박
  label_key1 = ["description","annotations"]
  label_key2_1 = ["image","date","worker","height","width","task","type","region"]
  label_key2_2 = ["disease","crop","area","grow","risk","points"]
  #"points"
  label_key3 = ["xtl","ytl","xbr","ybr"]
  label_area ={'00':0,'01': 1,'02': 2,'03':3,'04':4,'05':5,'06':6, '07':7}
  label_grow ={'11':0,'12':1,'13':2}

  # general args
  # size 512 하면 768kb. 소켓 전송시 64kb로 13번.
  # 224는 147kb 5배이상 차이남. 일단 둘다 샘플로 만들어보고 결정하는걸로.
  
  train_size = (224, 224)
  crop_name1 = ['[라벨]','[원천]']
  crop_name2 = ['고추_','토마토_','오이_','양배추_','배추_','애호박_',
                '콩_','무_','파_','호박_']
  crop_name3 = ['0.정상','1.질병']
  area=[1,3,3,3,3,3,3,3,3,3]
  train_model = 'densenet121'
  #num_classes = str(len(label1)*len(label2)) <- 모델에 들어갈 클래스 개수
  path = 'D:/ScanPlant_21913686/노지 작물 질병 진단 이미지(irx파일지우면안됨)' 
  save_path = 'D:/ScanPlant_21913686/'
  phase = ['Training','Validation'] #,'Test']
  
  """
  고추01 : 고추 탄저병         오이 12 : 오이 흰가루병
  무04 : 무 노균병             토마토 15 : 토마토 잎마름병
  배추05 : 배추 검은썩음병      콩 14 : 콩 점무늬병
  애호박08 : 애호박 흰가루병    파17 : 파 노균병
  양배추10 : 양배추 무름병      호박20 : 호박 흰가루병
  """
  categories = [0,1,4,5,8,10,12,14,15,17,20]

  # testing args
  test_model_path = 'weights/sample.pt' #경로 붙이기

