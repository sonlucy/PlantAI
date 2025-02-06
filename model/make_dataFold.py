"""
# 파일명 : "make_dataFold.py"
# 프로그램의 목적 및 기본 기능:
# 라벨 오류 탐색, 라벨별 데이터 개수 파악
# 프로그램 작성자: 배재규
# 최종 Update : Version 1.2, 2023년 11월 22일 (배재규) 
========================================================
#프로그램 보완 필요 내용
#isWorngLabel labels리스트에 append하는 과정에서 검토를 같이하면 굳이 경로설정 없이 필요한거만 골라낼 수 있다.
#데이터셋 만들때 바로 호출해서 붙이는 것도 가능. 일단 유효하지 않은 걸 분리하는게 우선이라 따로 실행.
========================================================
# 프로그램 수정/보완작업자 일자 수정/보완 내용
# 배재규 2023/11/21 v1.1 프로그램 저장
# 배재규 2023/11/22 v1.2 match_label.py와 병합, isWorngLabel 구현 후 테스트 못함.
# 배재규 2023/11/29 v1.3
# label_name에 경로 붙여야함
# 함수 compare에서 label_name을 나누는걸로 변경. 폴더이름scanplant_에서 꼬임
# int 비교를 위해 namedict에 int() 설정
========================================================
"""

import numpy as np
import os
import json
from config import config as cfg
import shutil
import cv2
import matplotlib.pyplot as plt

"""
#numpy axis는 3이 최대. list 변수에 저장하지 않고서는 4차원 표현 불가.
#dataset 호출 없이 mean,std 구하려면 numpy.mean 말고 for문으로 더하고 나누면 된다.
#이미지 데이터는 정규화 필요없다.
#차집합은 잘못된 라벨 데이터들을 전부 test로 넘겨버린다
testlist = [x for x in range(len(label_list)) if x not in trainlist]
for c in testlist:
    shutil.copy(label_list[c], croplist_test)
"""

#name, ext = os.path.splitext(file) name에 이름 ext에 확장자 저장. 그냥 문자열 쓰는게 낫다.
#절대경로 중 이름만 추출. 이름 V,v로 시작.
def extract_name(path):
    nameindex1 = ""
    nameindex2 = ""
    t1 = ['V006','v006']
    t2 = ['.json','.Json','.JSON']
    
    for i in t1:
        if path.find(i) != -1: 
            nameindex1 = path.find(i) #python은 -1을 false로 인식하지 않는다
            break
    if nameindex1 == "" : return 'error'
    
    name1 = path[nameindex1:] #찾은 인덱스로부터 끝까지 슬라이스.

    #중간에 .이 있어서 첫 .까지로 못 끊는다
    for i in t2:
        if name1.find(i) != -1:
            nameindex2 = name1.find(i)
            break
    if nameindex2 == "" : return 'error'
    
    name2 = name1[:nameindex2] #.jpg 남음. 이미지 이름을 추출가능.
    return name2

#_79_질병유무_질병코드_작물코드_작물부위_생육단계_피해정도_작업자ID_
#수집날짜_촬영순서_동일객체촬영순서.확장자
def compare(path,label_name, json_decoded):    
    m = "" #임시 str
    c = 0 #_단위로 나눈 수 저장
    namedict = {} #저장용 dict
    for i in label_name: #외부에서 받은 json파일 이름을 char단위로 받은 후 _ 찾아서 나눔
        if i == "_":
            namedict[c] = m #m구문 키c에 저장
            c += 1 # 키 1증가
            m = "" # m 초기화
            continue
        m += i

    name = extract_name(path)

    #[1]:79,[2]:질병유무,[3]:질병코드 ~ [7]
    if name != json_decoded[cfg.label_key1[0]].get(cfg.label_key2_1[0]) : return False
    if int(namedict[2]) != json_decoded[cfg.label_key1[0]].get(cfg.label_key2_1[6]) : return False
    # int type비교라서 :02d 필요없음
    for e,i in enumerate(cfg.label_key2_2):
        if int(namedict[e+3]) != json_decoded["annotations"].get(i):return False
        if e == 4 : return True # +0,1,2,3,4 [7]까지 확인 후 True반환

#잘못된 거 이동, extract_name은 확장자를 잘라내므로 사용하면 안된다
def moveLabel(path_wronglabel,wronglabel,movepath):
    if not os.path.exists(movepath):
        os.makedirs(movepath)
    shutil.move(path_wronglabel,f'{movepath}/{wronglabel}')
    
#_79부터_risk까지 라벨과 라벨파일제목 비교. 논리값 반환. none인 경우를 잡기 위함. 이후 이미지 파일과 제목끼리 비교
def isWorngLabel(path,movepath,imgpath):
    labels = []
    wornglabels =[]
    for label_name in os.listdir(path):
        #if label_name.endswith(('.json,','.JSON','.Json')) endswith는 .jpg.json으로 첫 .부터 확장자를 추출한다
        with open(f'{path}/{label_name}') as json_file: 
            json_decoded = json.load(json_file)
        if compare(f'{path}/{label_name}',label_name, json_decoded):
            labels.append(f'{path}/{label_name}')
        else:
            moveLabel(f'{path}/{label_name}',label_name,movepath)
            wornglabels.append(f'{movepath}/{label_name}')

    for wornglabel in wornglabels:
        n = extract_name(wornglabel)
        for imgname in os.listdir(imgpath):
            if n == imgname :
                shutil.move(f'{imgpath}/{imgname}',f'{movepath}/{imgname}')

    return labels #리스트 반환 가능. https://velog.io/@tenacious_mzzz/python-%EC%A0%84%EC%97%AD%EB%B3%80%EC%88%98-%EB%B0%B0%EC%97%B4참고

#9.증강은 질병밖에 없어서 질병으로 이동
def moveAugmentedLabel():
    try:
        for a in cfg.phase:
            path = f'{cfg.path}/{a}'
            for b in cfg.crop_name1:
                for c in cfg.crop_name2:
                    datapath = f'{path}/{b}{c}{cfg.crop_name3[2]}'
                    movepath = f'{path}/{b}{c}{cfg.crop_name3[1]}'
                    for i in os.listdir(datapath):
                        shutil.move(f'{datapath}/{i}',f'{movepath}/{i}')
            #if not os.path.exists(label_path):
            #raise Exception(f'labelError :\n img:{img_path}\n label:{label_path}') #반복문에서는 {i}써서 몇번째까지 했는지도 출력.

    except Exception as ex:
        print(f'오류:\ndatapath:{datapath}\nmovepath:{movepath}\nerrmsg:{ex}')

def makedict(dict_all_labels):
    for a in cfg.phase:
        for k in cfg.crop_name2:
            for m in cfg.crop_name3:
                for l in range(6):
                    dict_all_labels[f'{a}{k}{m}{l}']=[]

    for a1 in cfg.phase:
            path = f'{cfg.path}/{a1}'
            for b1 in cfg.crop_name2:
                for d1 in cfg.crop_name3:    
                    datapath = f'{path}/{cfg.crop_name1[0]}{b1}{d1}'
                    for label_name in os.listdir(datapath):
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
                        dict_all_labels[f'{a1}{b1}{d1}{int(namedict[5])}'].append(f'{datapath}/{label_name}')

def removeLabel(dict_all_labels,flag):
    
    for a in cfg.phase:
        for k in cfg.crop_name2:
            for l in range(6):
                dict_all_labels[f'{a}{k}{cfg.crop_name3[1]}{l}']=[]
                flag[f'{a}{k}{cfg.crop_name3[1]}{l}']=False

    for a1 in cfg.phase:
            path = f'{cfg.path}/{a1}'
            for b1 in cfg.crop_name2:               
                datapath = f'{path}/{cfg.crop_name1[0]}{b1}{cfg.crop_name3[1]}'
                for label_name in os.listdir(datapath):
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
                        
                    """##_79[1]_질병유무[2]_질병코드[3]_작물코드_작물부위_생육단계_피해정도_작업자ID_
                    if int(namedict[2]) == 2 or int(namedict[2]) == 3 :
                        if os.path.exists(f'{datapath}/{label_name}'):
                            os.remove(f'{datapath}/{label_name}')
                        continue
                    if int(namedict[3]) not in cfg.categories: 
                        if os.path.exists(f'{datapath}/{label_name}'):
                            os.remove(f'{datapath}/{label_name}')
                        continue
                    if int(namedict[5]) == 6 or int(namedict[5]) == 7 : 
                        if os.path.exists(f'{datapath}/{label_name}'):
                            os.remove(f'{datapath}/{label_name}')
                        continue"""
                
                    dict_all_labels[f'{a1}{b1}{cfg.crop_name3[1]}{int(namedict[5])}'].append(f'{datapath}/{label_name}')
                    
    for a in cfg.phase:
        for k in cfg.crop_name2:
            temp = []
            for l in range(6):
                temp.append(len(dict_all_labels[f'{a}{k}{cfg.crop_name3[1]}{l}']))
            
            temp.sort(reverse=True)
            for l in range(6):
                if temp[0] == len(dict_all_labels[f'{a}{k}{cfg.crop_name3[1]}{l}']):
                    flag[f'{a}{k}{cfg.crop_name3[1]}{l}']=True
                    continue

    #질병라벨 중 가장 많은 부위만 추출
    for a in cfg.phase:
        for k in cfg.crop_name2:
            for l in range(6):
                if flag[f'{a}{k}{cfg.crop_name3[1]}{l}']==True:
                    continue
                for i in dict_all_labels[f'{a}{k}{cfg.crop_name3[1]}{l}']:
                    if os.path.exists(i):
                        os.remove(i)

def saveNamelist(arr,fullname_txt):
    with open(fullname_txt, 'w', encoding='UTF-8') as f:
        for name in arr:
            f.write(name+'\n')

def loadNamelist(arr,fullname_txt):
    with open(fullname_txt, 'r', encoding='UTF-8') as f:
        while True:
            line = f.readline()
            if not line : break
            line = line.replace('\n','') #savenamelist는 끝의\n도 저장한다
            arr.append(line)
        f.close()

#set datapath
#name, ext = os.path.splitext(file) 파일을 확장자와 분리
if __name__ == '__main__':
    #각 경로별로 dict로 구성해서 각 식물별 질병 총합, 정상 총합 구하기(마지막에)
    all_labels = dict()
    all_labels[1]=[]
    loadNamelist(all_labels[1],f'{cfg.path}/{cfg.phase[0]}{cfg.crop_name2[0]}{cfg.crop_name3[1]}1.txt')
    for i in range(9):
        all_labels[i+2]=[]
        loadNamelist(all_labels[i+2],f'{cfg.path}/{cfg.phase[0]}{cfg.crop_name2[i+1]}{cfg.crop_name3[1]}3.txt')

    all_labels[11]=[]
    loadNamelist(all_labels[11],f'{cfg.path}/{cfg.phase[1]}{cfg.crop_name2[0]}{cfg.crop_name3[1]}1.txt')
    for i in range(9):
        all_labels[i+12]=[]
        loadNamelist(all_labels[i+12],f'{cfg.path}/{cfg.phase[1]}{cfg.crop_name2[i+1]}{cfg.crop_name3[1]}3.txt')
    
    for k in all_labels.keys():
        print(f'{k} : {len(all_labels[k])}')

    #if not os.path.exists(label_path):
    #raise Exception(f'labelError :\n img:{img_path}\n label:{label_path}') #반복문에서는 {i}써서 몇번째까지 했는지도 출력.
