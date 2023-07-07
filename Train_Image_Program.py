import os
import cv2
import numpy as np
from keras.models import Sequential
from keras import layers
import subprocess
from PIL import Image
# 학습할 영상을 업데이트하여 저장하는 프로그램
# 폴더 삭제 함수
import shutil

def remove_folder(folder_path):
    shutil.rmtree(folder_path)

def video_update(vid_path,y,output_dir='전처리 npy 데이터'):

    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(vid_path))[0] + '_frames')
    os.makedirs(output_dir, exist_ok=True)

    #비디오에서 프레임 추출 -> 지금은 초당 5프레임 추출
    subprocess.call(['ffmpeg', '-i', vid_path, '-r', '5', '-f', 'image2', os.path.join(output_dir, 'frame-%03d.png')])

    # 추출한 이미지 전처리
    img_list = []
    for i in range(1, 51):
        img_path = os.path.join(output_dir, f'frame-{i:03d}.png')
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))
        img = img.astype('float32') / 255.0
        img_list.append(img)

    # 이미지 배열을 1,50,64,64,3 규격에 맞게 수정 -> 3차원에서 4차원으로
    img_array = np.expand_dims(np.array(img_list), axis=0)

    # 저장된 넘파이 데이터 경로 
    X = np.load('전처리 npy 데이터/X.new.npy')
    Y = np.load('전처리 npy 데이터/Y.new.npy')

    # 이미지 배열과 레이블을 저장된 데이터에 추가
    X = np.concatenate((X, img_array), axis=0)
    Y = np.concatenate((Y, np.array([[y]])), axis=0)

    # 수정된 데이터 저장 
    np.save('전처리 npy 데이터/X.new.npy', X)
    np.save('전처리 npy 데이터/Y.new.npy', Y)
    remove_folder(output_dir)
