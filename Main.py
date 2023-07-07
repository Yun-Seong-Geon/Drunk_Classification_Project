import os
import cv2
import numpy as np
from keras.models import Sequential
from keras import layers
from keras.models import load_model
from PIL import Image
import subprocess
import shutil


# 폴더 삭제 함수
def remove_folder(folder_path):
    shutil.rmtree(folder_path)

# 영상 예측 함수
def video_to_predictions(num_frames=50, output_dir='test'):

    video_path = input('주소를 입력하시오: ')
    
    # 새로운 폴더 생성
    output_dir = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + '_frames')
    os.makedirs(output_dir, exist_ok=True)

    # 비디오 파일에서 프레임 추출
    cmd = f"ffmpeg -i {video_path} -vf scale=64:64 -q:v 1 -r {num_frames}/10 {output_dir}/frame_%03d.jpg"
    subprocess.call(cmd, shell=True)

    # 추출된 프레임들을 PIL Image로 변환하여 모델에 예측
    model = load_model('학습모델/saved_model.h5')
    test_list = []
    for dir_path, _, file_names in os.walk(output_dir):
        img_list = []
        for i, file_name in enumerate(sorted(file_names)):
            if i >= num_frames:
                break
            img_path = os.path.join(dir_path, file_name)
            img = cv2.imread(img_path)
            img = cv2.resize(img, (64, 64))
            img = img.astype('float32') / 255.0
            img_list.append(img)
        test_list.append(np.array(img_list))
        
    # 이미지데이터셋 넘파이로 저장하여 예측
    test_list = np.array(test_list)
    pred = model.predict(test_list)
    result = np.mean(pred[:, :, :], axis=1)
    result = round(float(result), 3)

    # 학습이 완료됐다면 폴더 삭제
    remove_folder(output_dir)
    # 결과값 리턴
    return result*100

result = video_to_predictions()
print(f'{result}%')






