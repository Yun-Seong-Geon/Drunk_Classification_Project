import os
import numpy as np

# 매핑 정보를 정의한 딕셔너리
label_mapping = {
    15: 1,  # 얼굴을 1로 매핑
    18: 1,  # 얼굴을 1로 매핑
    16: 2,  # 얼굴색 정상을 2로 매핑
    17: 3,  # 눈풀림 정상을 3으로 매핑
    4 : 1,  # 얼굴을 1로 매핑
    6 : 4,  # 얼굴색 취함을 4로 매핑
    7 : 5,  # 눈풀림을 5로 매핑
}

# 텍스트 파일이 있는 폴더 경로
label_folder = "non_alcohol/01-00"

# 텍스트 파일 경로 리스트 초기화
text_file_paths = []

# 폴더 내의 텍스트 파일 순회
for root, dirs, files in os.walk(label_folder):
    for file in files:
        if file.endswith(".txt"):  # 텍스트 파일인 경우만 처리
            text_file_paths.append(os.path.join(root, file))

# new_X 초기화
new_X = np.load("전처리 npy 데이터/X.npy")

# 텍스트 파일 경로 리스트 순회
# 텍스트 파일 경로 리스트 순회
for file_path in text_file_paths:
    with open(file_path, encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
                values = line.strip().split()
                label = int(values[0])
                mapped_label = label_mapping[label]  # 매핑된 라벨
                x_label1 = float(values[1])
                x_label2 = float(values[2])
                x_label3 = float(values[3])
                x_label4 = float(values[4])
                label_info = [[mapped_label, x_label1, x_label2, x_label3, x_label4]]
                label_info = np.expand_dims(label_info, axis=(1, 2, 3))  # shape: (1, 1, 1, 1, 5)
                label_info = np.tile(label_info, (13, 50, 64, 64, 1))  # shape: (13, 50, 64, 64, 5)
                new_X = np.concatenate([new_X, label_info], axis=4)  # shape: (13, 50, 64, 64, 8)
                print('하나 종료')



# new_X 저장
np.save("new_x.npy", new_X)
