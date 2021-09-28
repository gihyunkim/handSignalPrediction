# 데이터 보기
import pandas as pd
import numpy as np
from glob import glob
import shutil

# 이미지데이터 로딩
from PIL import Image
import cv2
from tqdm import tqdm

# 파일경로 설정
import os
import json

# Others
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter('ignore')
import matplotlib.pyplot as plt

data_path = 'E:\\dacon'

train_path = data_path + '\\train'
test_path = data_path + '\\test'

new_image_directory = data_path + '\\new_images'
new_train_image_directory = new_image_directory + '\\train'
new_test_image_directory = new_image_directory + '\\test'

action_information = pd.read_csv(data_path + '\\action_information.csv')
sample_submission = pd.read_csv(data_path + '\\sample_submission.csv')

def make_new_dir(path) :
    if os.path.isdir(path) == False:
        os.makedirs(path)

make_new_dir(new_image_directory)
make_new_dir(new_train_image_directory)
make_new_dir(new_test_image_directory)

# Train 데이터에 있는 폴더를 glob로 불러와
# sorted method를 통해 숫자 순으로 정렬합니다.
train_folders = sorted(glob(train_path + '\\*'), key = lambda x : int(x.split('\\')[-1].replace('file_','')))
test_folders  = sorted(glob(test_path + '\\*'), key = lambda x : int(x.split('\\')[-1].replace('file_','')))

classes = pd.get_dummies(action_information[['Label']], columns = ['Label']).to_numpy()

train_directories = np.array(sorted(glob(train_path + '\\*'), key=lambda x: int(x.split('\\')[-1].split('_')[-1])))

for train_directory in tqdm(train_directories, total=len(train_directories)):
    file_name = train_directory.split('\\')[-1]
    make_new_dir(new_train_image_directory + '\\' + file_name)

    image_paths = sorted(glob(train_directory + '\\*.jpg'), key=lambda x: int(x.split('\\')[-1].replace('.jpg', '')))
    json_path = glob(train_directory + '\\*.json')[0]

    js = json.load(open(json_path))
    target = js.get('action')
    target = classes[target]
    bounding_boxes = js.get('sequence').get('bounding_box')
    bounding_boxes = [(float(a), float(b), float(c), float(d)) for a, b, c, d in
                      bounding_boxes]  # 실수형태로 변환, (left, top, right, bottom)

    folder_name = image_paths[0].split("\\")[-2]
    js_path = "\\".join(image_paths[0].split("\\")[:-1]) + "\\" + folder_name + ".json"
    shutil.copy2(js_path, new_train_image_directory + image_paths[0].split('\\train')[1][:-4]+".json")

    for image_path, bounding_box in zip(image_paths, bounding_boxes):

        image = Image.open(image_path)
        image = image.crop(bounding_box)  # left top right bottom
        image = image.resize((224, 224))
        image.save(new_train_image_directory + image_path.split('\\train')[1])

''' for test data'''
test_directories = np.array(sorted(glob(test_path + '\\*'), key=lambda x: int(x.split('\\')[-1].split('_')[-1])))

for test_directory in tqdm(test_directories, total=len(test_directories)):
    file_name = test_directory.split('\\')[-1]
    make_new_dir(new_test_image_directory + '\\' + file_name)

    image_paths = sorted(glob(test_directory + '\\*.jpg'), key=lambda x: int(x.split('\\')[-1].replace('.jpg', '')))
    json_path = glob(test_directory + '\\*.json')[0]

    js = json.load(open(json_path))
    target = js.get('action')
    target = classes[target]
    bounding_boxes = js.get('sequence').get('bounding_box')
    bounding_boxes = [(float(a), float(b), float(c), float(d)) for a, b, c, d in
                      bounding_boxes]  # 실수형태로 변환, (left, top, right, bottom)

    for image_path, bounding_box in zip(image_paths, bounding_boxes):
        image = Image.open(image_path)
        image = image.crop(bounding_box)  # left top right bottom
        image = image.resize((224, 224))
        image.save(new_test_image_directory + image_path.split('\\test')[1])
