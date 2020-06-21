import xml.etree.ElementTree as ET
import os
import cv2
import tqdm
import sklearn.model_selection

PATH = 'custom_dataset'
files = os.listdir(PATH)
files = sorted(files)
files = [path for path in files if path.endswith('xml')]

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', '-y', default='train', type=str)

def make_dataset(files, type='train'):
    _id = 1
    for file in tqdm.tqdm(files):
        img_file = file[:-3] + 'png'
        if not os.path.exists(os.path.join(PATH, img_file)):
            img_file = file[:-3] + 'jpg'

        tree = ET.parse(os.path.join(PATH, file))
        root = tree.getroot()
        image = cv2.imread(os.path.join(PATH, img_file))
        cv2.imwrite(f'ocr_dataset/{type}/images/{str(_id).zfill(3)}.png', image)

        objects = tree.findall('object')

        with open(f'ocr_dataset/{type}/loc_gt/{str(_id).zfill(3)}_GT.txt', 'w') as f:

            for object in objects:
                name = object.find('name').text
                bndbox = object.find('bndbox')
                xmin = bndbox.find('xmin').text
                ymin = bndbox.find('ymin').text
                xmax = bndbox.find('xmax').text
                ymax = bndbox.find('ymax').text
                f.write(f'0 0 0 0 0 {xmin} {ymin} {xmax} {ymax} "{name}"\n')
                f.write('\n')
        _id += 1
    print('done')

train, test = sklearn.model_selection.train_test_split(
    files, train_size=0.8, random_state=42
)

os.makedirs('ocr_dataset/train/loc_gt', exist_ok=True)
os.makedirs('ocr_dataset/train/images', exist_ok=True)
os.makedirs('ocr_dataset/test/loc_gt', exist_ok=True)
os.makedirs('ocr_dataset/test/images', exist_ok=True)

make_dataset(train, 'train')
make_dataset(test, 'test')