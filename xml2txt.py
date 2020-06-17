import xml.etree.ElementTree as ET
import os
import cv2
import tqdm

PATH = 'custom_dataset'
files = os.listdir(PATH)
files = sorted(files)
files = [path for path in files if path.endswith('xml')]
_id = 1
os.makedirs('ocr_train', exist_ok=True)
os.makedirs('ocr_train/loc_gt', exist_ok=True)
os.makedirs('ocr_train/training_images', exist_ok=True)
for file in tqdm.tqdm(files):
    img_file = file[:-3]+'png'
    if not os.path.exists(os.path.join(PATH, img_file)):
        img_file = file[:-3] + 'jpg'

    tree = ET.parse(os.path.join(PATH, file))
    root = tree.getroot()
    image = cv2.imread(os.path.join(PATH, img_file))
    cv2.imwrite(f'ocr_train/training_images/{str(_id).zfill(3)}.png', image)

    objects = tree.findall('object')

    with open(f'ocr_train/loc_gt/{str(_id).zfill(3)}_GT.txt', 'w') as f:

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