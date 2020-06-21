import os
import numpy as np
import imgaug
# import matplotlib.pyplot as plt
import sklearn.model_selection
import tensorflow as tf
import glob
import math
from tensorflow.compat.v1.keras.backend import set_session
import cv2
import detection
import datasets
import tools
import pipeline

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', '-b', default=1, type=int)
parser.add_argument('--output', '-o', default='model', type=str)

args = parser.parse_args()

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

def get_ocr_detector_dataset(type):
    training_gt_dir = f'../ocr_dataset/{type}/loc_gt'
    training_images_dir = f'../ocr_dataset/{type}/images'
    dataset = []
    for gt_filepath in glob.glob(os.path.join(training_gt_dir, '*.txt')):
        image_id = os.path.split(gt_filepath)[1].split('_')[0]
        image_path = os.path.join(training_images_dir, image_id + '.png')
        if not os.path.exists(image_path):
            print()
            print(image_id)
            print()
        lines = []
        with open(gt_filepath, 'r') as f:
            current_line = []
            for row in f.read().split('\n'):
                if row == '':
                    lines.append(current_line)
                    current_line = []
                else:
                    row = row.split(' ')[5:]
                    character = row[-1][1:-1]
                    if character == '':
                        continue
                    x1, y1, x2, y2 = map(int, row[:4])
                    current_line.append((np.array([[x1, y1], [x2, y1], [x2, y2], [x1,
                                                                                  y2]]), character))
        # Some lines only have illegible characters and if skip_illegible is True,
        # then these lines will be blank.
        lines = [line for line in lines if line]
        dataset.append((image_path, lines, 1))
    return dataset

train = get_ocr_detector_dataset('train')
validation = get_ocr_detector_dataset('test')
# train, validation = sklearn.model_selection.train_test_split(
#     dataset, train_size=0.8, random_state=42
# )
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(
    scale=(1.0, 1.2),
    rotate=(-5, 5)
    ),
    imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
])
generator_kwargs = {'width': 640, 'height': 640}
training_image_generator = datasets.get_detector_image_generator(
    labels=train,
    augmenter=None,
    **generator_kwargs
)
validation_image_generator = datasets.get_detector_image_generator(
    labels=validation,
    **generator_kwargs
)
# import tools
# import cv2
# image, lines, confidence = next(training_image_generator)
# canvas = tools.drawBoxes(image=image, boxes=lines, boxes_format='lines')
#
# cv2.imwrite('test.png', canvas)

detector = detection.Detector()

batch_size = args.batch_size
training_generator, validation_generator = [
    detector.get_batch_generator(
        image_generator=image_generator, batch_size=batch_size
    ) for image_generator in
    [training_image_generator, validation_image_generator]
]
detector.model.fit_generator(
    generator=training_generator,
    steps_per_epoch=math.ceil(len(train) / batch_size),
    epochs=10,
    workers=0,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
        tf.keras.callbacks.CSVLogger(os.path.join('log', f'{args.output}.csv')),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('model', f'{args.output}.h5'))
    ],
    validation_data=validation_generator,
    validation_steps=math.ceil(len(validation) / batch_size)
)

print('[INFO] training done')

def output(img_path, out_path):
    img_path = img_path[0]
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    detector = detection.Detector()
    detector.model.load_weights(f'model/{output}.h5')

    pipe = pipeline.Pipeline(detector=detector)
    predictions = pipe.recognize(images=[img])[0]
    drawn = tools.drawBoxes(
        image=img, boxes=predictions, boxes_format='predictions'
    )
    print(
        'Predicted:', [text for text, box in predictions]
    )
    cv2.imwrite(out_path, drawn)

for i, img in enumerate(validation_image_generator):
    output(img, f'output_{i}.png')