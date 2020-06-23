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
import recognition
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

def get_ocr_recog_dataset(type):
    training_gt_dir = f'../ocr_dataset/{type}/loc_gt'
    training_images_dir = f'../ocr_dataset/{type}/images'
    dataset = []
    for gt_filepath in glob.glob(os.path.join(training_gt_dir, '*.txt')):
        image_id = os.path.split(gt_filepath)[1].split('_')[0]
        image_path = os.path.join(training_images_dir, image_id + '.png')
        if not os.path.exists(image_path):
            print(image_id)
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
        for line in lines:
            dataset.append((image_path, line[0][0], line[0][1]))
    return dataset

train = get_ocr_recog_dataset('train')
validation = get_ocr_recog_dataset('test')

train = [(filepath, box, word) for filepath, box, word in train]
validation = [(filepath, box, word) for filepath, box, word in validation]

augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Affine(
    scale=(1.0, 1.2),
    rotate=(-5, 5)
    ),
    imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
    imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
])
generator_kwargs = {'width': 640, 'height': 640}
batch_size = args.batch_size
augmenter = imgaug.augmenters.Sequential([
    imgaug.augmenters.Rotate()
])

recognizer = recognition.Recognizer()
recognizer.compile()

(training_image_gen, training_steps), (validation_image_gen, validation_steps) = [
    (
        datasets.get_recognizer_image_generator(
            labels=labels,
            height=recognizer.model.input_shape[1],
            width=recognizer.model.input_shape[2],
            alphabet=recognizer.alphabet,
            augmenter=augmenter
        ),
        len(labels) // batch_size
    ) for labels, augmenter in [(train, augmenter), (validation, None)]
]

for label in train:
    print(label)
    break

training_generator, validation_generator = [
    recognizer.get_batch_generator(
        image_generator=image_generator, batch_size=batch_size
    ) for image_generator in
    [training_image_gen, validation_image_gen]
]

recognizer.training_model.fit_generator(
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
