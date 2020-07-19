import ws_dataset
import datasets
import imgaug
import tensorflow as tf
import random
import detection
from tensorflow.compat.v1.keras.backend import set_session
import sklearn.model_selection
import numpy as np

"""
Set session to avoid bug
"""
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)

"""
datasets
"""

detector = detection.Detector()



# augmenter = imgaug.augmenters.Sequential([
#     imgaug.augmenters.Affine(
#         scale=(1.0, 1.2),
#         rotate=(-5, 5)
#     ),
#     imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
#     imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
# ])




"""
training
"""
# image, character_bboxes, new_words, confidence_mask, confidences = data

def single_image_loss(pre_loss, loss_label):
    batch_size = pre_loss.shape[0]
    sum_loss = tf.math.reduce_mean(pre_loss.reshape(-1))*0
    pre_loss = pre_loss.reshape(batch_size, -1)
    loss_label = loss_label.reshape(batch_size, -1)

    for i in range(batch_size):
        average_number = 0
        positive_pixel = len(pre_loss[i][(loss_label[i] >= 0.1)])
        average_number += positive_pixel
        if positive_pixel != 0:
            posi_loss = tf.math.reduce_mean(pre_loss[i][(loss_label[i] >= 0.1)])
            sum_loss += posi_loss

            if len(pre_loss[i][(loss_label[i] < 0.1)]) < 3*positive_pixel:
                nega_loss = tf.math.reduce_mean(pre_loss[i][(loss_label[i] < 0.1)])
                average_number += len(pre_loss[i][(loss_label[i] < 0.1)])
            else:
                nega_loss = tf.math.reduce_mean(tf.math.top_k(pre_loss[i][(loss_label[i] < 0.1)], 3*positive_pixel)[0])
                average_number += 3*positive_pixel
            sum_loss += nega_loss
        else:
            nega_loss = tf.math.reduce_mean(tf.math.top_k(pre_loss[i], 500)[0])
            average_number += 500
            sum_loss += nega_loss

    return sum_loss

def maploss(gh_label, gah_label, p_gh, p_gah, mask):
    loss1 = tf.losses.mse(gh_label, p_gh)
    loss2 = tf.losses.mse(gah_label, p_gah)
    loss_g = tf.multiply(loss1, mask)
    loss_a = tf.multiply(loss2, mask)

    char_loss = single_image_loss(loss_g, gh_label)
    affi_loss = single_image_loss(loss_a, gah_label)
    return char_loss/loss_g.shape[0] + affi_loss/loss_a.shape[0]

if __name__ == '__main__':
    import math
    import os

    # synth_dataset = datasets.prepare_dataset('../icdar2013/images')
    # synth_train, synth_validation = sklearn.model_selection.train_test_split(
    #     synth_dataset, train_size=0.8, random_state=42
    # )
    # synth_train_length = len(synth_train)
    # generator_kwargs = {'width': 640, 'height': 640}
    #
    # training_synth_generator = datasets.get_detector_image_generator(
    #     labels=synth_train,
    #     augmenter=None,
    #     **generator_kwargs
    # )
    # validation_synth_generator = datasets.get_detector_image_generator(
    #     labels=synth_validation,
    #     **generator_kwargs
    # )

    real_dataset = ws_dataset.prepare_dataset('../ICDAR2017/training_images/')
    real_train, real_validation = sklearn.model_selection.train_test_split(
        real_dataset, train_size=0.8, random_state=42
    )
    augmenter = imgaug.augmenters.Sequential([
        imgaug.augmenters.Affine(
            scale=(1.0, 1.2),
            rotate=(-5, 5)
        ),
        imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
        imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
    ])

    training_image_generator = ws_dataset.real_data_generator(real_train,
                                                             width=640,
                                                             height=640,
                                                             augmenter=augmenter,
                                                             )
    validation_image_generator = ws_dataset.real_data_generator(real_validation,
                                                               width=640,
                                                               height=640,
                                                               augmenter=None)

    # import tools
    # import cv2
    # canvas = tools.drawBoxes(image=image, boxes=lines, boxes_format='lines')
    # cv2.imwrite('test.png', canvas)

    batch_size = 1
    training_generator, validation_generator = [
        detector.get_batch_generator(
            image_generator=image_generator, batch_size=batch_size
        ) for image_generator in
        [training_image_generator, validation_image_generator]
    ]
    detector.model.fit_generator(
        generator=training_generator,
        steps_per_epoch=math.ceil(len(real_train) / batch_size),
        epochs=10,
        workers=0,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(restore_best_weights=True, patience=5),
            tf.keras.callbacks.CSVLogger(os.path.join('./log', 'detector_icdar2013.csv')),
            tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join('./model', 'detector_icdar2013.h5'))
        ],
        validation_data=validation_generator,
        validation_steps=math.ceil(len(real_validation) / batch_size)
    )