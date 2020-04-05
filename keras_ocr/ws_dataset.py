from detection import Detector, getBoxes
from recognition import Recognizer
import itertools
import random
import tools
import numpy as np
import os
import imgaug
import Polygon as plg
import cv2
import imgproc
from utils import *
import gaussian
from mep import mep

BASE_PATH = '../ICDAR2017/'
gaussianTransformer = gaussian.GaussianTransformer(imgSize=1024, region_threshold=0.35, affinity_threshold=0.15)

def read_gt(txt):
    # lines = open(txt, encoding='utf-8').readlines()
    # bboxes = []
    # words = []
    # for line in lines:
    #     ori_box = line.strip().encode('utf-8').decode('utf-8-sig').split(',')
    #     box = [int(ori_box[j]) for j in range(8)]
    #     word = ori_box[8:]
    #     word = ','.join(word)
    #     box = np.array(box, np.int32).reshape(4, 2)
    #     if word == '###':
    #         words.append('###')
    #         bboxes.append(box)
    #         continue
    #     area, p0, p3, p2, p1, _, _ = mep(box)
    #
    #     bbox = np.array([p0, p1, p2, p3])
    #     distance = 10000000
    #     index = 0
    #     for i in range(4):
    #         d = np.linalg.norm(box[0] - bbox[i])
    #         if distance > d:
    #             index = i
    #             distance = d
    #     new_box = []
    #     for i in range(index, index + 4):
    #         new_box.append(bbox[i % 4])
    #     new_box = np.array(new_box)
    #     bboxes.append(np.array(new_box))
    #     words.append(word)
    # return bboxes, words
    label = []
    with open(txt) as f:
        gts = f.readlines()

        for gt in gts:
            bboxes = []
            words = []
            ori_box = gt.strip().encode('utf-8').decode('utf-8-sig').split(',')
            box = [int(ori_box[j]) for j in range(8)]
            word = ori_box[9:]
            word = ','.join(word)
            box = np.array(box, np.int32).reshape(4, 2)
            if word == '###':
                words.append('###')
                bboxes.append(box)
                continue
            if len(word.strip()) == 0:
                continue

            try:
                area, p0, p3, p2, p1, _, _ = mep(box)
            except Exception as e:
                print(e)

            bbox = np.array([p0, p1, p2, p3])
            distance = 10000000
            index = 0
            for i in range(4):
                d = np.linalg.norm(box[0] - bbox[i])
                if distance > d:
                    index = i
                    distance = d
            new_box = []
            for i in range(index, index + 4):
                new_box.append(bbox[i % 4])
            new_box = np.array(new_box)
            label.append((new_box, word))
    return label


def prepare_data(img_path, data_id):
    loc_path = os.path.join(BASE_PATH, 'loc_gt', f'gt_img_{data_id}.txt')
    loc_data = read_gt(loc_path)
    return [img_path, loc_data]


def prepare_dataset():
    image_paths = os.listdir('../ICDAR2017/training_images/')
    dataset = []
    for image_path in image_paths:
        data_id = image_path.split('_')[1].split('.')[0]
        image_path = os.path.join('../ICDAR2017/training_images/', image_path)
        data = prepare_data(image_path, data_id)
        dataset.append(data)
    return dataset

def get_detector_ws_image_generator(detector,
                                    labels,
                                    width,
                                    height,
                                    augmenter=None,
                                    area_threshold=0.5,
                                    focused=False,):
    labels = labels.copy()

    for index in itertools.cycle(range(len(labels))):
        # if index == 0:
        #     random.shuffle(labels)
        image_filepath, lines = labels[index]
        image = tools.read(image_filepath)

        if augmenter is not None:
            image, lines = tools.augment(boxes=lines,
                                         boxes_format='words',
                                         image=image,
                                         area_threshold=area_threshold,

                                         augmenter=augmenter)

        image, scale = tools.fit(image,
                                width=width,
                                 height=height,
                                 mode='letterbox',
                                 return_scale=True)

        lines = tools.adjust_boxes(boxes=lines, boxes_format='words', scale=scale)
        # lines = [lines]
        confidence_mask = np.zeros((image.shape[0], image.shape[1]), np.float32)

        confidences = []
        character_bboxes = []
        new_words = []

        detector = Detector()

        for i, line in enumerate(lines[0]):
            word_bbox, word = line[0], line[1]
            word_bbox = np.float32(word_bbox)
            if len(word_bbox) > 0:
                for _ in range(len(word_bbox)):
                    if word == '###' or len(word.strip()) == 0:

                        cv2.fillPoly(confidence_mask, [np.int32(word_bbox)], (0))
            pursedo_bboxes, bbox_region_scores, confidence = inference_character_box(detector,
                                                                                     image,
                                                                                     word,
                                                                                     word_bbox)
            confidences.append(confidence)
            cv2.fillPoly(confidence_mask, [np.int32(word_bbox)], (confidence))
            new_words.append(word)
            character_bboxes.append(pursedo_bboxes)

        yield image, character_bboxes, new_words, confidence_mask, confidences



def resizeGt(gtmask, size):
    return cv2.resize(gtmask, size)

if __name__ == '__main__':
    import tensorflow as tf
    import cv2
    import sklearn
    import datasets
    from tensorflow.compat.v1.keras.backend import set_session

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    config.log_device_placement = True  # to log device placement (on which device the operation ran)
    sess = tf.compat.v1.Session(config=config)
    set_session(sess)

    detector = Detector(load_from_torch=False)

    augmenter = imgaug.augmenters.Sequential([
        imgaug.augmenters.Affine(
            scale=(1.0, 1.2),
            rotate=(-5, 5)
        ),
        imgaug.augmenters.GaussianBlur(sigma=(0, 0.5)),
        imgaug.augmenters.Multiply((0.8, 1.2), per_channel=0.2)
    ])

    dataset = prepare_dataset()
    data_loader = get_detector_ws_image_generator(detector,
                                                  dataset,
                                                  width=640,
                                                  height=640,
                                                  augmenter=None,
                                                  )

    for i in range(10):
        data = next(data_loader)

        # image, character_bboxes, new_words, confidence_mask, confidence
        image, character_bboxes, new_words, confidence_mask, confidences = data

        if len(confidences) == 0:
            confidences = 1.0
        else:
            confidences = np.array(confidences).mean()

        region_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_scores = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        affinity_bboxes = []

        if len(character_bboxes) > 0:
            region_scores = gaussianTransformer.generate_region(region_scores.shape, character_bboxes)
            affinity_scores, affinity_bboxes = gaussianTransformer.generate_affinity(region_scores.shape,
                                                                                     character_bboxes,
                                                                                     new_words)

        saveImage(f'test{i}.png', image.copy(), character_bboxes,
                  affinity_bboxes, region_scores,
                  affinity_scores, confidence_mask)
        if i == 100:
            break


# res = detector.detect([img])
    # for r in res:
    #     cavas = tools.drawBoxes(img, r)
    # cv2.imwrite('tt.png', cavas)
