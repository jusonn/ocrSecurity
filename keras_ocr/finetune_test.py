import cv2
import detection
import numpy as np
import tools
import tensorflow as tf
from tensorflow.compat.v1.keras.backend import set_session
import pipeline

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
set_session(sess)


def output(img_path, out_path):
    img = cv2.imread(img_path)
    # img = cv2.resize(img, (img.shape[1] // 2, img.shape[0] // 2))
    detector = detection.Detector()
    detector.model.load_weights('model/detector_icdar2013.h5')

    pipe = pipeline.Pipeline(detector=detector)
    predictions = pipe.recognize(images=[img])[0]
    drawn = tools.drawBoxes(
        image=img, boxes=predictions, boxes_format='predictions'
    )
    print(
        'Predicted:', [text for text, box in predictions]
    )
    cv2.imwrite(out_path, drawn)

# img = cv2.imread('../ocr_train/012.png')
# '../ICDAR2017/training_images/img_5042.jpg'

output('../ICDAR2017/training_images/img_1921.jpg', 'real1.png')
output('../ICDAR2017/training_images/img_2003.jpg', 'real2.png')
output('../ICDAR2017/training_images/img_4358.jpg', 'real3.png')
output('../ICDAR2017/training_images/img_4055.jpg', 'real4.png')
output('../ocr_train/012.png', 'doc.png')