import typing
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras2onnx


def build_model(alphabet, height, width, color, filters, rnn_units, dropout,
                rnn_steps_to_discard, pool_size, stn=True):
    inputs = keras.layers.Input((height, width, 3 if color else 1), batch_size=16)
    x = keras.layers.Permute((2, 1, 3))(inputs)
    
    x = keras.layers.Dense(4)(x)
    x = keras.models.Model(inputs=inputs, outputs=x)
    return x


if __name__ == '__main__':
    import os

    model = build_model(1,24, 24, 3, 7, 2, 0.1, 2, 4)
    onnx_model = keras2onnx.convert_keras(model, model.name)
    keras2onnx.save_model(onnx_model, 'model.onnx')

    print('ocr2onnx DONE')

    os.system('./onnx2trt model.onnx -o model.trt -d 16')
    print('onnx2trt DONE')