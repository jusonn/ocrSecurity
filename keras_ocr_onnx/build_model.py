import typing
import tensorflow as tf
from tensorflow import keras
import numpy as np
import keras2onnx
from util_rec import _transform

"""
    'height': 31,
    'width': 200,
    'color': False,
    'filters': (64, 128, 256, 256, 512, 512, 512),
    'rnn_units': (128, 128),
    'dropout': 0.25,
    'rnn_steps_to_discard': 2,
    'pool_size': 2,
    'stn': True,
"""

def build_model(alphabet, height, width, color, filters, rnn_units, dropout,
                rnn_steps_to_discard, pool_size, stn=True):
    inputs = keras.layers.Input((height, width, 3 if color else 1), batch_size=1)
    x = keras.layers.Permute((2, 1, 3))(inputs)
    # x = keras.layers.Lambda(lambda x: x[:, :, ::-1])(x)
    # x = x[..., ::-1]
    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', name='conv_1')(x)
    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', name='conv_2')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_3')(x)
    x = keras.layers.BatchNormalization(name='bn_3')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2,  2), name='maxpool_3')(x)
    x = keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', name='conv_4')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_5')(x)
    x = keras.layers.BatchNormalization(name='bn_5')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2), name='maxpool_5')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_6')(x)
    x = keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', name='conv_7')(x)
    x = keras.layers.BatchNormalization(name='bn_7')(x)

    stn_input_output_shape = (width // pool_size**2, height // pool_size**2, 512)
    stn_input_layer = keras.layers.Input(stn_input_output_shape, batch_size=1)
    locnet_y = keras.layers.Conv2D(16, (5, 5), padding='same', activation='relu')(stn_input_layer)
    locnet_y = keras.layers.Conv2D(32, (5, 5), padding='same', activation='relu')(locnet_y)
    locnet_y = keras.layers.Flatten()(locnet_y)
    locnet_y = keras.layers.Dense(64, activation='relu')(locnet_y)
    locnet_y = keras.layers.Dense(6, weights=[
        np.zeros((64, 6), dtype=np.float32),
        np.float32([[1, 0, 0], [0, 1, 0]]).flatten()
    ])(locnet_y)
    localization_net = keras.models.Model(inputs=stn_input_layer, outputs=locnet_y)

    x= keras.layers.Lambda(_transform,
                           output_shape=stn_input_output_shape)([x, localization_net(x)])
    x = keras.layers.Reshape(target_shape=(width // pool_size**2,
                                           (height // pool_size ** 2) * 512),
                            name='reshape')(x)
    # x = keras.layers.Dense(4)(x)
    x = keras.models.Model(inputs=inputs, outputs=x)
    return x


if __name__ == '__main__':
    import os

    model = build_model(1,31, 200, 1, 7, 2, 0.1, 2, 2)
    # model.summary()
    onnx_model = keras2onnx.convert_keras(model, model.name)
    print(dir(onnx_model))
    # print(onnx_model.graph)
    keras2onnx.save_model(onnx_model, 'model.onnx')

    print('ocr2onnx DONE')

    # os.system('./onnx2trt model.onnx -o model.trt -d 16 -b 1')