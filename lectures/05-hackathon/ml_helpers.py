"""
Some helper functions for building the model and training
"""

import datetime
import io
import os
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf # can take some time the first time this is imported

from astropy.io import fits
from astropy.visualization import (
    AsymmetricPercentileInterval,
    ImageNormalize,
    LinearStretch,
    LogStretch,
)
from tensorflow.data import Dataset
from tensorflow.keras import layers, Model, optimizers

with open('config.json', 'r') as f:
    config = json.loads(f)

def build_model(config):
    triplet_input = layers.Input(shape=config['triplet_input_shape'], name='triplet')
    metadata_input = layers.Input(shape=config['metadata_input_shape'], name='metadata')

    # add your code here
    # # Build the CNN branch
    # first conv block
    x_conv = layers.Conv2D(16, (3,3), activation='relu', padding='same', name='conv2d_1_1')(triplet_input)
    x_conv = layers.Conv2D(16, (3,3), activation='relu', padding='same', name='conv2d_1_2')(x_conv)
    x_conv = layers.MaxPooling2D(pool_size=(2,2), strides=2, name='maxpool_1')(x_conv)
    x_conv = layers.Dropout(config['dropout_rate_1'], name='dropout_1')(x_conv)
    
    # second conv block
    x_conv = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_2_1')(x_conv)
    x_conv = layers.Conv2D(32, (3,3), activation='relu', padding='same', name='conv2d_2_2')(x_conv)
    x_conv = layers.MaxPooling2D(pool_size=(2,2), strides=4, name='maxpool_2')(x_conv)
    x_conv = layers.Dropout(config['dropout_rate_2'], name='dropout_2')(x_conv)
    
    # we flatten the output
    x_conv = layers.Flatten()(x_conv)
    x_conv = layers.Dense(1, activation='relu', name='fc_first')(x_conv)
    # output = layers.Dense(1, activation='sigmoid', name='fc_out')(x_conv)
    
    # # Build the MLP branch
    # insert your code here
    x_metadata = layers.Dense(64, activation='relu', name='dense1')(meta_input)
    x_metadata = layers.Dropout(config['dropout_rate_3'], name='dropout1')(x_metadata)
    x_metadata = layers.Dense(32, activation='relu', name='dense2')(x_metadata)
    x_metadata = layers.Dropout(config['dropout_rate_3'], name='dropout2')(x_metadata)
    
    # output = layers.Dense(1, activation='sigmoid', name='fc_out')(x_metadata)
    
    # # Combine the two branches and add the rest of the model before the output
    x = layers.Concatenate()([x_conv, x_metadata])

    output = layers.Dense(1, activation='sigmoid', name='fc_out')(x)

    model = Model(inputs=[triplet_input, meta_input], outputs=output, name="acai")

    optimizer = optimizers.Adam(
        learning_rate=config['learning_rate'],
        beta_1=config['beta_1'],
        beta_2=config['beta_2']
    )
    model.compile(optimizer=optimizer, loss=config['loss'], metrics=config['metrics'])

    return model

