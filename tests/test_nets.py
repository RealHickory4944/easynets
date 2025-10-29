import pytest
import easynets as en
import tensorflow as tf

def test_create_network():
  net = en.nets.create_network(num_layers=4,
                               layers=[2, 5, 5, 1], # input and output included
                               activation=['relu', 'relu', 'sigmoid']) # only hidden and output
  expected = tf.keras.models.Sequential([
    tf.keras.layers.Dense(5, activation='relu', input_shape=(2,)),
    tf.keras.layers.Dense(5, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
  ])
  expected.compile(optimizer='adam',
                   loss='mse',
                   metrics=['accuracy'])
  assert net is expected
