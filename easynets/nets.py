import tensorflow as tf

class Model(tf.keras.models.Sequential):
  def train(self, *args, **kwargs):
    return self.fit(*args, **kwargs)

def create_network(num_layers, layers, activation):
  for i in range(num_layers - 1):
    if i == 0:
      x = [(tf.keras.layers.Dense(layers[1], activation=activation[i], input_shape=(layers[0],)))]
    else:
      x.append(tf.keras.layers.Dense(layers[i + 1], activation=activation[i]))
  x = Model(x)
  x.compile(optimizer='adam',
            loss='mse',
            metrics=['accuracy'])
  return x
