import tensorflow as tf

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
  x.train = x.fit
  return x
