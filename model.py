import tensorflow as tf

from HyperParameters import HP

def get_decoderRNN():
    #last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)
    decoder_input = tf.keras.Input(shape=(1, HP.latent_dim + HP.action_dimension))
    decoder_h0 = tf.keras.Input(shape=(HP.dec_hidden_size))
    decoder_c0 = tf.keras.Input(shape=(HP.dec_hidden_size))

    decoderLSTM = tf.keras.layers.LSTM(HP.dec_hidden_size, return_sequences=True,
                                       return_state=True, name = "LSTM_decoder")

    decoder_output, _, _ = decoderLSTM(decoder_input, initial_state=[decoder_h0, decoder_c0])

    decoder_output = tf.keras.layers.BatchNormalization()(decoder_output)

    output_dimension = HP.action_dimension
    distribution_output = tf.keras.layers.Dense(output_dimension, activation="softmax",
                                                name="output_layer")(decoder_output)

    #distribution_output = distribution_output*47

    model = tf.keras.Model([decoder_input, decoder_h0, decoder_c0],
                           outputs = distribution_output)

    #model.load_weights("models/model_weight_carrot_50_epochs.h5", by_name = True)

    return model

class Sampling(tf.keras.layers.Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_log_var)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch,dim))
    return z_mean + tf.exp(0.5*z_log_var) * epsilon

class HPF(tf.keras.layers.Layer):
  def call(self, inputs):
    hpf = tf.constant([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], tf.float32)
    hpf = tf.expand_dims(tf.expand_dims(hpf, -1), -1)
    return tf.nn.conv2d(inputs, hpf, strides=[1,1,1,1], padding='SAME')

def CNNModel():
  def convolutional(filters, strides, activation='relu'):
    return tf.keras.layers.Conv2D(filters, 2, strides, activation=activation, padding='SAME')
    
  state_inp = tf.keras.Input(shape=(HP.state_dims[0], HP.state_dims[1], 1))
  #state_inp = tf.keras.layers.BatchNormalization()(state_inp)
  
  x = HPF()(state_inp)
  x = convolutional(4, 2)(x)
  x = convolutional(4, 1)(x)
  x = convolutional(8, 2)(x)
  x = convolutional(8, 1)(x)
  x = convolutional(8, 2)(x)
  x = convolutional(8, 1, activation='tanh')(x)
  x = tf.keras.layers.Flatten()(x)
  x = tf.keras.layers.BatchNormalization()(x)
  
  mu = tf.keras.layers.Dense(HP.latent_dim, activation='linear')(x)
  presig = tf.keras.layers.Dense(HP.latent_dim, activation='linear')(x)

  latent_z = Sampling()((mu, presig))
  
  model = tf.keras.Model(state_inp, outputs=latent_z)

  return model

def getCritic():
  state_inp = tf.keras.Input(shape=(HP.state_dims[0], HP.state_dims[1], 1))
  prev_action_inp = tf.keras.Input(shape=(HP.action_dimension))
  action_inp = tf.keras.Input(shape=(HP.action_dimension))

  cnn_enc = CNNModel()
  
  #net_inp = tf.keras.layers.Concatenate()((state_inp, goal_inp))

  enc = cnn_enc(state_inp)
  
  x = tf.keras.layers.Concatenate()((enc, prev_action_inp, action_inp))
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dense(256, activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dense(64, activation='relu')(x)
  x = tf.keras.layers.BatchNormalization()(x)
  x = tf.keras.layers.Dense(1, activation='linear')(x)

  return tf.keras.Model([state_inp, prev_action_inp, action_inp], outputs=x)

def getActor():
  state_inp = tf.keras.Input(shape=(HP.state_dims[0], HP.state_dims[1], 1))
  prev_action_inp = tf.keras.Input(shape=(HP.action_dimension))

  decoder = get_decoderRNN()
  cnn_enc = CNNModel()

  enc = cnn_enc(state_inp)
  
  enc = tf.keras.layers.BatchNormalization()(enc)
  states = tf.keras.layers.Dense(2*HP.dec_hidden_size, activation='tanh')(enc)
  h0, c0 = tf.split(states, 2, axis=1)
  inp = tf.keras.layers.Concatenate()((enc, prev_action_inp))
  inp = tf.keras.layers.BatchNormalization()(inp)

  action = decoder([tf.expand_dims(inp, axis=1), h0, c0])

  return tf.keras.Model([state_inp, prev_action_inp], outputs=action)
  
@tf.function
def update_target(target_weights, weights, tau=HP.tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))
