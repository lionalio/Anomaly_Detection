from keras.backend.cntk_backend import binary_crossentropy
import tensorflow as tf
from keras.layers import Dense, Layer, Embedding, LSTM, \
    SpatialDropout1D, Conv1D, MaxPooling1D, BatchNormalization, \
        Dropout, Flatten, BatchNormalization
from keras import Model, Input, regularizers
from keras.layers.core import Activation
from keras.models import Sequential
from keras.metrics import Mean
from tensorflow.python.eager.backprop import GradientTape
from keras import backend as K


# Ref: 
# 1/ https://ml-lectures.org/docs/unsupervised_learning/Anomaly_Detection_RNN_AE_VAE.html
# 2/ https://keras.io/examples/generative/vae/
# 3/ https://stanford.edu/~jduchi/projects/general_notes.pdf

class Sampling(Layer):
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch, dim = tf.shape(z_mean)[0], tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))

        return z_mean - tf.exp(0.5*z_log_var) * epsilon


def model_encoder(input_data):
    inputs = Input(shape=(input_data.shape[1], ))
    x = Dense(32, activation='relu')(inputs)
    x = Dense(16, activation='relu')(x)
    x = Dense(8, activation='relu')(x)

    z_mean = Dense(8)(x)
    z_log_var = Dense(8)(x)

    z = Sampling()([z_mean, z_log_var])
    encoder = Model(inputs, [z_mean, z_log_var, z])

    return encoder


def model_decoder(input_data):
    inputs = Input(shape=(8,))
    x = Dense(16, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    outputs = Dense(input_data.shape[1], activation='relu')(x)
    decoder = Model(inputs, outputs)

    return decoder


class VAE(Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = Mean(name="total_loss")
        self.reconstruction_loss_tracker = Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    def train_step(self, data):
        if isinstance(data, tuple) or isinstance(data, list):
            data = data[0]
        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            rec = self.decoder(z)
            rec_loss = tf.reduce_mean(
                tf.keras.losses.binary_crossentropy(data, rec)
            )
            kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
            kl_loss = tf.reduce_mean(kl_loss)
            kl_loss *= -0.5
            tot_loss = rec_loss + kl_loss
        grads = tape.gradient(tot_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(tot_loss)
        self.reconstruction_loss_tracker.update_state(rec_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded[0])

        return decoded