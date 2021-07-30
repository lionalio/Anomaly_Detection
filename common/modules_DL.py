from keras.layers import Dense, Embedding, LSTM, \
    SpatialDropout1D, Conv1D, MaxPooling1D, BatchNormalization, \
        Dropout, Flatten, BatchNormalization
from keras import Model, Input, regularizers
from keras.models import Sequential


def model_auto_encoder(input_shape):
    input_df = Input( shape = (input_shape, ))
    x = Dense(10, activation = 'relu', kernel_regularizer=regularizers.l2(0.0))(input_df)
    x = Dense(5, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    encoded = Dense(10, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    decoded = Dense(input_shape, kernel_initializer='glorot_uniform')(x)
    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return autoencoder, encoder


def model_auto_encoder2(input_shape):
    input_df = Input( shape = (input_shape, ))
    x = Dense(10, activation = 'relu')(input_df)
    x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(2000, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    encoded = Dense(10, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    x = Dense(2000, activation = 'relu', kernel_initializer='glorot_uniform')(encoded)
    x = Dense(500, activation = 'relu', kernel_initializer='glorot_uniform')(x)
    decoded = Dense(input_shape, kernel_initializer='glorot_uniform')(x)
    autoencoder = Model(input_df, decoded)
    encoder = Model(input_df, encoded)
    autoencoder.compile(optimizer = 'adam', loss = 'mean_squared_error')

    return autoencoder, encoder