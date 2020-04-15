import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow import optimizers, metrics, losses

class DNN(Model):
    def __init__(self, hidden_dim=500, output_dim=5, dropout_rate=0.75):
        super().__init__()
        self.l1 = Dense(hidden_dim, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        use_bias=True, activity_regularizer=None, bias_regularizer=None, kernel_regularizer=None,
                        bias_constraint=None, kernel_constraint=None, name="dense1")
        self.d1 = Dropout(rate=0.75, noise_shape=None, name="dropout1")
        self.l2 = Dense(hidden_dim, activation='relu', kernel_initializer='glorot_uniform', bias_initializer='zeros',
                        use_bias=True, activity_regularizer=None, bias_regularizer=None, kernel_regularizer=None,
                        bias_constraint=None, kernel_constraint=None, name="dense2")
        self.d2 = Dropout(rate=0.75, noise_shape=None, name="dropout2")
        self.y = Dense(output_dim, activation='softmax', kernel_initializer='glorot_uniform', bias_initializer='zeros',
                       use_bias=True, activity_regularizer=None, kernel_regularizer=None, bias_regularizer=None,
                       kernel_constraint=None, bias_constraint=None, name="output")
        self.ls = [self.l1, self.d1, self.l2, self.d2, self.y]

    def call(self, x):
        for layer in self.ls:
            x = layer(x)
        return x
