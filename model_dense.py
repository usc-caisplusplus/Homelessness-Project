from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from model import Model
class Dense(Model):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def build(self):
        inputs = layers.Input(shape = self.input_shape)

        dense1 = layers.Dense(120, activation = 'relu')(inputs)
        dense2 = layers.Dense(60, activation = 'relu')(dense1)
        dense3 = layers.Dense(30, activation = 'relu')(dense2)
        dense4 = layers.Dense(15, activation = 'relu')(dense3)
        dense5 = layers.Dense(1, activation = 'sigmoid')(dense4)

        self.model = models.Model(inputs = inputs, outputs = dense5)
        super().build()

    def compile(self):
        self.model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])
        super().compile()
