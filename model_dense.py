from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from model import Model
class Dense(Model):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def build(self):
        inputs = layers.Input(shape = self.input_shape)

        dense1 = layers.Dense(40, activation = 'relu')(inputs)
        dense2 = layers.Dense(20, activation = 'relu')(dense1)
        dense3 = layers.Dense(1, activation = 'sigmoid')(dense2)

        self.model = models.Model(inputs = inputs, outputs = dense3)
        super().build()

    def compile(self):
        self.model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
        super().compile()
