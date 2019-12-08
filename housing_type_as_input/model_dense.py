from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam

from model import Model
class Dense(Model):
    def __init__(self, input_shape):
        super().__init__(input_shape)

    def build(self):
        inputs = layers.Input(shape = self.input_shape)

        dense1 = layers.Dense(60, activation = 'relu')(inputs)
        dense2 = layers.Dense(120, activation = 'relu')(dense1)
        dense3 = layers.Dense(80, activation = 'relu')(dense2)
        dense4 = layers.Dense(40, activation = 'relu')(dense3)
        dense5 = layers.Dense(20, activation = 'relu')(dense4)
        dense6 = layers.Dense(1, activation = 'sigmoid')(dense5)

        self.model = models.Model(inputs = inputs, outputs = dense6)
        super().build()

    def compile(self):
        self.model.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy', 'binary_accuracy'])
        super().compile()
