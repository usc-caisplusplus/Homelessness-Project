if __name__ == '__main__':
    import numpy as np
    X = np.load('dataset/inputs.npy')
    Y = np.load('dataset/outputs.npy')
    num_samples = X.shape[0]
    x_train = X[:(num_samples*9)//10]
    y_train = Y[:(num_samples*9)//10]
    x_test = X[(num_samples*9)//10:]
    y_test = Y[(num_samples*9)//10:]

    from model_dense import Dense

    dense = Dense((59))
    dense.build()
    dense.compile()

    dense.model.fit(x_train, y_train, epochs = 3, batch_size=64, validation_data = (x_test, y_test))
    dense.model.save_weights('dense_weights.h5')
