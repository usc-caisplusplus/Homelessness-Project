if __name__ == '__main__':
    import numpy as np
    X = np.load('dataset/inputs.npy')
    Y = np.load('dataset/outputs.npy')

    true_mask = (Y == 1).astype('bool').flatten()

    X_true = X[true_mask]
    X_false = X[~true_mask]
    Y_true = Y[true_mask]
    Y_false = Y[~true_mask]

    smaller = min(X_true.shape[0], X_false.shape[0])
    X = np.vstack((X_true[:smaller], X_false[:smaller]))
    Y = np.vstack((Y_true[:smaller], Y_false[:smaller]))

    print(X.shape)

    num_samples = X.shape[0]
    x_train = X[:(num_samples*9)//10]
    y_train = Y[:(num_samples*9)//10]
    x_test = X[(num_samples*9)//10:]
    y_test = Y[(num_samples*9)//10:]

    from model_dense import Dense

    dense = Dense((59))
    dense.build()
    dense.compile()

    print(dense.model.summary())

    dense.model.fit(x_train, y_train, epochs = 20, batch_size=16, validation_data = (x_test, y_test), shuffle = True)
    dense.model.save_weights('dense_weights.h5')
