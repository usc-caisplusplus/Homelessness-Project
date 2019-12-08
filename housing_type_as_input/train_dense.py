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

    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state = 0, shuffle = True)

    from model_dense import Dense

    dense = Dense((59))
    dense.build()
    dense.compile()

    print(dense.model.summary())

    dense.model.fit(x_train, y_train, epochs = 20, batch_size=32, validation_data = (x_test, y_test), shuffle = True)
    dense.model.save_weights('dense_weights.h5')
