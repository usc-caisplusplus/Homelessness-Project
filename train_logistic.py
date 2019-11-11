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

    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.model_selection import train_test_split
    from sklearn import metrics

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.1, random_state = 0, shuffle = True)

    logreg = LogisticRegressionCV(penalty = 'elasticnet', solver = 'saga', max_iter = 5000, l1_ratios = [0.1, 0.1], Cs = 10, cv = 5)
    logreg.fit(X_train, Y_train)

    print('Accuracy: {}'.format(logreg.score(X_test, Y_test)))

    predictions = logreg.predict(X_test)
    np.save('logreg_pred.npy', predictions)
    #print(np.hstack((predictions[:, 1], Y_test)))
    print(logreg.coef_)
    np.save('logreg_coeff.npy', logreg.coef_)
