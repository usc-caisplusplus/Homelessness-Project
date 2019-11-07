if __name__ = '__main__':
    from model_dense import Dense

    dense = Dense((41))
    dense.build()
    dense.compile()

    dense.model.fit(x_train, y_train, epochs = 1, batch_size=64, validation_data = (x_test, y_test))
    dense.model.save_weights('dense_weights.h5')
