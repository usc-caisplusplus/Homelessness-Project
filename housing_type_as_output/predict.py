if __name__ == '__main__':
    t = 'ssvf'

    import numpy as np
    X = np.load('/home/haydenshively/Developer/Homelessness-Project/dataset/allinputs.npy')
    Y = np.load('/home/haydenshively/Developer/Homelessness-Project/dataset/alloutputs.npy')

    from model_dense import Dense
    dir = '/home/haydenshively/Developer/Homelessness-Project/weights/'

    dense_psh = Dense((52,))
    dense_psh.build()
    dense_psh.compile()
    dense_psh.model.load_weights(dir + 'psh.h5')
    psh_outcomes = dense_psh.model.predict(X)

    dense_ssvf = Dense((52,))
    dense_ssvf.build()
    dense_ssvf.compile()
    dense_ssvf.model.load_weights(dir + 'ssvf.h5')
    ssvf_outcomes = dense_ssvf.model.predict(X)

    dense_rrh = Dense((52,))
    dense_rrh.build()
    dense_rrh.compile()
    dense_rrh.model.load_weights(dir + 'rrh.h5')
    rrh_outcomes = dense_rrh.model.predict(X)

    dense_family = Dense((52,))
    dense_family.build()
    dense_family.compile()
    dense_family.model.load_weights(dir + 'family.h5')
    family_outcomes = dense_family.model.predict(X)

    dense_hudvash = Dense((52,))
    dense_hudvash.build()
    dense_hudvash.compile()
    dense_hudvash.model.load_weights(dir + 'HUD-VASH.h5')
    hudvash_outcomes = dense_hudvash.model.predict(X)

    dense_selfresolve = Dense((52,))
    dense_selfresolve.build()
    dense_selfresolve.compile()
    dense_selfresolve.model.load_weights(dir + 'selfresolve.h5')
    selfresolve_outcomes = dense_selfresolve.model.predict(X)

    all_outcomes = np.hstack((psh_outcomes, ssvf_outcomes, rrh_outcomes, family_outcomes, hudvash_outcomes, selfresolve_outcomes))
    # np.save('all_predicted_outcomes.npy', all_outcomes)
    best_outcomes = all_outcomes.argmax(axis=1)

    actual_outcomes = np.load('housing_types.npy')
    actual_outs_num = np.zeros_like(actual_outcomes, dtype='uint8')
    actual_outs_num[actual_outcomes == 'psh'] = 0
    actual_outs_num[actual_outcomes == 'ssvf'] = 1
    actual_outs_num[actual_outcomes == 'rrh'] = 2
    actual_outs_num[actual_outcomes == 'family'] = 3
    actual_outs_num[actual_outcomes == 'HUD-VASH'] = 4
    actual_outs_num[actual_outcomes == 'self resolve'] = 5

    matches = best_outcomes == actual_outs_num
    print('Individuals who were placed in optimal housing, according to our models:')
    print(matches[matches == True].shape)
    print('Total number of individuals')
    print(best_outcomes.shape)
    print('')

    print('Type of housing distribution, if everyone were placed optimally:')
    print('psh')
    print(best_outcomes[best_outcomes == 0].shape[0]/best_outcomes.shape[0])
    print('ssvf')
    print(best_outcomes[best_outcomes == 1].shape[0]/best_outcomes.shape[0])
    print('rrh')
    print(best_outcomes[best_outcomes == 2].shape[0]/best_outcomes.shape[0])
    print('family')
    print(best_outcomes[best_outcomes == 3].shape[0]/best_outcomes.shape[0])
    print('HUD-VASH')
    print(best_outcomes[best_outcomes == 4].shape[0]/best_outcomes.shape[0])
    print('Self Resolve')
    print(best_outcomes[best_outcomes == 5].shape[0]/best_outcomes.shape[0])

