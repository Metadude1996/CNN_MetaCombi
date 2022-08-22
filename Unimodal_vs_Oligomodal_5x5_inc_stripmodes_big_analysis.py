import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os

def cm_to_inch(x):
    return x / 2.54

def testset_output_allCNN():
    "load in pre-k train and test data"
    dataprek = np.load(u'.\\data_unimodal_vs_oligomodal_inc_stripmodes_train_trainraw_test_5x5.npz')
    indU = np.argwhere(dataprek['y_test'] == 0)
    indO = np.argwhere(dataprek['y_test'] == 1)
    indUO = np.append(indU, indO)
    x_test = dataprek['x_test'][indUO]
    y_test = dataprek['y_test'][indUO]
    nf1 = 20
    nf2 = 80
    nf3 = 160
    nh1 = 1000
    lr = 0.0005
    N_kfolds = 10

    for kfold in range(N_kfolds):
        filepath = u".\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\saves\\nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}" \
                   u"\\kfold={:d}\\1\\".format(nf1, nf2, nf3, nh1, lr, kfold)
        # if os.path.isfile(
        #         'D:\\data\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testres_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}'
        #         '_kfold{:d}.npy'.
        #                 format(nf1, nf2, nf3, nh1, lr, kfold)):
        #     print('skip nf1{:d} nf2{:d} nf3{:d} nh1{:d} lr{:.4f} kfold{:d}'.format(nf1, nf2, nf3, nh1, lr, kfold))
        #     continue
        if os.path.isfile(filepath + 'variables\\variables.data-00001-of-00002') and os.path.isfile(
                filepath + 'variables\\variables.data-00000-of-00002'):
            print('loading model')
            model = tf.keras.models.load_model(
                filepath, custom_objects=None, compile=False
            )
            model.summary()
            # np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index,
            #          val_index=val_index)
            # ind_trainval = np.load(
            #     u'D:\\data\\cnniter_HP_GS_SKF_5x5_AB_5050\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\kfold_data_indices.npz'.format(
            #         nf, nh, lr, kfold))
            # ind_val = ind_trainval['val_index']
            outval = model.predict(x_test)
            np.save(
                'D:\\data\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testres_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}'
                '_kfold{:d}.npy'.format(
                    nf1, nf2, nf3, nh1, lr, kfold), outval)
            tf.keras.backend.clear_session()
        elif os.path.isfile(filepath + 'variables\\variables.data-00000-of-00001'):
            print('loading model')
            model = tf.keras.models.load_model(
                filepath, custom_objects=None, compile=False
            )
            model.summary()
            # np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index,
            #          val_index=val_index)
            # ind_trainval = np.load(
            #     u'D:\\data\\cnniter_HP_GS_SKF_5x5_AB_5050\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\kfold_data_indices.npz'.format(
            #         nf, nh, lr, kfold))
            # ind_val = ind_trainval['val_index']
            outval = model.predict(x_test)
            np.save(
                'D:\\data\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testres_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}'
                '_kfold{:d}.npy'.format(
                    nf1, nf2, nf3, nh1, lr, kfold), outval)
            tf.keras.backend.clear_session()
    return 0


def CM_testset():
    dataprek = np.load(u'.\\data_unimodal_vs_oligomodal_inc_stripmodes_train_trainraw_test_5x5.npz')
    indU = np.argwhere(dataprek['y_test'] == 0)
    indO = np.argwhere(dataprek['y_test'] == 1)
    indUO = np.append(indU, indO)
    x_test = dataprek['x_test'][indUO]
    y_test = dataprek['y_test'][indUO]

    nf1 = 20
    nf2 = 80
    nf3 = 160
    nh1 = 1000
    lr = 0.0005
    CM_mat = np.zeros((10, 5), dtype=int)
    for kfold in range(10):
        outval = np.load(
            'D:\\data\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testres_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}'
            '_kfold{:d}.npy'.format(
                nf1, nf2, nf3, nh1, lr, kfold))
        outclass = np.argmax(outval, axis=1)
        TP = np.sum(np.logical_and(y_test == 1, outclass == 1))
        FP = np.sum(np.logical_and(y_test == 0, outclass == 1))
        TN = np.sum(np.logical_and(y_test == 0, outclass == 0))
        FN = np.sum(np.logical_and(y_test == 1, outclass == 0))

        CM_mat[kfold, 0] = kfold

        CM_mat[kfold, 1] = TP
        CM_mat[kfold, 2] = FP
        CM_mat[kfold, 3] = TN
        CM_mat[kfold, 4] = FN
    np.save('D:\\data\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testset_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}'
            '_TP_FP_TN_FN.npy'.format(nf1, nf2, nf3, nh1, lr), CM_mat)
    return 0

def plot_BA_Volume_UO_vs_CI():
    k = 5
    nf1 = 20
    nf2 = 80
    nf3 = 160
    nh1 = 1000
    lr = 0.0005
    ConMat = np.load(r'D:\\data\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testset_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}'
                     r'_lr={:.4f}_TP_FP_TN_FN.npy'.format(nf1, nf2, nf3, nh1, lr))
    total_pred = np.sum(ConMat[-1, :])
    beta_class = np.divide(ConMat[-1, 1] + ConMat[-1, -1, -1, -1, 4], total_pred)
    testPM_nfnh_UO = np.load('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testPM_nfnh_{:d}x{:d}.npy'.format(k, k))
    testPM_nfnh_CI = np.load('.\\testPM_nfnh_{:d}x{:d}.npy'.format(k, k))
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    "plot BA as function of nh"
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    f, ax = plt.subplots()
    ax.plot(nh_list, testPM_nfnh_UO[0, :, 2, 0], '.-', label='Unimodal vs Oligomodal', c='tab:blue')
    ax.fill_between(nh_list, testPM_nfnh_UO[0, :, 2, 0]-np.sqrt(testPM_nfnh_UO[0, :, 2, 1]),
                    testPM_nfnh_UO[0, :, 2, 0] + np.sqrt(testPM_nfnh_UO[0, :, 2, 1]), alpha=0.3, color='tab:blue')
    ax.plot(nh_list, testPM_nfnh_CI[-1, :, 2, 0], '.-', label='Incompatible vs Compatible', c='tab:pink')
    ax.fill_between(nh_list, testPM_nfnh_CI[-1, :, 2, 0] - np.sqrt(testPM_nfnh_CI[-1, :, 2, 1]),
                    testPM_nfnh_CI[-1, :, 2, 0] + np.sqrt(testPM_nfnh_CI[-1, :, 2, 1]), alpha=0.3, color='tab:pink')
    ax.set_xlabel('$n_h$')
    ax.set_ylabel('$\mathrm{BA}$')
    ax.set_yscale('log')
    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.minorticks_off()
    plt.legend()
    f.savefig('.\\cnniter_Unimodal_vs_Oligomodal_inc_stripmodes_5x5\\BalAcc_vs_nh_nf20_UO_vs_CI.pdf', facecolor=f.get_facecolor())
    f.savefig('.\\cnniter_Unimodal_vs_Oligomodal_inc_stripmodes_5x5\\BalAcc_vs_nh_nf20_UO_vs_CI.svg', facecolor=f.get_facecolor())
    f.savefig('.\\cnniter_Unimodal_vs_Oligomodal_inc_stripmodes_5x5\\BalAcc_vs_nh_nf20_UO_vs_CI.png', dpi=400, facecolor=f.get_facecolor())
    plt.show()
    plt.close()

    f, ax = plt.subplots()
    ax.plot(nh_list, testPM_nfnh_UO[0, :, 6, 0] - testPM_nfnh_UO[0, :, 8, 0], '.-', c='tab:blue',
            label='Unimodal vs Oligomodal')
    ax.fill_between(nh_list, testPM_nfnh_UO[0, :, 6, 0] - testPM_nfnh_UO[0, :, 8, 0] -
                    (np.sqrt(testPM_nfnh_UO[0, :, 6, 1]) + np.sqrt(testPM_nfnh_UO[0, :, 8, 1])),
                    testPM_nfnh_UO[0, :, 6, 0] - testPM_nfnh_UO[0, :, 8, 0] +
                    (np.sqrt(testPM_nfnh_UO[0, :, 6, 1]) + np.sqrt(testPM_nfnh_UO[0, :, 8, 1])), alpha=0.3,
                    color='tab:blue')
    ax.plot(nh_list, testPM_nfnh_CI[-1, :, 5, 0] - beta_class, '.-', c='tab:pink', label='Incompatible vs Compatible')
    ax.fill_between(nh_list, testPM_nfnh_CI[-1, :, 5, 0] - beta_class - np.sqrt(testPM_nfnh_CI[0, :, 5, 1]),
                    testPM_nfnh_CI[-1, :, 5, 0] - beta_class + np.sqrt(testPM_nfnh_CI[0, :, 5, 1]), alpha=0.3,
                    color='tab:pink')
    ax.set_xlabel('$n_h$')
    ax.set_ylabel(r'$\bar{\beta}-\beta$')
    ax.set_yscale('log')
    # ax.yaxis.set_major_locator(MaxNLocator(30))
    ax.minorticks_off()
    plt.legend()
    f.savefig('.\\cnniter_Unimodal_vs_Oligomodal_inc_stripmodes_5x5\\Volume_vs_nh_nf20_UO_vs_CI.pdf', facecolor=f.get_facecolor())
    f.savefig('.\\cnniter_Unimodal_vs_Oligomodal_inc_stripmodes_5x5\\Volume_vs_nh_nf20_UO_vs_CI.svg', facecolor=f.get_facecolor())
    f.savefig('.\\cnniter_Unimodal_vs_Oligomodal_inc_stripmodes_5x5\\Volume_vs_nh_nf20_UO_vs_CI.png', dpi=400,
              facecolor=f.get_facecolor())
    plt.show()
    plt.close()
    return 0

def analyse_wrongclassification():
    "load data"
    k = 5
    nf1 = 20
    nf2 = 80
    nf3 = 160
    nh1 = 1000
    lr = 0.0005
    extended = False
    if k < 5:
        datastring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\PixelRep_{:d}x{:d}.txt'.format(k, k)
    else:
        datastring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\PixelRep_{:d}x{:d}.npy'.format(k, k)
    datastring_extended = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\PixelRep_{:d}x{:d}_extended.npy'. \
        format(k, k)
    if k == 5:
        resultsstring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_' \
                        r'vs_oligomodal_vs_plurimodal_i_Scen_slope_modes_M1k_5x5_fixn4.txt'
    elif k > 6:
        resultsstring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_' \
                        r'vs_oligomodal_vs_plurimodal_i_Scen_slope_modes_M1k_{:d}x{:d}.txt'.format(
            k, k)
        resultsstring_extended = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_' \
                                 r'Scen_slope_modes_M1k_{:d}x{:d}_extended.txt'.format(
            k, k)
    else:
        resultsstring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_' \
                        r'vs_oligomodal_vs_plurimodal_i_Scen_slope_modes_M1k_{:d}x{:d}_fixn4.txt'.format(
            k, k)

    # data = np.loadtxt(datastring, delimiter=',')
    # results = np.loadtxt(resultsstring, delimiter=',')
    # np.save(r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\PixelRep_5x5.npy', data)
    # np.save(r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_i_Scen_slope_offset_M1k_5x5_fixn4.npy', results)
    if k < 5:
        data = np.loadtxt(datastring, delimiter=',')
    else:
        data = np.load(datastring)
    if extended:
        data = np.append(data, np.load(datastring_extended), axis=0)
    if k > 6:
        results = np.loadtxt(resultsstring, delimiter=',')
        if extended:
            results = np.append(results, np.loadtxt(resultsstring_extended, delimiter=','), axis=0)
    else:
        results = np.loadtxt(resultsstring, delimiter=',')
    data = np.reshape(data, (-1, 2 * k, 2 * k))
    x_total = data

    "select only unimodal vs oligomodal (drop plurimodal and rest)"
    ind_U = np.argwhere(results[:, 3].astype(int) == 1)
    ind_O = np.argwhere(results[:, 3].astype(int) > 1)
    ind_UO = np.append(ind_U, ind_O, axis=0)
    x_total = x_total[ind_UO[:, 0]]
    res_total = results[ind_UO[:, 0]]

    dataprek = np.load(u'.\\data_unimodal_vs_oligomodal_inc_stripmodes_train_trainraw_test_5x5.npz')
    indU = np.argwhere(dataprek['y_test'] == 0)
    indO = np.argwhere(dataprek['y_test'] == 1)
    indUO = np.append(indU, indO)
    x_test = dataprek['x_test'][indUO]
    y_test = dataprek['y_test'][indUO]

    inds_test = np.zeros(np.shape(x_test)[0], dtype=int)
    for i in range(len(inds_test)):
        inds_test[i] = np.argwhere(np.all(x_total[:, :, :] == x_test[i, 1:2*k+1, 1:2*k+1, 0], axis=(1, 2)))[0, 0]

    # acc_M_kfold = np.zeros((10, int(np.nanmax(res_total[inds_test, 3]))),
    #                            dtype=float)
    # acc_M_kavg = np.zeros((int(np.nanmax(res_total[inds_test, 3])), 2),
    #                           dtype=float)
    # for kfold in range(10):
    #     # start = i*n_lr
    #     outval = np.load(
    #         'D:\\data\\cnn_Unimodal_vs_Oligomodal_big_5x5\\testres_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}'
    #             '_kfold{:d}.npy'.
    #             format(nf1, nf2, nf3, nh1, lr, kfold))
    #     out = np.argmax(outval, axis=1)
    #     for m in range(1, int(np.nanmax(res_total[inds_test, 3]))+1):
    #         inds_m = np.argwhere(res_total[inds_test, 3] == m)
    #         acc_M_kfold[kfold, m-1] = np.sum(out[inds_m[:, 0]] == y_test[inds_m[:, 0]]) \
    #                                           / np.shape(inds_m)[0]
    # acc_M_kavg[:, 0] = np.nanmean(acc_M_kfold[:], axis=0)
    # acc_M_kavg[:, 1] = np.nanvar(acc_M_kfold[:], axis=0)
    #
    # np.save('.\\cnn_Unimodal_vs_Oligomodal_big_5x5\\acc_M_kfold.npy', acc_M_kfold)
    # np.save('.\\cnn_Unimodal_vs_Oligomodal_big_5x5\\acc_M_kavg.npy', acc_M_kavg)
    acc_M_kfold = np.load('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\acc_M_kfold.npy')
    acc_M_kavg = np.load('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\acc_M_kavg.npy')

    f, ax = plt.subplots()
    for m in range(int(np.nanmax(res_total[inds_test, 3]))):
        ax.plot(m, acc_M_kavg[m, 0], '.-')
        # ax.fill_between(m, acc_M_kavg[m, 0] - np.sqrt(acc_M_kavg[m, 1]), acc_M_kavg[m, 0] + np.sqrt(acc_M_kavg[m, 1]),
        #                 alpha=0.3)
    ax.set_xlabel('$M_k(\infty)$')
    ax.set_ylabel('accuracy')
    ax.set_title('accuracy satured modes')
    plt.legend()
    f.savefig('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\accuracy_saturated_modes_'
              'nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}.pdf'.format(nf1, nf2, nf3, nh1, lr),
              facecolor=f.get_facecolor())
    f.savefig('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\accuracy_saturated_modes_'
              'nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}.svg'.format(nf1, nf2, nf3, nh1, lr),
              facecolor=f.get_facecolor())
    f.savefig('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\accuracy_saturated_modes_'
              'nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_lr={:.4f}.png'.format(nf1, nf2, nf3, nh1, lr),
              facecolor=f.get_facecolor(),
              dpi=400)
    plt.show()
    plt.close()
    return 0


            # red = np.logical_and.reduce((ConMat[:, 0].astype(int) == nf, ConMat[:, 1].astype(int) == nh, ConMat[:, 2] == lr))
            # CM_arg = np.argwhere(red)
            # np.logical_and(np.logical_and(ConMat[:, 0].astype(int) ==nf, ConMat[:, 1].astype(int) == nh), ConMat[:, 2]==lr))

def test_random_walk_trueclass(k):
    savedir = u'.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_{:d}x{:d}\\random_walks\\'.format(k, k)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    # resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_Unimodal_vs_Oligomodal_5x5\\kfold_avg_val_results.txt'
    # KFres = np.loadtxt(resultstring, delimiter='\t')
    "KFres: [nf, nh, lr, val_acc_avg, val_acc_var, val_loss_avg, val_loss_var, val_prec_avg, val_prec_var, val_rec_avg, val_rec_var, val_f1_avg, val_f1_var]"
    "choose performance measure (PM) index"
    PM_index = 3

    Nsteps = k * k

    nf1 = 20
    nf2 = 80
    nf3 = 160
    nh1 = 1000
    lr = 0.0005

    tOpOtotOpU = np.zeros((10, Nsteps), dtype=int)
    tOpOtotOpO = np.zeros((10, Nsteps), dtype=int)
    tOpOtotUpU = np.zeros((10, Nsteps), dtype=int)
    tOpOtotUpO = np.zeros((10, Nsteps), dtype=int)

    tOpUtotOpU = np.zeros((10, Nsteps), dtype=int)
    tOpUtotOpO = np.zeros((10, Nsteps), dtype=int)
    tOpUtotUpU = np.zeros((10, Nsteps), dtype=int)
    tOpUtotUpO = np.zeros((10, Nsteps), dtype=int)

    tUpOtotOpU = np.zeros((10, Nsteps), dtype=int)
    tUpOtotOpO = np.zeros((10, Nsteps), dtype=int)
    tUpOtotUpU = np.zeros((10, Nsteps), dtype=int)
    tUpOtotUpO = np.zeros((10, Nsteps), dtype=int)

    tUpUtotOpU = np.zeros((10, Nsteps), dtype=int)
    tUpUtotOpO = np.zeros((10, Nsteps), dtype=int)
    tUpUtotUpU = np.zeros((10, Nsteps), dtype=int)
    tUpUtotUpO = np.zeros((10, Nsteps), dtype=int)

    batch_size = 5000
    for j, l in enumerate(range(0, 11208, batch_size)):
        for i in range(batch_size * j, batch_size * (j + 1)):
            if os.path.exists(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\configlist_test_{:d}.npy'.format(k, k, i)) \
                    and os.path.exists(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\lmlist_test_{:d}.npy'.format(k, k, i)):
                if i == batch_size * j:

                    configs = np.load(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\configlist_test_{:d}.npy'.format(k, k, i))
                    lmlist = np.load(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\lmlist_test_{:d}.npy'.format(k, k, i))
                    nmlist = np.load(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\nmlist_test_{:d}.npy'.format(k, k, i))
                else:
                    configs = np.append(configs, np.load(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\configlist_test_{:d}.npy'
                                                         .format(k, k, i)), axis=0)
                    lmlist = np.append(lmlist, np.load(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\lmlist_test_{:d}.npy'
                                                       .format(k, k, i)), axis=0)
                    nmlist = np.append(nmlist, np.load(u'D:\\data\\random_walks_Unimodal_vs_Oligomodal_inc_stripmodes_{:d}x{:d}\\nmlist_test_{:d}.npy'
                                                       .format(k, k, i)))
        print('configs loaded batch {:d}'.format(j))
        labels = nmlist - 3*lmlist
        if configs.size == 0:
            continue

        for kfold in range(10):
            filepath = u".\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\saves\\nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}_" \
                       u"lr={:.4f}\\kfold={:d}\\1\\". \
                format(int(nf1), int(nf2), int(nf3), int(nh1), float(lr), int(kfold))
            tf.keras.backend.clear_session()
            model = tf.keras.models.load_model(
                filepath, custom_objects=None, compile=True
            )
            model.summary()

            out = model.predict(configs[:, :, :, np.newaxis].astype(np.float32))
            y_pred = np.argmax(out, axis=1)
            # ind_P = np.argwhere(y_pred[1:] == 1)
            # ind_O = np.argwhere(y_pred[1:] == 0)
            for i in range(int(np.shape(labels)[0] / (Nsteps + 1))):
                ind_0 = i * (Nsteps + 1)
                ind_l = ind_0 + 1
                ind_r = ind_0 + 1 + Nsteps
                ind_tOpO = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 1, labels[ind_l:ind_r] > 1))
                ind_tOpU = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 0, labels[ind_l:ind_r] > 1))
                ind_tUpO = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 1, labels[ind_l:ind_r] == 1))
                ind_tUpU = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 0, labels[ind_l:ind_r] == 1))
                if y_pred[ind_0] == 1:
                    'predicted O'
                    if labels[ind_0] > 1:
                        "true label O"
                        tOpOtotOpO[kfold, ind_tOpO[:, 0]] += 1
                        tOpOtotOpU[kfold, ind_tOpU[:, 0]] += 1
                        tOpOtotUpO[kfold, ind_tUpO[:, 0]] += 1
                        tOpOtotUpU[kfold, ind_tUpU[:, 0]] += 1
                    else:
                        "true label U"
                        tUpOtotOpO[kfold, ind_tOpO[:, 0]] += 1
                        tUpOtotOpU[kfold, ind_tOpU[:, 0]] += 1
                        tUpOtotUpO[kfold, ind_tUpO[:, 0]] += 1
                        tUpOtotUpU[kfold, ind_tUpU[:, 0]] += 1
                else:
                    'predicted U'
                    if labels[ind_0] > 1:
                        "true label O"
                        tOpUtotOpO[kfold, ind_tOpO[:, 0]] += 1
                        tOpUtotOpU[kfold, ind_tOpU[:, 0]] += 1
                        tOpUtotUpO[kfold, ind_tUpO[:, 0]] += 1
                        tOpUtotUpU[kfold, ind_tUpU[:, 0]] += 1
                    else:
                        "true label U"
                        tUpUtotOpO[kfold, ind_tOpO[:, 0]] += 1
                        tUpUtotOpU[kfold, ind_tOpU[:, 0]] += 1
                        tUpUtotUpO[kfold, ind_tUpO[:, 0]] += 1
                        tUpUtotUpU[kfold, ind_tUpU[:, 0]] += 1

        del configs, labels
    np.savez(savedir + 'nf1{:d}_nf2{:d}_nf3{:d}_nh1{:d}_lr{:.4f}_kfolds_probability_classchange_randomwalk_'
                       '{:d}x{:d}_test_true_and_'
                       'predicted.npz'.format(int(nf1), int(nf2), int(nf3), int(nh1), float(lr), int(k), int(k)),
             tOpOtotOpO=tOpOtotOpO, tOpOtotOpU=tOpOtotOpU, tOpOtotUpO=tOpOtotUpO, tOpOtotUpU=tOpOtotUpU,
             tOpUtotOpO=tOpUtotOpO, tOpUtotOpU=tOpUtotOpU, tOpUtotUpO=tOpUtotUpO, tOpUtotUpU=tOpUtotUpU,
             tUpOtotOpO=tUpOtotOpO, tUpOtotOpU=tUpOtotOpU, tUpOtotUpO=tUpOtotUpO, tUpOtotUpU=tUpOtotUpU,
             tUpUtotOpO=tUpUtotOpO, tUpUtotOpU=tUpUtotOpU, tUpUtotUpO=tUpUtotUpO, tUpUtotUpU=tUpUtotUpU)
    return 0

def plot_CM_bestvalloss():
    f = plt.figure(1, figsize=(cm_to_inch(8.6), cm_to_inch(6.1)))
    xlength = 3.32
    ylength = 0.3
    # f = plt.figure(1, figsize=(cm_to_inch(4.45), cm_to_inch(4.15)))
    xoffset = 0.2 * (xlength / 8.6)
    yoffset = 0.2 * (xlength / 6.1)
    # xoffset = 0.25
    # yoffset =
    frac_pad_x = 0.07 * (2. / 3.)
    frac_pad_y = frac_pad_x * 8.6 / 6.1
    figfracx = (8.5 - xoffset * 8.6 - 2 * frac_pad_x * 8.6) / 8.6
    figfracy = figfracx * 8.6 / (6.1)

    PM_index = 1
    nf1 = 20
    nf2 = 80
    nf3 = 160
    nh1 = 1000
    lr = 0.0005

    k = 5

    resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_' \
                   u'5x5\\kfold_val_acc_loss_prec_rec_f1.npz'
    KFres = np.load(resultstring)['val_loss_list']
    bestkfold = np.nanargmin(KFres)

    ConMat = np.load(
        r'D:\\data\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\testset_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}'
        r'_lr={:.4f}_TP_FP_TN_FN.npy'.format(nf1, nf2, nf3, nh1, lr))
    "CM = [kfold, TP, FP, TN, FN]"
    CM = np.zeros((2, 2), dtype=int)
    CM[0, 0] = ConMat[bestkfold, 2]
    CM[1, 0] = ConMat[bestkfold, 1]
    CM[0, 1] = ConMat[bestkfold, 3]
    CM[1, 1] = ConMat[bestkfold, 4]

    vmax = np.amax(ConMat[bestkfold, :])
    colors = ['black', 'white']
    # ax = f.add_axes([xoffset + int((k - 3) / 2) * (frac_pad_x + figfracx / 3.),
    #                  yoffset + (k % 2) * (frac_pad_y + figfracy / 3.),
    #                  figfracx / 3., figfracy / 3.])
    ax = f.add_axes([xoffset, yoffset, 2* figfracx / 3, 2*figfracy / 3])
    ax.imshow(CM, cmap='Blues', vmin=0, vmax=vmax, origin='lower')
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    # ax.text(-0.44, -0.1, np.format_float_scientific(CM[0, 0], precision=2, unique=True, exp_digits=1, min_digits=2),
    #         c='white', fontsize=8)
    ax.text(-0, 0, '{:d}'.format(CM[0, 0]), c=colors[int(2*CM[0, 0]/(vmax+1))], fontsize=7, ha='center', va='center')
    ax.text(-0, 1, '{:d}'.format(CM[1, 0]), c=colors[int(2*CM[1, 0]/(vmax+1))], fontsize=7, ha='center', va='center')
    ax.text(1, -0, '{:d}'.format(CM[0, 1]), c=colors[int(2*CM[0, 1]/(vmax+1))], fontsize=7, ha='center', va='center')
    ax.text(1, 1, '{:d}'.format(CM[1, 1]), c=colors[int(2*CM[1, 1]/(vmax+1))], fontsize=7, ha='center', va='center')
    # ax.text(0.56, 0.9, np.format_float_scientific(CM[1, 1], precision=2, unique=True, exp_digits=1, min_digits=2),
    #         c='white', fontsize=8)
    ax.set_title(r'${:d} \times {:d}$'.format(k, k), fontsize=8, pad=0.01)
    ax.set_xlabel('predicted')

    ax.set_xticklabels(['O', 'U'])

    ax.set_ylabel('actual')

    ax.set_yticklabels(['U', 'O'])
    f.savefig('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\ConfusionMatrix_bestvalloss.pdf',
              facecolor=f.get_facecolor())
    f.savefig('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\ConfusionMatrix_bestvalloss.svg',
              facecolor=f.get_facecolor())
    f.savefig('.\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_5x5\\ConfusionMatrix_bestvalloss.png',
              facecolor=f.get_facecolor(), dpi=400)
    plt.show()
    plt.close()
    return 0


def main():
    print('hello world')
    # testset_output_allCNN()
    # CM_testset()
    # analyse_wrongclassification()
    # test_random_walk_trueclass(4)
    # plot_CM_bestvalloss()
    test_random_walk_trueclass(5)
    return 0

if __name__ == "__main__":
    main()