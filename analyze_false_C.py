"Ryan van Mastrigt, 29.07.2022"
"This script plots the unit cells falsely classified as C by a trained neural network"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

def false_C_StripMode_bestnet(k):
    "strip modes"
    dat = np.load('.\\testPM_nfnh_{:d}x{:d}.npy'.format(k, k))
    nhlist = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    nflist = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    PM_index = 3
    CMs = []
    if k < 4:
        CMs.append(np.load('D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\testset_nfnhlrkfold_TP_FP_TN_FN.npy'.
                           format(k, k)))
    elif k == 4:
        CMs.append(
            np.load('D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\testset_nfnhlrkfold_TP_FP_TN_FN.npy'.
                    format(k, k)))
    else:
        CMs.append(np.load('D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\testset_nfnhlrkfold_TP_FP_TN_FN.npy'.
                           format(k, k)))
    if k == 3:
        strings = [u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_3x3_AB_5050_2']
    elif k == 4:
        strings = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to9',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to13',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to17',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18']
    elif k == 5:
        strings = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_0to2',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_2to4',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_4to6',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_6to8',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_8to10',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_10to12',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_12to14',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_14to17',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_17to18'
        ]
    elif k == 6:
        strings = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_0to1',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_1to2',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_2to3',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_3to4',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_4to5',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_5to6',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_6to7',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_7to8',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_8to9',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_9to10',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_10to11',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_11to12',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_12to13',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_13to14',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_14to15',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_15to16',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_16to17',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_6x6_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_6x6_AB_5050_nh_17to18'
        ]
    elif k == 7:
        strings = [
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_0to1',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_1to2',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_2to3',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_3to4',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_4to5',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_5to6',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_6to7',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_7to8',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_8to9',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_9to10',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_10to11',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_11to12',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_12to13',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_13to14',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_14to15',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_15to16',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_16to17',
            u'E:\\data\\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_extended_7x7_AB_5050_nh_17to18'
        ]
    elif k == 8:
        strings = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_0to1',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_1to2',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_2to3',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_3to4',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_4to5',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_5to6',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_6to7',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_7to8',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_8to9',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_9to10',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_10to11',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_11to12',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_12to13',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_13to14',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_14to15',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_15to16',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_16to17',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_8x8_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_8x8_AB_5050_nh_17to18'
        ]
    for parts in range(len(strings)):
        resultstring = strings[parts]
        if parts == 0:
            KFres = np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t')
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'),
                              axis=0)
    arg = np.nanargmax(np.abs(KFres[:, PM_index]))
    nf = KFres[arg, 0].astype(int)
    nh = KFres[arg, 1].astype(int)
    lr = KFres[arg, 2]

    if k == 3:
        stringind = 0
        string = strings[stringind] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    elif k == 4:
        stringind = np.argwhere(nhlist == nh)[0, 0]
        string = strings[int(stringind / 4)] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    elif k == 5:
        stringind = np.argwhere(nhlist == nh)[0, 0]
        if stringind < 14:
            string = strings[int(stringind / 2)] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
        else:
            if stringind < 17:
                string = strings[-2] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
            else:
                string = strings[-1] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    elif k == 6:
        stringind = np.argwhere(nhlist == nh)[0, 0]
        string = strings[stringind] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    elif k == 7:
        stringind = np.argwhere(nhlist == nh)[0, 0]
        string = strings[stringind] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    elif k == 8:
        stringind = np.argwhere(nhlist == nh)[0, 0]
        string = strings[stringind] + u'\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    n_kfolds = 10
    res_kfolds = np.zeros((n_kfolds, 5))
    for kfold in range(10):
        res_kfolds[kfold] = np.loadtxt(string + u'kfold={:d}'.format(kfold) + u'\\results.txt', delimiter='\t')
        "results.txt: val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1"
    bestkfold = np.argmax(res_kfolds[:, 1])

    "test set results nf nh lr kfold"
    testres = np.load('D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.
        format(k, k, int(nf), int(nh), lr, int(bestkfold))
                      )
    dataprek = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\data_prek_xy_train_trainraw_test_{:d}x{:d}.npz'.
                   format(k, k, k, k))
    y_pred = np.argmax(testres, axis=1)
    indA = np.argwhere(dataprek['y_test'] == 0)
    indB = np.argwhere(dataprek['y_test'] == 1)
    indAB = np.append(indA, indB)
    x_test = dataprek['x_test'][indAB]
    y_test = dataprek['y_test'][indAB]
    ind_FC = np.argwhere(np.logical_and(y_pred==1, y_test==0))
    # nf_arg = np.argwhere(CMs[i][:, 0, 0, 0, 0] == nf)[0, 0]
    # nh_arg = np.argwhere(CMs[i][0, :, 0, 0, 1] == nh)[0, 0]
    # lr_arg = np.argwhere(CMs[i][0, 0, :, 0, 2] == lr)[0, 0]
    # fold_arg = np.argwhere(CMs[i][0, 0, 0, :, 3] == bestkfold)[0, 0]
    # "CM = [nf, nh, lr, kfold, TP, FP, TN, FN]"
    # CM = np.zeros((2, 2), dtype=int)
    # CM[0, 0] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 5]
    # CM[1, 0] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 4]
    # CM[0, 1] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 6]
    # CM[1, 1] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 7]
    fig, axes = plt.subplots(6, 5, figsize=(6.75, 6.75/6*5))
    with PdfPages(u'.\\StripMode_best_class_False_C_PixRep_{:d}x{:d}_connections.pdf'.format(k, k)) as pdf:
        for i, ind in enumerate(ind_FC[:, 0]):
            ax_j = i%5
            ax_i = int(i/5)
            f, ax = plt.subplots()
            ax.imshow(x_test[ind, 1:2*k+1, 1:2*k+1, 0], cmap='Greys', vmin=0, vmax=1)
            axes[ax_i, ax_j].imshow(x_test[ind, 1:2*k+1, 1:2*k+1, 0], cmap='Greys', vmin=0, vmax=1)
            for x in range(1, 2*k, 2):
                ax.axvline(x-0.5, color='grey')
                axes[ax_i, ax_j].axvline(x - 0.5, color='grey')
            for y in range(1, 2*k, 2):
                ax.axhline(y-0.5, color='grey')
                axes[ax_i, ax_j].axhline(y - 0.5, color='grey')

            drawmin = -0.5
            drawmax = 2*k - 0.5
            for x in range(k):
                for y in range(k):
                    "draw circles and lines for connected pairs"
                    circle1 = plt.Circle((2*x+0.5, 2*y+0.5), 0.1, color='tab:red', zorder=5)
                    circle2 = plt.Circle((2 * x + 0.5, 2 * y + 0.5), 0.1, color='tab:red', zorder=5)
                    # axes[ax_i, ax_j].add_patch(circle1)
                    ax.add_patch(circle1)
                    axes[ax_i, ax_j].add_patch(circle2)
                    if (np.sum(np.multiply(x_test[ind, 2*x:2*(x+1), 1+2*y:1+2*(y+1), 0], np.array([[1, 0], [1, 0]])))
                              == 2) or (np.sum(np.multiply(x_test[ind, 2*x:2*(x+1), 1+2*y:1+2*(y+1), 0],
                                                         np.array([[0, 1], [0, 1]]))) == 2):
                        "vertical connection"
                        ystart = 2*y + 0.5
                        xstart = 2*(x-1) + 0.5
                        xend = 2*x + 0.5
                        if xstart < drawmin:
                            "draw two lines"
                            ax.vlines(ystart, drawmin, xend, color='tab:orange', zorder=4, lw=2)
                            ax.vlines(ystart, xstart + 2*k, drawmax, color='tab:orange', zorder=4, lw=2)
                            axes[ax_i, ax_j].vlines(ystart, drawmin, xend, color='tab:orange', zorder=4, lw=2)
                            axes[ax_i, ax_j].vlines(ystart, xstart + 2 * k, drawmax, color='tab:orange', zorder=4, lw=2)
                        else:
                            # axes[ax_i, ax_j].hlines(y-0.5, x-0.5, x+2-0.5, color='tab:orange', width=0.02)
                            ax.vlines(ystart, xstart, xend, color='tab:orange', zorder=4, lw=2)
                            axes[ax_i, ax_j].vlines(ystart, xstart, xend, color='tab:orange', zorder=4, lw=2)
                    if np.sum(np.multiply(x_test[ind, 1+2*x:1+2*(x+1), 2*y:2*(y+1), 0], np.array([[1, 1], [0, 0]]))) == 2\
                            or np.sum(np.multiply(x_test[ind, 1+2*x:1+2*(x+1), 2*y:2*(y+1), 0], np.array([[0, 0], [1, 1]]))) == 2:
                        "horizontal connection"
                        # axes[ax_i, ax_j].vlines(x - 0.5, y - 0.5, y + 2 - 0.5, color='tab:orange', width=0.02)
                        xstart = 2*x + 0.5
                        ystart = 2*(y-1)+0.5
                        yend = 2*y + 0.5
                        if ystart < drawmin:
                            ax.hlines(xstart, drawmin, yend, color='tab:orange', zorder=4, lw=2)
                            ax.hlines(xstart, ystart + 2*k, drawmax, color='tab:orange', zorder=4, lw=2)
                            axes[ax_i, ax_j].hlines(xstart, drawmin, yend, color='tab:orange', zorder=4, lw=2)
                            axes[ax_i, ax_j].hlines(xstart, ystart + 2 * k, drawmax, color='tab:orange', zorder=4, lw=2)
                        else:
                            ax.hlines(xstart, ystart, yend, color='tab:orange', zorder=4, lw=2)
                            axes[ax_i, ax_j].hlines(xstart, ystart, yend, color='tab:orange', zorder=4, lw=2)
                        # ax.hlines(2*(x) + 0.5, 2*(y-1) + 0.5, 2*y + 0.5, color='tab:orange',zorder=4, lw=2)
                    if np.sum(np.multiply(x_test[ind, 2*x:2*(x+1), 2*y:2*(y+1), 0], np.array([[1, 0], [0, 1]]))) == 2:
                        "upper diagonal"
                        ystart = 2*(x-1)+0.5
                        yend = 2*x+0.5
                        xstart = 2*(y-1)+0.5
                        xend = 2*y+0.5
                        if xstart < drawmin:
                            if ystart < drawmin:
                                line = Line2D([drawmin, xend], [drawmin, yend], color='tab:orange', zorder=4, lw=2)
                                line2 = Line2D([xstart + 2*k, drawmax], [ystart+2*k, drawmax], color='tab:orange', zorder=4, lw=2)
                                line3 = Line2D([drawmin, xend], [drawmin, yend], color='tab:orange', zorder=4, lw=2)
                                line4 = Line2D([xstart + 2 * k, drawmax], [ystart + 2 * k, drawmax], color='tab:orange',
                                               zorder=4, lw=2)
                            else:
                                line = Line2D([drawmin, xend], [ystart+1, yend], color='tab:orange', zorder=4, lw=2)
                                line2 = Line2D([xstart + 2 * k, drawmax], [ystart, yend-1], color='tab:orange',
                                               zorder=4, lw=2)
                                line3 = Line2D([drawmin, xend], [ystart + 1, yend], color='tab:orange', zorder=4, lw=2)
                                line4 = Line2D([xstart + 2 * k, drawmax], [ystart, yend - 1], color='tab:orange',
                                               zorder=4, lw=2)
                            ax.add_line(line)
                            ax.add_line(line2)
                            axes[ax_i, ax_j].add_line(line3)
                            axes[ax_i, ax_j].add_line(line4)
                        elif ystart < drawmin:
                            line = Line2D([xstart+1, xend], [drawmin, yend], color='tab:orange', zorder=4, lw=2)
                            line2 = Line2D([xstart, xend-1], [ystart+2*k, drawmax], color='tab:orange',
                                           zorder=4, lw=2)
                            line3 = Line2D([xstart + 1, xend], [drawmin, yend], color='tab:orange', zorder=4, lw=2)
                            line4 = Line2D([xstart, xend - 1], [ystart + 2 * k, drawmax], color='tab:orange',
                                           zorder=4, lw=2)
                            ax.add_line(line)
                            ax.add_line(line2)
                            axes[ax_i, ax_j].add_line(line3)
                            axes[ax_i, ax_j].add_line(line4)
                        else:
                            line = Line2D([xstart, xend], [ystart, yend], color='tab:orange', zorder=4, lw=2)
                            line2 = Line2D([xstart, xend], [ystart, yend], color='tab:orange', zorder=4, lw=2)
                            # line = Line2D([2*(x-1)+0.5, 2*x+0.5], [2*(y-1)+0.5, 2*y+0.5], color='tab:orange', zorder=4, lw=2)
                            ax.add_line(line)
                            axes[ax_i, ax_j].add_line(line2)
                        # axes[ax_i, ax_j].add_line(line)
                        # ax.add_line(line)
                    if np.sum(np.multiply(x_test[ind, 2 * x:2 * (x + 1), 2 * y:2 * (y + 1), 0],
                                          np.array([[0, 1], [1, 0]]))) == 2:
                        "lower diagonal"
                        yend = 2 * (x-1) + 0.5
                        ystart = 2 * (x) + 0.5
                        xend = 2 * (y) + 0.5
                        xstart = 2 * (y-1) + 0.5
                        if xstart < drawmin:
                            if yend < drawmin:
                                line = Line2D([xstart+2*k, drawmax], [ystart, drawmin], color='tab:orange', zorder=4, lw=2)
                                line2 = Line2D([drawmin, xend], [drawmax, yend+2*k], color='tab:orange',
                                               zorder=4, lw=2)
                                line3 = Line2D([xstart + 2 * k, drawmax], [ystart, drawmin], color='tab:orange',
                                              zorder=4, lw=2)
                                line4 = Line2D([drawmin, xend], [drawmax, yend + 2 * k], color='tab:orange',
                                               zorder=4, lw=2)
                            else:
                                line = Line2D([xstart+2*k, drawmax], [ystart, ystart-1], color='tab:orange', zorder=4, lw=2)
                                line2 = Line2D([drawmin, xend], [ystart-1, yend], color='tab:orange',
                                               zorder=4, lw=2)
                                line3 = Line2D([xstart + 2 * k, drawmax], [ystart, ystart - 1], color='tab:orange',
                                              zorder=4, lw=2)
                                line4 = Line2D([drawmin, xend], [ystart - 1, yend], color='tab:orange',
                                               zorder=4, lw=2)
                            ax.add_line(line)
                            ax.add_line(line2)
                            axes[ax_i, ax_j].add_line(line3)
                            axes[ax_i, ax_j].add_line(line4)
                        elif yend < drawmin:
                            line = Line2D([xstart, xstart+1], [ystart, drawmin], color='tab:orange', zorder=4, lw=2)
                            line2 = Line2D([xstart+1, xend], [drawmax, yend+2*k], color='tab:orange',
                                           zorder=4, lw=2)
                            line3 = Line2D([xstart, xstart + 1], [ystart, drawmin], color='tab:orange', zorder=4, lw=2)
                            line4 = Line2D([xstart + 1, xend], [drawmax, yend + 2 * k], color='tab:orange',
                                           zorder=4, lw=2)
                            ax.add_line(line)
                            ax.add_line(line2)
                            axes[ax_i, ax_j].add_line(line3)
                            axes[ax_i, ax_j].add_line(line4)
                        else:
                            line = Line2D([xstart, xend], [ystart, yend],
                                          color='tab:orange', zorder=4, lw=2)
                            line2 = Line2D([xstart, xend], [ystart, yend],
                                          color='tab:orange', zorder=4, lw=2)
                            ax.add_line(line)
                            axes[ax_i, ax_j].add_line(line2)
                        # line = Line2D([2*(x-1) + 0.5, 2*x + 0.5], [2*y + 0.5, 2*(y-1) + 0.5], color='tab:orange', zorder=4, lw=2)
                        # # axes[ax_i, ax_j].add_line(line)
                        # ax.add_line(line)
            # plt.show()

            pdf.savefig(f)

            plt.close(f)
            axes[ax_i, ax_j].axes.xaxis.set_visible(False)
            axes[ax_i, ax_j].axes.yaxis.set_visible(False)
    # axes[-1, -1].axes.xaxis.set_visible(False)
    # axes[-1, -1].axes.yaxis.set_visible(False)
    axes[-1, -1].axis('off')
    fig.savefig(u'.\\figures\\Stripmode_best_class_FalseC_PixRep_{:d}x{:d}_gridfig_connections.svg'.format(k, k),
                facecolor=fig.get_facecolor())
    fig.savefig(u'.\\figures\\Stripmode_best_class_FalseC_PixRep_{:d}x{:d}_gridfig_connections.pdf'.format(k, k),
                facecolor=fig.get_facecolor())
    fig.savefig(u'.\\figures\\Stripmode_best_class_FalseC_PixRep_{:d}x{:d}_gridfig_connections.png'.format(k, k),
                facecolor=fig.get_facecolor(),
                dpi=400)
    plt.show()

    # d = pdf.infodict()
    # d['Title'] = 'Class C Pixel Representation and Mode Scaling'
    # d['Author'] = 'Ryan van Mastrigt'
    return 0

false_C_StripMode_bestnet(5)