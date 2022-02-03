import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import matplotlib.backends.backend_pdf
import os
import itertools
import sys
import pickle
import colorcet as cc
from scipy.optimize import curve_fit

sys.path.append(os.path.abspath('../MetaCombi'))
# from r'C:\\Users\\ryanv\\PycharmProjects\\MetaCombi\\PCA_modescaling.py' import Martin_rep
from PCA_modescaling import Martin_rep

def determine_precision(classifier_pred, true_pred):
    "calculate precision=number of correct pos classified/total pos classified"
    n_true_pos_class = classifier_pred + true_pred ==2
    n_true_pos_class = np.sum(n_true_pos_class)
    n_pos_class = np.sum(classifier_pred)
    return n_true_pos_class/n_pos_class

def determine_recall(classifier_pred, true_pred):
    "calculate recall = number of correct pos classified/ number of true pos"
    n_true_pos_class = classifier_pred + true_pred ==2
    n_true_pos_class = np.sum(n_true_pos_class)
    n_true_pos = np.sum(true_pred)
    return n_true_pos_class/n_true_pos

def MCC_score(classifier_pred, true_pred):
    n_true_pos = np.sum(np.logical_and(classifier_pred==1, true_pred==1), dtype="float64")
    n_false_pos = np.sum(np.logical_and(classifier_pred==1, true_pred==0), dtype="float64")
    n_true_neg = np.sum(np.logical_and(classifier_pred==0, true_pred==0), dtype="float64")
    n_false_neg = np.sum(np.logical_and(classifier_pred==0, true_pred==1), dtype="float64")
    MCC = n_true_pos*n_true_neg - n_false_pos*n_false_neg
    MCC /= np.sqrt((n_true_pos+n_false_pos)*(n_true_pos+n_false_neg)*(n_true_neg+n_false_pos)*(n_true_neg+n_false_neg))
    return MCC

def f1_score(precision, recall):
    return 2.*(precision*recall)/(precision+recall)

# def density_AnB_NN(out, ind_A, ind_B, title, savestring, pdfsave=False):
#     AminB = out[:, 0] - out[:, 1]
#     bins=50
#     histA, binedgesA = np.histogram(AminB[ind_A], bins=bins)
#     histB, binedgesB = np.histogram(AminB[ind_B], bins=bins)
#     histA = histA/np.shape(AminB[ind_A])[0]
#     histB = histB/np.shape(AminB[ind_B])[0]
#     binsizeA = binedgesA[1]-binedgesA[0]
#     binsizeB = binedgesB[1] - binedgesB[0]
#     xA = np.arange(binedgesA[0]+binsizeA/2., binedgesA[-1], binsizeA)
#     xB = np.arange(binedgesB[0]+binsizeB/2., binedgesB[-1], binsizeB)
#     fig, ax = plt.subplots()
#     ax.plot(xA, histA, 'b.--', label='A')
#     ax.plot(xB, histB, 'g.--', label='B')
#     ax.set_ylabel('pdf')
#     ax.set_xlabel('A-B')
#     ax.legend()
#     # ax.axvline(c='r')
#     ax.axvspan(np.amin(xB)-40, 0, facecolor='green', alpha=0.15)
#     ax.axvspan(0, np.amax(xA)+40, facecolor='blue', alpha=0.15)
#     ax.set_xlim([np.amin(xB)-10, np.amax(xA)+10])
#     ax.set_title(title)
#     plt.tight_layout()
#     if pdfsave:
#         plt.savefig(savestring+'.pdf')
#         plt.savefig(savestring+'.png', dpi=400)
#     plt.show()
#     plt.close()

def plot_confusion_matrix(cm, class_names, savestr, savecm):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')

    # Normalize the confusion matrix.
    cm_rownorm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=4)
    cm_colnorm = np.around(cm.astype('float') / cm.sum(axis=0)[np.newaxis, :], decimals=4)
    figure = plt.figure()
    plt.imshow(cm_rownorm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(r"$4 \times 4$")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)


    # Use white text if squares are dark; otherwise black.
    threshold = 0.5
    for i, j in itertools.product(range(cm_rownorm.shape[0]), range(cm_rownorm.shape[1])):
        color = "white" if cm_rownorm[i, j] > threshold else "black"
        plt.text(j, i, cm_rownorm[i, j], horizontalalignment="center", color=color, size='xx-large')

    #plt.tight_layout()
    plt.colorbar()
    plt.xlim(-0.5, cm_rownorm.shape[1]-0.5)
    plt.ylim(-0.5, cm_rownorm.shape[0]-0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savecm:
        plt.savefig(savestr+'_rownorm.pdf', transparent=True)
        plt.savefig(savestr+'_rownorm.png', transparent=True, dpi=400)
    plt.show()
    plt.close()

    figure = plt.figure()
    plt.imshow(cm_colnorm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title(r"$4 \times 4$")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)

    # Use white text if squares are dark; otherwise black.
    threshold = 0.5
    for i, j in itertools.product(range(cm_colnorm.shape[0]), range(cm_colnorm.shape[1])):
        color = "white" if cm_colnorm[i, j] > threshold else "black"
        plt.text(j, i, cm_colnorm[i, j], horizontalalignment="center", color=color, size='xx-large')

    # plt.tight_layout()
    plt.colorbar()
    plt.xlim(-0.5, cm_colnorm.shape[1] - 0.5)
    plt.ylim(-0.5, cm_colnorm.shape[0] - 0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if savecm:
        plt.savefig(savestr + '_colnorm.pdf', transparent=True)
        plt.savefig(savestr + '_colnorm.png', transparent=True, dpi=400)
    plt.show()
    plt.close()
    return

def F1score_heatmap(f1matrix, xticks, yticks, xlabel, ylabel, pdfsave=False, savestring= u'.\\', set_vmin=False, vmin=0,
                    cblabel='$F_1$ score', transpose=False, cbar=True, logc=False):
    "create a 2d heatmap of the f1 score"
    "input: 2d matrix: f1matrix[x, y], strings xlabel & ylabel"
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')
    # if transpose:
    "make figures a third of PRL single column"
    matplotlib.rcParams['figure.figsize'] = 3.375/3., 3.375/3.
    aspect = np.shape(f1matrix)[1] / np.shape(f1matrix)[0]
    f, ax = plt.subplots()
    if set_vmin:
        if logc:
            im = ax.imshow(f1matrix, origin='lower', cmap=cc.cm["linear_kbc_5_95_c73"],
                           aspect=aspect, norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=0.5))
        else:
            im = ax.imshow(f1matrix, origin='lower', cmap=cc.cm["diverging_bwr_20_95_c54"], vmin=vmin, vmax=1, aspect=aspect)
    else:
        im = ax.imshow(f1matrix, origin='lower', cmap=cc.cm["diverging_bwr_20_95_c54"], vmax=1, aspect=aspect)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(np.arange(0, np.shape(f1matrix)[1], 1))
    ax.set_yticks(np.arange(0, np.shape(f1matrix)[0], 1))
    ax.set_xticklabels(xticks)
    ax.set_yticklabels(yticks)
    if transpose:
        ax.set_yticks(np.arange(0, np.shape(f1matrix)[0], 5))
        ax.set_xticks(np.arange(0, np.shape(f1matrix)[1], 4))
    else:
        ax.set_xticks(np.arange(0, np.shape(f1matrix)[1], 5))
        ax.set_yticks(np.arange(0, np.shape(f1matrix)[0], 4))
    if cbar:
        cb = plt.colorbar(im, ax=ax)
        cb.ax.set_xlabel(cblabel)
        cb.ax.xaxis.set_label_position('top')
    plt.tight_layout()
    if pdfsave:
        plt.savefig(savestring+'.pdf', transparent=True)
        plt.savefig(savestring+'.png', dpi=400, transparent=True)
        plt.savefig(savestring + '.svg', transparent=True)
    plt.show()
    plt.close()

def nh_to_string(nh):
    if nh>0 and nh<11:
        string = 0
        return string
    if nh>11 and nh<19:
        string = 1
        return string
    if nh>19 and nh<51:
        string = 2
        return string
    if nh>51 and nh< 91:
        string = 3
        return string
    return -1

def validationset_output_allCNN():
    strings = [
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to2',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to4',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to6',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to8',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to10',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to12',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to14',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to17'
    ]
    "load in pre-k train and test data"
    dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    x_rest = dataprek['x_rest']
    y_rest = dataprek['y_rest']
    nf_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90])
    lr_list = np.array([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005])
    N_nf = len(nf_list)
    N_nh = len(nh_list)
    N_lr = len(lr_list)
    N_kfolds = 10

    # ftxt.write("MCC val set \t nf \t nh \t lr \t kfold \n")
    for i in range(N_nf*N_nh*N_lr):
        nf = nf_list[i%N_nf]
        nh = nh_list[int(i/N_nf)%N_nh]
        lr = lr_list[int(int(i/N_nf)/N_nh)%N_lr]
        for kfold in range(N_kfolds):
            filepath = u"D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
            if os.path.isfile(filepath + 'variables\\variables.data-00001-of-00002') and os.path.isfile(
                    filepath + 'variables\\variables.data-00000-of-00002'):
                model = tf.keras.models.load_model(
                    filepath, custom_objects=None, compile=False
                )
                model.summary()
                # np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index,
                #          val_index=val_index)
                ind_trainval = np.load(
                    u'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\kfold_data_indices.npz'.format(
                        nf, nh, lr, kfold))
                ind_val = ind_trainval['val_index']
                outval = model.predict(x_rest[ind_val])
                np.save('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\valres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold), outval)
                tf.keras.backend.clear_session()
    return 0

def testset_output_allCNN():
    strings = [
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to2',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to4',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to6',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to8',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to10',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to12',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to14',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to17'
    ]
    "load in pre-k train and test data"
    dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    indA = np.argwhere(dataprek['y_test'] == 0)
    indB = np.argwhere(dataprek['y_test'] == 1)
    indAB = np.append(indA, indB)
    x_test = dataprek['x_test'][indAB]
    y_test = dataprek['y_test'][indAB]
    nf_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    lr_list = np.array([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005])
    N_nf = len(nf_list)
    N_nh = len(nh_list)
    N_lr = len(lr_list)
    N_kfolds = 10

    # ftxt.write("MCC val set \t nf \t nh \t lr \t kfold \n")
    for i in range(N_nf*N_nh*N_lr):
        nf = nf_list[i%N_nf]
        nh = nh_list[int(i/N_nf)%N_nh]
        lr = lr_list[int(int(i/N_nf)/N_nh)%N_lr]
        for kfold in range(N_kfolds):
            if os.path.isfile(r'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}'
                              r'.npy'.format(nf, nh, lr, kfold)):
                print('skip nf{:d} nh{:d} lr{:.4f} kfold{:d}'.format(nf, nh, lr, kfold))
                continue
            filepath = u"D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
            if (os.path.isfile(filepath + 'variables\\variables.data-00001-of-00002') and os.path.isfile(
                    filepath + 'variables\\variables.data-00000-of-00002')) or \
                    os.path.isfile(filepath+'variables\\variables.data-00000-of-00001'):
                model = tf.keras.models.load_model(
                    filepath, custom_objects=None, compile=False
                )
                model.summary()
                # np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index,
                #          val_index=val_index)
                # ind_trainval = np.load(
                #     u'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\kfold_data_indices.npz'.format(
                #         nf, nh, lr, kfold))
                # ind_val = ind_trainval['val_index']
                outval = model.predict(x_test)
                np.save('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold), outval)
                tf.keras.backend.clear_session()
                print('saved nf{:d} nh{:d} lr{:.4f} kfold{:d}'.format(nf, nh, lr, kfold))
            else:
                print('{:d} \t {:d} \t {:.4f} \t {:d}'.format(nf, nh, lr, kfold))
                print('cant load')
    return 0

def testset_output_allCNN_MoreFilters():
    strings = [
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to1_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_1to2_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to3_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_3to4_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to5_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to6_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to7_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_7to8_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to9_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to10_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to11_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_11to12_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to13_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to14_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to15_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_15to16_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_16to17_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18_MoreFilters'
    ]
    "load in pre-k train and test data"
    dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    indA = np.argwhere(dataprek['y_test'] == 0)
    indB = np.argwhere(dataprek['y_test'] == 1)
    indAB = np.append(indA, indB)
    x_test = dataprek['x_test'][indAB]
    y_test = dataprek['y_test'][indAB]
    nf_list = np.array([22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    lr_list = np.array([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005])
    N_nf = len(nf_list)
    N_nh = len(nh_list)
    N_lr = len(lr_list)
    N_kfolds = 10

    # ftxt.write("MCC val set \t nf \t nh \t lr \t kfold \n")
    for i in range(N_nf*N_nh*N_lr):
        nf = nf_list[i%N_nf]
        nh = nh_list[int(i/N_nf)%N_nh]
        lr = lr_list[int(int(i/N_nf)/N_nh)%N_lr]
        stringind = int(i/N_nf)%N_nh
        for kfold in range(N_kfolds):
            filepath = strings[stringind] + u"\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
            if os.path.isfile(filepath + 'variables\\variables.data-00000-of-00001'):
                if os.path.isfile('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold)):
                    print('{:d} \t {:d} \t {:.4f} \t {:d}'.format(nf, nh, lr, kfold))
                    print('testres exist already')
                else:
                    model = tf.keras.models.load_model(
                        filepath, custom_objects=None, compile=False
                    )
                    model.summary()
                    # np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index,
                    #          val_index=val_index)
                    # ind_trainval = np.load(
                    #     u'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\kfold_data_indices.npz'.format(
                    #         nf, nh, lr, kfold))
                    # ind_val = ind_trainval['val_index']
                    outval = model.predict(x_test)
                    np.save('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold), outval)
                    tf.keras.backend.clear_session()
            else:
                print('{:d} \t {:d} \t {:.4f} \t {:d}'.format(nf, nh, lr, kfold))
                print('cant load')
    return 0

def testset_output_allCNN_MoreHidden():
    strings = [
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to1_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_1to2_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to3_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_3to4_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to5_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to6_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to7_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_7to8_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to9_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to10_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to11_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_11to12_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to13_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to14_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to15_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_15to16_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_16to17_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18_MoreHidden'
    ]
    "load in pre-k train and test data"
    dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    indA = np.argwhere(dataprek['y_test'] == 0)
    indB = np.argwhere(dataprek['y_test'] == 1)
    indAB = np.append(indA, indB)
    x_test = dataprek['x_test'][indAB]
    y_test = dataprek['y_test'][indAB]
    nf_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    nh_list = np.array([110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])
    lr_list = np.array([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005])
    N_nf = len(nf_list)
    N_nh = len(nh_list)
    N_lr = len(lr_list)
    N_kfolds = 10

    # ftxt.write("MCC val set \t nf \t nh \t lr \t kfold \n")
    for i in range(N_nf*N_nh*N_lr):
        nf = nf_list[i%N_nf]
        nh = nh_list[int(i/N_nf)%N_nh]
        lr = lr_list[int(int(i/N_nf)/N_nh)%N_lr]
        stringind = int(i/N_nf)%N_nh
        for kfold in range(N_kfolds):
            filepath = strings[stringind] + u"\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
            if os.path.isfile(filepath + 'variables\\variables.data-00000-of-00001'):
                if os.path.isfile('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold)):
                    print('{:d} \t {:d} \t {:.4f} \t {:d}'.format(nf, nh, lr, kfold))
                    print('testres exist already')
                else:
                    model = tf.keras.models.load_model(
                        filepath, custom_objects=None, compile=False
                    )
                    model.summary()
                    # np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index,
                    #          val_index=val_index)
                    # ind_trainval = np.load(
                    #     u'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\kfold_data_indices.npz'.format(
                    #         nf, nh, lr, kfold))
                    # ind_val = ind_trainval['val_index']
                    outval = model.predict(x_test)
                    np.save('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold), outval)
                    tf.keras.backend.clear_session()
            else:
                print('{:d} \t {:d} \t {:.4f} \t {:d}'.format(nf, nh, lr, kfold))
                print('cant load')
    return 0

def testset_output_allCNN_MoreFiltersHidden():
    strings = [
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to1_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_1to2_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to3_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_3to4_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to5_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to6_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to7_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_7to8_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to9_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to10_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to11_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_11to12_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to13_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to14_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to15_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_15to16_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_16to17_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18_MoreFiltersHidden'
    ]
    "load in pre-k train and test data"
    dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    indA = np.argwhere(dataprek['y_test'] == 0)
    indB = np.argwhere(dataprek['y_test'] == 1)
    indAB = np.append(indA, indB)
    x_test = dataprek['x_test'][indAB]
    y_test = dataprek['y_test'][indAB]
    nf_list = np.array([22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    nh_list = np.array([110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])
    lr_list = np.array([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005])
    N_nf = len(nf_list)
    N_nh = len(nh_list)
    N_lr = len(lr_list)
    N_kfolds = 10

    # ftxt.write("MCC val set \t nf \t nh \t lr \t kfold \n")
    for i in range(N_nf*N_nh*N_lr):
        nf = nf_list[i%N_nf]
        nh = nh_list[int(i/N_nf)%N_nh]
        lr = lr_list[int(int(i/N_nf)/N_nh)%N_lr]
        stringind = int(i/N_nf)%N_nh
        for kfold in range(N_kfolds):
            filepath = strings[stringind] + u"\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
            if os.path.isfile(filepath + 'variables\\variables.data-00000-of-00001'):
                if os.path.isfile('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold)):
                    print('{:d} \t {:d} \t {:.4f} \t {:d}'.format(nf, nh, lr, kfold))
                    print('testres exist already')
                else:
                    model = tf.keras.models.load_model(
                        filepath, custom_objects=None, compile=False
                    )
                    model.summary()
                    # np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index,
                    #          val_index=val_index)
                    # ind_trainval = np.load(
                    #     u'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\kfold_data_indices.npz'.format(
                    #         nf, nh, lr, kfold))
                    # ind_val = ind_trainval['val_index']
                    outval = model.predict(x_test)
                    np.save('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold), outval)
                    tf.keras.backend.clear_session()
            else:
                print('{:d} \t {:d} \t {:.4f} \t {:d}'.format(nf, nh, lr, kfold))
                print('cant load')
    return 0

def CM_testset():
    dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    indA = np.argwhere(dataprek['y_test'] == 0)
    indB = np.argwhere(dataprek['y_test'] == 1)
    indAB = np.append(indA, indB)
    x_test = dataprek['x_test'][indAB]
    y_test = dataprek['y_test'][indAB]

    nf_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    lr_list = np.array([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005])
    CM_mat = np.zeros((len(nf_list), len(nh_list), len(lr_list), 10, 8))
    for i, nf in enumerate(nf_list):
        for j, nh in enumerate(nh_list):
            for l, lr in enumerate(lr_list):
                for kfold in range(10):
                    if os.path.isfile('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold)):
                        outval = np.load('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold))
                        outclass = np.argmax(outval, axis=1)
                        TP = np.sum(np.logical_and(y_test==1, outclass==1))
                        FP = np.sum(np.logical_and(y_test==0, outclass==1))
                        TN = np.sum(np.logical_and(y_test==0, outclass==0))
                        FN = np.sum(np.logical_and(y_test==1, outclass==0))

                        CM_mat[i, j, l, kfold, 0] = nf
                        CM_mat[i, j, l, kfold, 1] = nh
                        CM_mat[i, j, l, kfold, 2] = lr
                        CM_mat[i, j, l, kfold, 3] = kfold

                        CM_mat[i, j, l, kfold, 4] = TP
                        CM_mat[i, j, l, kfold, 5] = FP
                        CM_mat[i, j, l, kfold, 6] = TN
                        CM_mat[i, j, l, kfold, 7] = FN
                    else:
                        CM_mat[i, j, l, kfold, 0] = nf
                        CM_mat[i, j, l, kfold, 1] = nh
                        CM_mat[i, j, l, kfold, 2] = lr
                        CM_mat[i, j, l, kfold, 3] = kfold

                        CM_mat[i, j, l, kfold, 4] = np.NaN
                        CM_mat[i, j, l, kfold, 5] = np.NaN
                        CM_mat[i, j, l, kfold, 6] = np.NaN
                        CM_mat[i, j, l, kfold, 7] = np.NaN

    np.save('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_nfnhlrkfold_TP_FP_TN_FN.npy', CM_mat)
    return 0

def CM_testset_expanded():
    dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    indA = np.argwhere(dataprek['y_test'] == 0)
    indB = np.argwhere(dataprek['y_test'] == 1)
    indAB = np.append(indA, indB)
    x_test = dataprek['x_test'][indAB]
    y_test = dataprek['y_test'][indAB]

    nf_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160,
                        170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])
    lr_list = np.array([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005])
    CM_mat = np.zeros((len(nf_list), len(nh_list), len(lr_list), 10, 8))
    for i, nf in enumerate(nf_list):
        for j, nh in enumerate(nh_list):
            for l, lr in enumerate(lr_list):
                for kfold in range(10):
                    if os.path.isfile('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold)):
                        outval = np.load('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testres_nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}.npy'.format(nf, nh, lr, kfold))
                        outclass = np.argmax(outval, axis=1)
                        TP = np.sum(np.logical_and(y_test==1, outclass==1))
                        FP = np.sum(np.logical_and(y_test==0, outclass==1))
                        TN = np.sum(np.logical_and(y_test==0, outclass==0))
                        FN = np.sum(np.logical_and(y_test==1, outclass==0))

                        CM_mat[i, j, l, kfold, 0] = nf
                        CM_mat[i, j, l, kfold, 1] = nh
                        CM_mat[i, j, l, kfold, 2] = lr
                        CM_mat[i, j, l, kfold, 3] = kfold

                        CM_mat[i, j, l, kfold, 4] = TP
                        CM_mat[i, j, l, kfold, 5] = FP
                        CM_mat[i, j, l, kfold, 6] = TN
                        CM_mat[i, j, l, kfold, 7] = FN
                    else:
                        CM_mat[i, j, l, kfold, 0] = nf
                        CM_mat[i, j, l, kfold, 1] = nh
                        CM_mat[i, j, l, kfold, 2] = lr
                        CM_mat[i, j, l, kfold, 3] = kfold

                        CM_mat[i, j, l, kfold, 4] = np.NaN
                        CM_mat[i, j, l, kfold, 5] = np.NaN
                        CM_mat[i, j, l, kfold, 6] = np.NaN
                        CM_mat[i, j, l, kfold, 7] = np.NaN

    np.save('D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_expanded_nfnhlrkfold_TP_FP_TN_FN.npy', CM_mat)
    return 0

def foldavg_heatmap_valacc_and_testCM():
    # resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\kfold_avg_val_results.txt'
    k = 4
    strings = [
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to9',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to13',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to17']
    for parts in range(len(strings)):
        resultstring = strings[parts]
        if parts == 0:
            KFres = KFres = np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t')
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'), axis=0)
    # KFres = np.loadtxt(resultstring, delimiter='\t')
    # ConMat = np.loadtxt(r'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_TP_FP_TN_FN_lm.txt', delimiter=',')
    ConMat = np.load(r'D:\\data\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_nfnhlrkfold_TP_FP_TN_FN.npy')
    "load in TP, FP, TN, FN over test set. ConMat=[nf, nh, lr, kfold, TP, FP, TN, FN]"
    "KFres: [nf, nh, lr, val_acc_avg, val_acc_var, val_loss_avg, val_loss_var, val_prec_avg, val_prec_var, val_rec_avg, val_rec_var, val_f1_avg, val_f1_var]"
    "choose performance measure (PM) index"
    PM_index = 3
    nf_list = np.unique(KFres[:, 0])
    nh_list = np.unique(KFres[:, 1])
    PM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 2))
    "CM_nfnh: [nf, nh, {TP, FP, TN, FN}, {avg, var}]"
    CM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 4, 2))
    n_lr = 6
    for i in range(np.shape(nf_list)[0]):
        for j in range(np.shape(nh_list)[0]):
            # start = i*n_lr
            nfnhargs = np.argwhere(np.logical_and(KFres[:, 0] == nf_list[i], KFres[:, 1] == nh_list[j]))
            arg = np.nanargmax(np.abs(KFres[nfnhargs, PM_index]))
            nf = KFres[nfnhargs[arg, 0], 0].astype(int)
            nh = KFres[nfnhargs[arg, 0], 1].astype(int)
            lr = KFres[nfnhargs[arg, 0], 2]

            nf_arg = np.argwhere(ConMat[:, 0, 0, 0, 0] == nf)[0, 0]
            nh_arg = np.argwhere(ConMat[0, :, 0, 0, 1] == nh)[0, 0]
            lr_arg = np.argwhere(ConMat[0, 0, :, 0, 2] == lr)[0, 0]

            # red = np.logical_and.reduce((ConMat[:, 0].astype(int) == nf, ConMat[:, 1].astype(int) == nh, ConMat[:, 2] == lr))
            # CM_arg = np.argwhere(red)
                # np.logical_and(np.logical_and(ConMat[:, 0].astype(int) ==nf, ConMat[:, 1].astype(int) == nh), ConMat[:, 2]==lr))

            PM_nfnh[i, j, 0] = KFres[nfnhargs[arg, 0], PM_index]
            PM_nfnh[i, j, 1] = KFres[nfnhargs[arg, 0], PM_index+1]

            CM_nfnh[i, j, 0, 0] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 4])
            CM_nfnh[i, j, 0, 1] = np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 4])
            CM_nfnh[i, j, 1, 0], CM_nfnh[i, j, 1, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 5]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 5])
            CM_nfnh[i, j, 2, 0], CM_nfnh[i, j, 2, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 6]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 6])
            CM_nfnh[i, j, 3, 0], CM_nfnh[i, j, 3, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 7]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 7])







    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap', set_vmin=False, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True, savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\F1score_heatmap_vmin0', set_vmin=True, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_vmin0_5', set_vmin=True, vmin=0.5, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), 'number of hidden neurons', 'number of filters', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_vmin0_5_verbose', set_vmin=True, vmin=0.5, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0].T, nf_list.astype(int), nh_list.astype(int), 'number of filters',
                    'number of hidden neurons', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_transpose_vmin0_5_verbose',
                    set_vmin=True,
                    vmin=0.5, cblabel='acc', transpose=True)
    F1score_heatmap(PM_nfnh[:, :, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$',
                    '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_transpose_vmin0_5',
                    set_vmin=True,
                    vmin=0.5, cblabel='acc', transpose=True)

    total_Pos = np.sum([ConMat[0, 0, 0, 0, 4], ConMat[0, 0, 0, 0, 7]])
    total_Neg = np.sum([ConMat[0, 0, 0, 0, 5], ConMat[0, 0, 0, 0, 6]])
    F1score_heatmap(CM_nfnh[:, :, 0, 0].T/total_Pos, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\ConMat_TP_heatmap_transpose', set_vmin=True, vmin=0, cblabel='TBR', transpose=True)
    F1score_heatmap(CM_nfnh[:, :, 1, 0].T/total_Neg, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\ConMat_FP_heatmap_transpose', set_vmin=True, vmin=0, cblabel='FBR', transpose=True)
    F1score_heatmap(CM_nfnh[:, :, 2, 0].T/total_Neg, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\ConMat_TN_heatmap_transpose', set_vmin=True, vmin=0, cblabel='TAR', transpose=True)
    F1score_heatmap(CM_nfnh[:, :, 3, 0].T/total_Pos, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\ConMat_FN_heatmap_transpose', set_vmin=True, vmin=0, cblabel='FAR', transpose=True)
    "select the nf, nh, lr combi with the highest PM-score"
    # arg = np.nanargmax(np.abs(KFres[:, PM_index]))
    # nf = KFres[arg, 0].astype(int)
    # nh = KFres[arg, 1].astype(int)
    # lr = KFres[arg, 2]
    # string = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    # n_kfolds = 10
    # res_kfolds = np.zeros((n_kfolds, 5))
    # for kfold in range(10):
    #     res_kfolds[kfold] = np.loadtxt(string+u'kfold={:d}'.format(kfold)+u'\\results.txt', delimiter='\t')
    #     "results.txt: val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1"
    # bestkfold = np.argmax(res_kfolds[:, 1])
    # print(bestkfold)
    # filepath = r'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\'.format(nf, nh, lr, bestkfold)
    # model = tf.keras.models.load_model(
    #     filepath, custom_objects=None, compile=True
    # )
    # model.summary()
    # "load in pre-k train and test data"
    # dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    #
    # "remove class C"
    # indA = np.argwhere(dataprek['y_test'] == 0)
    # indB = np.argwhere(dataprek['y_test'] == 1)
    # indAB = np.append(indA, indB)
    # x_test = dataprek['x_test'][indAB]
    # y_test = dataprek['y_test'][indAB]
    #
    # out = model.predict(x_test)
    # accuracymet = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
    # accuracymet.update_state(y_test, np.argmax(out, axis=1))
    # accuracy = accuracymet.result()
    # precision = determine_precision(np.argmax(out, axis=1), y_test)
    # recall = determine_recall(np.argmax(out, axis=1), y_test)
    # f1score = f1_score(precision, recall)
    # cm = tf.math.confusion_matrix(y_test, np.argmax(out, axis=1), num_classes=2).numpy()
    # plot_confusion_matrix(cm, ['A', 'B'], '.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\cm_testset_bestvalF1', True)
    # ind_A = np.argwhere(y_test==0)
    # ind_B = np.argwhere(y_test==1)
    # print(accuracy)
    # density_AnB_NN(out, ind_A, ind_B, '$n_f = ${:d}, $n_h = ${:d}, kfold = {:d} \t $F_1$ score = {:.3f}'.format(nf, nh, bestkfold, f1score), u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\pdf_testset_truelabel_nf{:d}_nh{:d}_lr{:.4f}_logy'.format(nf, nh, lr), pdfsave=True, logy=True)
    return 0

def foldavg_heatmap_valacc_and_testTPR_TNR_BA():
    # resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\kfold_avg_val_results.txt'
    k = 4
    strings = [
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to9',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to13',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to17',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18']
    for parts in range(len(strings)):
        resultstring = strings[parts]
        if parts == 0:
            KFres = KFres = np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t')
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'), axis=0)
    # KFres = np.loadtxt(resultstring, delimiter='\t')
    # ConMat = np.loadtxt(r'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_TP_FP_TN_FN_lm.txt', delimiter=',')
    ConMat = np.load(r'D:\\data\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_nfnhlrkfold_TP_FP_TN_FN.npy')
    "load in TP, FP, TN, FN over test set. ConMat=[nf, nh, lr, kfold, TP, FP, TN, FN]"
    "KFres: [nf, nh, lr, val_acc_avg, val_acc_var, val_loss_avg, val_loss_var, val_prec_avg, val_prec_var, val_rec_avg, val_rec_var, val_f1_avg, val_f1_var]"
    "choose performance measure (PM) index"
    PM_index = 3
    nf_list = np.unique(KFres[:, 0])
    nh_list = np.unique(KFres[:, 1])
    PM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 2))
    "CM_nfnh: [nf, nh, {TP, FP, TN, FN}, {avg, var}]"
    testPM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 7, 2))
    n_lr = 6
    for i in range(np.shape(nf_list)[0]):
        for j in range(np.shape(nh_list)[0]):
            # start = i*n_lr
            nfnhargs = np.argwhere(np.logical_and(KFres[:, 0] == nf_list[i], KFres[:, 1] == nh_list[j]))
            arg = np.nanargmax(np.abs(KFres[nfnhargs, PM_index]))
            nf = KFres[nfnhargs[arg, 0], 0].astype(int)
            nh = KFres[nfnhargs[arg, 0], 1].astype(int)
            lr = KFres[nfnhargs[arg, 0], 2]

            nf_arg = np.argwhere(ConMat[:, 0, 0, 0, 0] == nf)[0, 0]
            nh_arg = np.argwhere(ConMat[0, :, 0, 0, 1] == nh)[0, 0]
            lr_arg = np.argwhere(ConMat[0, 0, :, 0, 2] == lr)[0, 0]

            # red = np.logical_and.reduce((ConMat[:, 0].astype(int) == nf, ConMat[:, 1].astype(int) == nh, ConMat[:, 2] == lr))
            # CM_arg = np.argwhere(red)
                # np.logical_and(np.logical_and(ConMat[:, 0].astype(int) ==nf, ConMat[:, 1].astype(int) == nh), ConMat[:, 2]==lr))

            PM_nfnh[i, j, 0] = KFres[nfnhargs[arg, 0], PM_index]
            PM_nfnh[i, j, 1] = KFres[nfnhargs[arg, 0], PM_index+1]

            TPR = ConMat[nf_arg, nh_arg, lr_arg, :, 4]/(ConMat[nf_arg,nh_arg, lr_arg,:,  4] + ConMat[nf_arg, nh_arg, lr_arg,:, 7])
            TNR = ConMat[nf_arg, nh_arg, lr_arg, :, 6]/(ConMat[nf_arg, nh_arg, lr_arg, :, 6] + ConMat[nf_arg, nh_arg, lr_arg, :, 5])
            BA = (TPR+TNR)/2.

            PPV = np.divide(ConMat[nf_arg, nh_arg, lr_arg, :, 4],
                            ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 5],
                            out=np.zeros_like(ConMat[nf_arg, nh_arg, lr_arg, :, 4]),
                            where=ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 5] != 0)
            NPV = np.divide(ConMat[nf_arg, nh_arg, lr_arg, :, 6],
                            ConMat[nf_arg, nh_arg, lr_arg, :, 6] + ConMat[nf_arg, nh_arg, lr_arg, :, 7],
                            out=np.zeros_like(ConMat[nf_arg, nh_arg, lr_arg, :, 6]),
                            where=ConMat[nf_arg, nh_arg, lr_arg, :, 6] + ConMat[nf_arg, nh_arg, lr_arg, :, 7] != 0)
            total_pred = np.sum(ConMat[nf_arg, nh_arg, lr_arg, :, 4:8], axis=1)
            P_ratio = np.divide(ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 5],
                                total_pred)
            O_ratio = np.divide(ConMat[nf_arg, nh_arg, lr_arg, :, 6] + ConMat[nf_arg, nh_arg, lr_arg, :, 7],
                                total_pred)

            testPM_nfnh[i, j, 0, 0] = np.nanmean(TPR)
            testPM_nfnh[i, j, 0, 1] = np.nanvar(TPR)
            testPM_nfnh[i, j, 1, 0] = np.nanmean(TNR)
            testPM_nfnh[i, j, 1, 1] = np.nanvar(TNR)
            testPM_nfnh[i, j, 2, 0] = np.nanmean(BA)
            testPM_nfnh[i, j, 2, 1] = np.nanvar(BA)
            testPM_nfnh[i, j, 3, 0] = np.nanmean(PPV)
            testPM_nfnh[i, j, 3, 1] = np.nanvar(PPV)
            testPM_nfnh[i, j, 4, 0] = np.nanmean(NPV)
            testPM_nfnh[i, j, 4, 1] = np.nanvar(NPV)
            testPM_nfnh[i, j, 5, 0] = np.nanmean(P_ratio)
            testPM_nfnh[i, j, 5, 1] = np.nanvar(P_ratio)
            testPM_nfnh[i, j, 6, 0] = np.nanmean(O_ratio)
            testPM_nfnh[i, j, 6, 1] = np.nanvar(O_ratio)
            # CM_nfnh[i, j, 1, 0], CM_nfnh[i, j, 1, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 5]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 5])
            # CM_nfnh[i, j, 2, 0], CM_nfnh[i, j, 2, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 6]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 6])
            # CM_nfnh[i, j, 3, 0], CM_nfnh[i, j, 3, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 7]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 7])
    np.save('.\\testPM_nfnh_{:d}x{:d}.npy'.format(k, k), testPM_nfnh)
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_square', set_vmin=False, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True, savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\F1score_heatmap_vmin0', set_vmin=True, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_square_vmin0_5', set_vmin=True, vmin=0.5, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), 'number of hidden neurons', 'number of filters', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_square_vmin0_5_verbose', set_vmin=True, vmin=0.5, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0].T, nf_list.astype(int), nh_list.astype(int), 'number of filters',
                    'number of hidden neurons', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_square_transpose_vmin0_5_verbose',
                    set_vmin=True,
                    vmin=0.5, cblabel='acc', transpose=True)
    F1score_heatmap(PM_nfnh[:, :, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$',
                    '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_heatmap_square_transpose_vmin0_5',
                    set_vmin=True,
                    vmin=0.5, cblabel='acc', transpose=True)

    total_Pos = np.sum([ConMat[0, 0, 0, 0, 4], ConMat[0, 0, 0, 0, 7]])
    total_Neg = np.sum([ConMat[0, 0, 0, 0, 5], ConMat[0, 0, 0, 0, 6]])

    F1score_heatmap(testPM_nfnh[:, :, 0, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_TPR_heatmap_square_transpose', set_vmin=True, vmin=0, cblabel='TBR', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 1, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_TNR_heatmap_square_transpose', set_vmin=True, vmin=0, cblabel='TAR', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 2, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_BA_heatmap_square_transpose', set_vmin=True, vmin=0, cblabel='BA', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 3, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_PPV_heatmap_square_transpose', set_vmin=True,
                    vmin=0, cblabel='PPV', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 4, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_NPV_heatmap_square_transpose', set_vmin=True,
                    vmin=0, cblabel='OPV', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 5, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_Pratio_heatmap_square_transpose', set_vmin=True,
                    vmin=0, cblabel='pred P ratio', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 6, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_Oratio_heatmap_square_transpose', set_vmin=True,
                    vmin=0, cblabel='pred O ratio', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 5, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_Pratio_logc_heatmap_square_transpose',
                    set_vmin=True,
                    vmin=.5*10**-2, cblabel='pred P ratio', transpose=True, logc=True)

    F1score_heatmap(testPM_nfnh[:, :, 0, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_TPR_heatmap_square_transpose_nocbar', set_vmin=True,
                    vmin=0, cblabel='TBR', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 1, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_TNR_heatmap_square_transpose_nocbar', set_vmin=True,
                    vmin=0, cblabel='TAR', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 2, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_BA_heatmap_square_transpose_nocbar', set_vmin=True, vmin=0,
                    cblabel='BA', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 3, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_PPV_heatmap_square_transpose_nocbar',
                    set_vmin=True,
                    vmin=0, cblabel='PPV', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 4, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_NPV_heatmap_square_transpose_nocbar',
                    set_vmin=True,
                    vmin=0, cblabel='OPV', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 5, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_Pratio_heatmap_square_transpose_nocbar',
                    set_vmin=True,
                    vmin=0, cblabel='pred P ratio', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 6, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_Oratio_heatmap_square_transpose_nocbar',
                    set_vmin=True,
                    vmin=0, cblabel='pred O ratio', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 5, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_Pratio_logc_heatmap_square_transpose_nocbar',
                    set_vmin=True,
                    vmin=0.5*10**-2, cblabel='pred P ratio', transpose=True, cbar=False, logc=True)

    "select the nf, nh, lr combi with the highest PM-score"
    # arg = np.nanargmax(np.abs(KFres[:, PM_index]))
    # nf = KFres[arg, 0].astype(int)
    # nh = KFres[arg, 1].astype(int)
    # lr = KFres[arg, 2]
    # string = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    # n_kfolds = 10
    # res_kfolds = np.zeros((n_kfolds, 5))
    # for kfold in range(10):
    #     res_kfolds[kfold] = np.loadtxt(string+u'kfold={:d}'.format(kfold)+u'\\results.txt', delimiter='\t')
    #     "results.txt: val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1"
    # bestkfold = np.argmax(res_kfolds[:, 1])
    # print(bestkfold)
    # filepath = r'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\'.format(nf, nh, lr, bestkfold)
    # model = tf.keras.models.load_model(
    #     filepath, custom_objects=None, compile=True
    # )
    # model.summary()
    # "load in pre-k train and test data"
    # dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    #
    # "remove class C"
    # indA = np.argwhere(dataprek['y_test'] == 0)
    # indB = np.argwhere(dataprek['y_test'] == 1)
    # indAB = np.append(indA, indB)
    # x_test = dataprek['x_test'][indAB]
    # y_test = dataprek['y_test'][indAB]
    #
    # out = model.predict(x_test)
    # accuracymet = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
    # accuracymet.update_state(y_test, np.argmax(out, axis=1))
    # accuracy = accuracymet.result()
    # precision = determine_precision(np.argmax(out, axis=1), y_test)
    # recall = determine_recall(np.argmax(out, axis=1), y_test)
    # f1score = f1_score(precision, recall)
    # cm = tf.math.confusion_matrix(y_test, np.argmax(out, axis=1), num_classes=2).numpy()
    # plot_confusion_matrix(cm, ['A', 'B'], '.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\cm_testset_bestvalF1', True)
    # ind_A = np.argwhere(y_test==0)
    # ind_B = np.argwhere(y_test==1)
    # print(accuracy)
    # density_AnB_NN(out, ind_A, ind_B, '$n_f = ${:d}, $n_h = ${:d}, kfold = {:d} \t $F_1$ score = {:.3f}'.format(nf, nh, bestkfold, f1score), u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\pdf_testset_truelabel_nf{:d}_nh{:d}_lr{:.4f}_logy'.format(nf, nh, lr), pdfsave=True, logy=True)
    return 0

def foldavg_heatmap_valacc_and_testTPR_TNR_BA_expanded():
    # resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\kfold_avg_val_results.txt'
    k = 4
    strings = [
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to9',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to13',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to17',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to1_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_1to2_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to3_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_3to4_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to5_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to6_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to7_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_7to8_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to9_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to10_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to11_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_11to12_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to13_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to14_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to15_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_15to16_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_16to17_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18_MoreFilters',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to1_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_1to2_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to3_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_3to4_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to5_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to6_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to7_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_7to8_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to9_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to10_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to11_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_11to12_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to13_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to14_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to15_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_15to16_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_16to17_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18_MoreHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to1_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_1to2_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to3_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_3to4_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to5_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to6_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to7_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_7to8_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to9_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to10_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to11_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_11to12_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to13_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to14_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to15_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_15to16_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_16to17_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to1_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_1to2_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_2to3_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_3to4_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_4to5_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to6_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_6to7_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_7to8_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_8to9_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to10_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_10to11_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_11to12_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_12to13_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to14_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_14to15_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_15to16_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_16to17_MoreFiltersHidden',
        u'E:\\results\\cnniter_HP_GS_SKF_4x4_AB_5050_extended\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18_MoreFiltersHidden'
    ]
    for parts in range(len(strings)):
        resultstring = strings[parts]
        if parts == 0:
            KFres = np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t')
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'), axis=0)
    # KFres = np.loadtxt(resultstring, delimiter='\t')
    # ConMat = np.loadtxt(r'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_TP_FP_TN_FN_lm.txt', delimiter=',')
    ConMat = np.load(r'D:\\data\cnniter_HP_GS_SKF_4x4_AB_5050_2\\testset_expanded_nfnhlrkfold_TP_FP_TN_FN.npy')
    "load in TP, FP, TN, FN over test set. ConMat=[nf, nh, lr, kfold, TP, FP, TN, FN]"
    "KFres: [nf, nh, lr, val_acc_avg, val_acc_var, val_loss_avg, val_loss_var, val_prec_avg, val_prec_var, val_rec_avg, val_rec_var, val_f1_avg, val_f1_var]"
    "choose performance measure (PM) index"
    PM_index = 3
    nf_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40])
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 110, 120, 130, 140, 150, 160,
                        170, 180, 190, 200, 210, 220, 230, 240, 250, 260, 270, 280])
    PM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 2))
    "CM_nfnh: [nf, nh, {TP, FP, TN, FN}, {avg, var}]"
    testPM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 3, 2))
    n_lr = 6
    for i in range(np.shape(nf_list)[0]):
        for j in range(np.shape(nh_list)[0]):
            # start = i*n_lr
            nfnhargs = np.argwhere(np.logical_and(KFres[:, 0] == nf_list[i], KFres[:, 1] == nh_list[j]))
            arg = np.nanargmax(np.abs(KFres[nfnhargs, PM_index]))
            nf = KFres[nfnhargs[arg, 0], 0].astype(int)
            nh = KFres[nfnhargs[arg, 0], 1].astype(int)
            lr = KFres[nfnhargs[arg, 0], 2]

            nf_arg = np.argwhere(ConMat[:, 0, 0, 0, 0] == nf)[0, 0]
            nh_arg = np.argwhere(ConMat[0, :, 0, 0, 1] == nh)[0, 0]
            lr_arg = np.argwhere(ConMat[0, 0, :, 0, 2] == lr)[0, 0]

            # red = np.logical_and.reduce((ConMat[:, 0].astype(int) == nf, ConMat[:, 1].astype(int) == nh, ConMat[:, 2] == lr))
            # CM_arg = np.argwhere(red)
                # np.logical_and(np.logical_and(ConMat[:, 0].astype(int) ==nf, ConMat[:, 1].astype(int) == nh), ConMat[:, 2]==lr))

            PM_nfnh[i, j, 0] = KFres[nfnhargs[arg, 0], PM_index]
            PM_nfnh[i, j, 1] = KFres[nfnhargs[arg, 0], PM_index+1]

            TPR = ConMat[nf_arg, nh_arg, lr_arg, :, 4]/(ConMat[nf_arg,nh_arg, lr_arg,:,  4] + ConMat[nf_arg, nh_arg, lr_arg,:, 7])
            TNR = ConMat[nf_arg, nh_arg, lr_arg, :, 6]/(ConMat[nf_arg, nh_arg, lr_arg, :, 6] + ConMat[nf_arg, nh_arg, lr_arg, :, 5])
            BA = (TPR+TNR)/2.

            testPM_nfnh[i, j, 0, 0] = np.nanmean(TPR)
            testPM_nfnh[i, j, 0, 1] = np.nanvar(TPR)
            testPM_nfnh[i, j, 1, 0] = np.nanmean(TNR)
            testPM_nfnh[i, j, 1, 1] = np.nanvar(TNR)
            testPM_nfnh[i, j, 2, 0] = np.nanmean(BA)
            testPM_nfnh[i, j, 2, 1] = np.nanvar(BA)
            # CM_nfnh[i, j, 1, 0], CM_nfnh[i, j, 1, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 5]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 5])
            # CM_nfnh[i, j, 2, 0], CM_nfnh[i, j, 2, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 6]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 6])
            # CM_nfnh[i, j, 3, 0], CM_nfnh[i, j, 3, 1] = np.nanmean(ConMat[nf_arg, nh_arg, lr_arg, :, 7]), np.nanvar(ConMat[nf_arg, nh_arg, lr_arg, :, 7])

    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_expanded_heatmap_square', set_vmin=False, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True, savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\F1score_expanded_heatmap_vmin0', set_vmin=True, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), '$n_h$', '$n_f$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_expanded_heatmap_square_vmin0_5', set_vmin=True, vmin=0.5, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0], nh_list.astype(int), nf_list.astype(int), 'number of hidden neurons', 'number of filters', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_expanded_heatmap_square_vmin0_5_verbose', set_vmin=True, vmin=0.5, cblabel='acc')
    F1score_heatmap(PM_nfnh[:, :, 0].T, nf_list.astype(int), nh_list.astype(int), 'number of filters',
                    'number of hidden neurons', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_expanded_heatmap_square_transpose_vmin0_5_verbose',
                    set_vmin=True,
                    vmin=0.5, cblabel='acc', transpose=True)
    F1score_heatmap(PM_nfnh[:, :, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$',
                    '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\PMscore_expanded_heatmap_square_transpose_vmin0_5',
                    set_vmin=True,
                    vmin=0.5, cblabel='acc', transpose=True)

    total_Pos = np.sum([ConMat[0, 0, 0, 0, 4], ConMat[0, 0, 0, 0, 7]])
    total_Neg = np.sum([ConMat[0, 0, 0, 0, 5], ConMat[0, 0, 0, 0, 6]])
    F1score_heatmap(testPM_nfnh[:, :, 0, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_expanded_TPR_heatmap_square_transpose', set_vmin=True, vmin=0.5, cblabel='TBR', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 1, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_expanded_TNR_heatmap_square_transpose', set_vmin=True, vmin=0.5, cblabel='TAR', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 2, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_expanded_BA_heatmap_square_transpose', set_vmin=True, vmin=0.5, cblabel='BA', transpose=True)
    F1score_heatmap(testPM_nfnh[:, :, 0, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_expanded_TPR_heatmap_square_transpose_nocbar', set_vmin=True,
                    vmin=0.5, cblabel='TBR', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 1, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_expanded_TNR_heatmap_square_transpose_nocbar', set_vmin=True,
                    vmin=0.5, cblabel='TAR', transpose=True, cbar=False)
    F1score_heatmap(testPM_nfnh[:, :, 2, 0].T, nf_list.astype(int), nh_list.astype(int), '$n_f$', '$n_h$', pdfsave=True,
                    savestring='.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\test_expanded_BA_heatmap_square_transpose_nocbar', set_vmin=True, vmin=0.5,
                    cblabel='BA', transpose=True, cbar=False)

    "select the nf, nh, lr combi with the highest PM-score"
    # arg = np.nanargmax(np.abs(KFres[:, PM_index]))
    # nf = KFres[arg, 0].astype(int)
    # nh = KFres[arg, 1].astype(int)
    # lr = KFres[arg, 2]
    # string = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
    # n_kfolds = 10
    # res_kfolds = np.zeros((n_kfolds, 5))
    # for kfold in range(10):
    #     res_kfolds[kfold] = np.loadtxt(string+u'kfold={:d}'.format(kfold)+u'\\results.txt', delimiter='\t')
    #     "results.txt: val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1"
    # bestkfold = np.argmax(res_kfolds[:, 1])
    # print(bestkfold)
    # filepath = r'D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\'.format(nf, nh, lr, bestkfold)
    # model = tf.keras.models.load_model(
    #     filepath, custom_objects=None, compile=True
    # )
    # model.summary()
    # "load in pre-k train and test data"
    # dataprek = np.load(u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\data_prek_xy_train_trainraw_test.npz')
    #
    # "remove class C"
    # indA = np.argwhere(dataprek['y_test'] == 0)
    # indB = np.argwhere(dataprek['y_test'] == 1)
    # indAB = np.append(indA, indB)
    # x_test = dataprek['x_test'][indAB]
    # y_test = dataprek['y_test'][indAB]
    #
    # out = model.predict(x_test)
    # accuracymet = tf.keras.metrics.Accuracy(name='accuracy', dtype=None)
    # accuracymet.update_state(y_test, np.argmax(out, axis=1))
    # accuracy = accuracymet.result()
    # precision = determine_precision(np.argmax(out, axis=1), y_test)
    # recall = determine_recall(np.argmax(out, axis=1), y_test)
    # f1score = f1_score(precision, recall)
    # cm = tf.math.confusion_matrix(y_test, np.argmax(out, axis=1), num_classes=2).numpy()
    # plot_confusion_matrix(cm, ['A', 'B'], '.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\cm_testset_bestvalF1', True)
    # ind_A = np.argwhere(y_test==0)
    # ind_B = np.argwhere(y_test==1)
    # print(accuracy)
    # density_AnB_NN(out, ind_A, ind_B, '$n_f = ${:d}, $n_h = ${:d}, kfold = {:d} \t $F_1$ score = {:.3f}'.format(nf, nh, bestkfold, f1score), u'.\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\pdf_testset_truelabel_nf{:d}_nh{:d}_lr{:.4f}_logy'.format(nf, nh, lr), pdfsave=True, logy=True)
    return 0

def test_random_walk_trueclass(k, nf):
    if k == 3 or k == 4:
        savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\'.format(k, k)
    else:
        savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\random_walks\\'.format(k, k)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    strings = [
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to9',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to13',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to17',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18']
    for parts in range(len(strings)):
        resultstring = strings[parts]
        if parts == 0:
            KFres = np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t')
            # KFres = KFres[np.newaxis, ...]
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'), axis=0)
    nh_list = np.unique(KFres[:, 1])
    # resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\kfold_avg_val_results.txt'
    # KFres = np.loadtxt(resultstring, delimiter='\t')
    "KFres: [nf, nh, lr, val_acc_avg, val_acc_var, val_loss_avg, val_loss_var, val_prec_avg, val_prec_var, val_rec_avg, val_rec_var, val_f1_avg, val_f1_var]"
    "choose performance measure (PM) index"
    PM_index = 3
    # nf = 20
    # nh = 10
    params_best_networks = np.zeros((len(nh_list), 3))
    for j, nh in enumerate(nh_list):
        nf_ind = np.argwhere(KFres[:, 0] == nf)
        nh_ind = np.argwhere(KFres[nf_ind[:, 0], 1] == nh)
        inds = nf_ind[nh_ind[:, 0], 0]
        lr_ind = np.argmax(KFres[inds[:], PM_index])
        ind = inds[lr_ind]
        lr = KFres[ind, 2]

        # stringind = np.argwhere(nh_list == nh)
        # # stringind=0
        # string = strings[stringind[0, 0]] + '\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
        # n_kfolds = 10
        # res_kfolds = np.zeros((n_kfolds, 5))
        # for kfold in range(10):
        #     res_kfolds[kfold] = np.loadtxt(string + u'kfold={:d}'.format(kfold) + u'\\results.txt', delimiter='\t')
        #     "results.txt: val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1"
        # bestkfold = np.argmax(res_kfolds[:, 4])
        params_best_networks[j, 0] = nh
        params_best_networks[j, 1] = nf
        params_best_networks[j, 2] = lr
        # params_best_networks[j, 3] = bestkfold
        # print('best k-fold:')
        # print(bestkfold)


    Nsteps = k * k

    tPpPtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpPtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpPtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpPtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    tPpOtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpOtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpOtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpOtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    tOpPtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpPtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpPtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpPtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    tOpOtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpOtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpOtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpOtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    batch_size = 5000
    for j, l in enumerate(range(0, 150000, batch_size)):
        for i in range(batch_size * j, batch_size * (j + 1)):
            if i == batch_size * j:
                configs = np.load(u'D:\\data\\random_walks_{:d}x{:d}\\configlist_test_{:d}.npy'.format(k, k, i))
                labels = np.load(u'D:\\data\\random_walks_{:d}x{:d}\\lmlist_test_{:d}.npy'.format(k, k, i))
            else:
                configs = np.append(configs, np.load(u'D:\\data\\random_walks_{:d}x{:d}\\configlist_test_{:d}.npy'
                                                     .format(k, k, i)), axis=0)
                labels = np.append(labels, np.load(u'D:\\data\\random_walks_{:d}x{:d}\\lmlist_test_{:d}.npy'
                                                   .format(k, k, i)), axis=0)
        print('configs loaded batch {:d}'.format(j))
        for m in range(np.shape(params_best_networks)[0]):
            nf = params_best_networks[m, 1]
            nh = params_best_networks[m, 0]
            lr = params_best_networks[m, 2]
            for kfold in range(10):
                filepath = u"D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".\
                    format(int(nf), int(nh), float(lr), int(kfold))
                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(
                    filepath, custom_objects=None, compile=True
                )
                model.summary()

                out = model.predict(configs[:, :, :, np.newaxis].astype(np.float32))
                y_pred = np.argmax(out, axis=1)
                # ind_P = np.argwhere(y_pred[1:] == 1)
                # ind_O = np.argwhere(y_pred[1:] == 0)
                for i in range(batch_size):
                    ind_0 = i * (Nsteps+1)
                    ind_l = ind_0 + 1
                    ind_r = ind_0 + 1 + Nsteps
                    ind_tPpP = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 1, labels[ind_l:ind_r] > 0))
                    ind_tPpO = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 0, labels[ind_l:ind_r] > 0))
                    ind_tOpP = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 1, labels[ind_l:ind_r] == 0))
                    ind_tOpO = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 0, labels[ind_l:ind_r] == 0))
                    if y_pred[ind_0] == 1:
                        'predicted P'
                        if labels[ind_0] > 0:
                            "true label P"
                            tPpPtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tPpPtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tPpPtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tPpPtotOpO[m, kfold, ind_tOpO[:, 0]] += 1
                        else:
                            "true label O"
                            tOpPtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tOpPtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tOpPtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tOpPtotOpO[m, kfold, ind_tOpO[:, 0]] += 1
                    else:
                        'predicted O'
                        if labels[ind_0] > 0:
                            "true label P"
                            tPpOtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tPpOtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tPpOtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tPpOtotOpO[m, kfold, ind_tOpO[:, 0]] += 1
                        else:
                            "true label O"
                            tOpOtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tOpOtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tOpOtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tOpOtotOpO[m, kfold, ind_tOpO[:, 0]] += 1

        del configs, labels
    for m in range(np.shape(params_best_networks)[0]):
        nh = params_best_networks[m, 0]
        nf = params_best_networks[m, 1]
        lr = params_best_networks[m, 2]
        np.savez(savedir + 'nf{:d}_nh{:d}_lr{:.4f}_kfolds_probability_classchange_randomwalk_{:d}x{:d}_test_true_and_'
                           'predicted.npz'.format(int(nf), int(nh), float(lr), int(k), int(k)),
                 tPpPtotPpP=tPpPtotPpP[m], tPpPtotPpO=tPpPtotPpO[m], tPpPtotOpP=tPpPtotOpP[m], tPpPtotOpO=tPpPtotOpO[m],
                 tPpOtotPpP=tPpOtotPpP[m], tPpOtotPpO=tPpOtotPpO[m], tPpOtotOpP=tPpOtotOpP[m], tPpOtotOpO=tPpOtotOpO[m],
                 tOpPtotPpP=tOpPtotPpP[m], tOpPtotPpO=tOpPtotPpO[m], tOpPtotOpP=tOpPtotOpP[m], tOpPtotOpO=tOpPtotOpO[m],
                 tOpOtotPpP=tOpOtotPpP[m], tOpOtotPpO=tOpOtotPpO[m], tOpOtotOpP=tOpOtotOpP[m], tOpOtotOpO=tOpOtotOpO[m])
    return 0

def test_random_walk_trueclass_euclidian(k, nf):
    if k == 3 or k == 4:
        savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks_euclidian\\'.format(k, k)
    else:
        savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\random_walks_euclidian\\'.format(k, k)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    strings = [
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to9',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to13',
        u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to17']
    for parts in range(len(strings)):
        resultstring = strings[parts]
        if parts == 0:
            KFres = np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t')
            # KFres = KFres[np.newaxis, ...]
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'), axis=0)
    nh_list = np.unique(KFres[:, 1])
    # resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\kfold_avg_val_results.txt'
    # KFres = np.loadtxt(resultstring, delimiter='\t')
    "KFres: [nf, nh, lr, val_acc_avg, val_acc_var, val_loss_avg, val_loss_var, val_prec_avg, val_prec_var, val_rec_avg, val_rec_var, val_f1_avg, val_f1_var]"
    "choose performance measure (PM) index"
    PM_index = 3
    # nf = 20
    # nh = 10
    params_best_networks = np.zeros((len(nh_list), 3))
    for j, nh in enumerate(nh_list):
        nf_ind = np.argwhere(KFres[:, 0] == nf)
        nh_ind = np.argwhere(KFres[nf_ind[:, 0], 1] == nh)
        inds = nf_ind[nh_ind[:, 0], 0]
        lr_ind = np.argmax(KFres[inds[:], PM_index])
        ind = inds[lr_ind]
        lr = KFres[ind, 2]

        # stringind = np.argwhere(nh_list == nh)
        # # stringind=0
        # string = strings[stringind[0, 0]] + '\\logs\\nf={:d}_nh={:d}_lr={:.4f}\\'.format(nf, nh, lr)
        # n_kfolds = 10
        # res_kfolds = np.zeros((n_kfolds, 5))
        # for kfold in range(10):
        #     res_kfolds[kfold] = np.loadtxt(string + u'kfold={:d}'.format(kfold) + u'\\results.txt', delimiter='\t')
        #     "results.txt: val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1"
        # bestkfold = np.argmax(res_kfolds[:, 4])
        params_best_networks[j, 0] = nh
        params_best_networks[j, 1] = nf
        params_best_networks[j, 2] = lr
        # params_best_networks[j, 3] = bestkfold
        # print('best k-fold:')
        # print(bestkfold)


    Nsteps = k * k

    tPpPtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpPtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpPtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpPtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    tPpOtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpOtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpOtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tPpOtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    tOpPtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpPtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpPtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpPtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    tOpOtotPpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpOtotPpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpOtotOpO = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)
    tOpOtotOpP = np.zeros((np.shape(params_best_networks)[0], 10, Nsteps), dtype=int)

    batch_size = 5000
    for j, l in enumerate(range(0, 150000, batch_size)):
        for i in range(batch_size * j, batch_size * (j + 1)):
            if i == batch_size * j:
                configs = np.load(u'D:\\data\\random_walks_{:d}x{:d}_euclidian\\configlist_test_{:d}.npy'.format(k, k, i))
                labels = np.load(u'D:\\data\\random_walks_{:d}x{:d}_euclidian\\lmlist_test_{:d}.npy'.format(k, k, i))
            else:
                configs = np.append(configs, np.load(u'D:\\data\\random_walks_{:d}x{:d}_euclidian\\configlist_test_{:d}.npy'
                                                     .format(k, k, i)), axis=0)
                labels = np.append(labels, np.load(u'D:\\data\\random_walks_{:d}x{:d}_euclidian\\lmlist_test_{:d}.npy'
                                                   .format(k, k, i)), axis=0)
        print('configs loaded batch {:d}'.format(j))
        for m in range(np.shape(params_best_networks)[0]):
            nf = params_best_networks[m, 1]
            nh = params_best_networks[m, 0]
            lr = params_best_networks[m, 2]
            for kfold in range(10):
                filepath = u"D:\\data\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".\
                    format(int(nf), int(nh), float(lr), int(kfold))
                tf.keras.backend.clear_session()
                model = tf.keras.models.load_model(
                    filepath, custom_objects=None, compile=True
                )
                model.summary()

                out = model.predict(configs[:, :, :, np.newaxis].astype(np.float32))
                y_pred = np.argmax(out, axis=1)
                # ind_P = np.argwhere(y_pred[1:] == 1)
                # ind_O = np.argwhere(y_pred[1:] == 0)
                for i in range(batch_size):
                    ind_0 = i * (Nsteps+1)
                    ind_l = ind_0 + 1
                    ind_r = ind_0 + 1 + Nsteps
                    ind_tPpP = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 1, labels[ind_l:ind_r] > 0))
                    ind_tPpO = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 0, labels[ind_l:ind_r] > 0))
                    ind_tOpP = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 1, labels[ind_l:ind_r] == 0))
                    ind_tOpO = np.argwhere(np.logical_and(y_pred[ind_l:ind_r] == 0, labels[ind_l:ind_r] == 0))
                    if y_pred[ind_0] == 1:
                        'predicted P'
                        if labels[ind_0] > 0:
                            "true label P"
                            tPpPtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tPpPtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tPpPtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tPpPtotOpO[m, kfold, ind_tOpO[:, 0]] += 1
                        else:
                            "true label O"
                            tOpPtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tOpPtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tOpPtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tOpPtotOpO[m, kfold, ind_tOpO[:, 0]] += 1
                    else:
                        'predicted O'
                        if labels[ind_0] > 0:
                            "true label P"
                            tPpOtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tPpOtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tPpOtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tPpOtotOpO[m, kfold, ind_tOpO[:, 0]] += 1
                        else:
                            "true label O"
                            tOpOtotPpP[m, kfold, ind_tPpP[:, 0]] += 1
                            tOpOtotPpO[m, kfold, ind_tPpO[:, 0]] += 1
                            tOpOtotOpP[m, kfold, ind_tOpP[:, 0]] += 1
                            tOpOtotOpO[m, kfold, ind_tOpO[:, 0]] += 1

        del configs, labels
    for m in range(np.shape(params_best_networks)[0]):
        nh = params_best_networks[m, 0]
        nf = params_best_networks[m, 1]
        lr = params_best_networks[m, 2]
        np.savez(savedir + 'nf{:d}_nh{:d}_lr{:.4f}_kfolds_probability_classchange_randomwalk_{:d}x{:d}_test_true_and_'
                           'predicted.npz'.format(int(nf), int(nh), float(lr), int(k), int(k)),
                 tPpPtotPpP=tPpPtotPpP[m], tPpPtotPpO=tPpPtotPpO[m], tPpPtotOpP=tPpPtotOpP[m], tPpPtotOpO=tPpPtotOpO[m],
                 tPpOtotPpP=tPpOtotPpP[m], tPpOtotPpO=tPpOtotPpO[m], tPpOtotOpP=tPpOtotOpP[m], tPpOtotOpO=tPpOtotOpO[m],
                 tOpPtotPpP=tOpPtotPpP[m], tOpPtotPpO=tOpPtotPpO[m], tOpPtotOpP=tOpPtotOpP[m], tOpPtotOpO=tOpPtotOpO[m],
                 tOpOtotPpP=tOpOtotPpP[m], tOpOtotPpO=tOpOtotPpO[m], tOpOtotOpP=tOpOtotOpP[m], tOpOtotOpO=tOpOtotOpO[m])
    return 0

def load_data_kfolds(nf, nh, lr, kp, tOpOtotOpO_broken=False):
    if kp > 4:
        savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\random_walks\\'.format(kp, kp)
    else:
        savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\'.format(kp, kp)
    dat = np.load(
        savedir + 'nf{:d}_nh{:d}_lr{:.4f}_kfolds_probability_classchange_randomwalk_{:d}x{:d}_test_true_and_predicted'
                  '.npz'.format(
            int(nf), int(nh), float(lr), int(kp), int(kp)))
    tPpPtotPpP = dat['tPpPtotPpP']
    tPpPtotPpO = dat['tPpPtotPpO']
    tPpPtotOpO = dat['tPpPtotOpO']
    tPpPtotOpP = dat['tPpPtotOpP']
    tPpOtotPpP = dat['tPpOtotPpP']
    tPpOtotPpO = dat['tPpOtotPpO']
    tPpOtotOpO = dat['tPpOtotOpO']
    tPpOtotOpP = dat['tPpOtotOpP']
    tOpPtotPpP = dat['tOpPtotPpP']
    tOpPtotPpO = dat['tOpPtotPpO']
    tOpPtotOpO = dat['tOpPtotOpO']
    tOpPtotOpP = dat['tOpPtotOpP']
    tOpOtotPpP = dat['tOpOtotPpP']
    tOpOtotPpO = dat['tOpOtotPpO']
    tOpOtotOpO = dat['tOpOtotOpO']
    tOpOtotOpP = dat['tOpOtotOpP']

    if kp == 3 or kp == 4:
        datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050'
                                   '_2\\'
                                   .format(kp, kp))
        data = np.load(datapath + '\\data_prek_xy_train_trainraw_test.npz')
    else:
        datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050'
                                   '\\'
                                   .format(kp, kp))
        data = np.load(datapath + '\\data_prek_xy_train_trainraw_test_{:d}x{:d}.npz'.format(kp, kp))

    "remove nan data"
    # nanind = np.argwhere(np.isnan(raw_results[:, 2]))
    # raw_data = np.delete(raw_data, nanind[:, 0], axis=0)
    # raw_results = np.delete(raw_results, nanind[:, 0], axis=0)
    y_test = data['y_test']
    x_test = data['x_test'][:, :, :, 0]

    Number_of_O = np.sum(y_test == 0)
    Number_of_P = np.sum(y_test == 1)

    if tOpOtotOpO_broken:
        tOpOtotOpO = Number_of_O - tOpOtotPpO - tOpOtotPpP - tOpOtotOpP - tOpPtotOpO - tOpPtotPpO - tOpPtotPpP - tOpPtotOpP

    tOpOtotO = np.add(tOpOtotOpP, tOpOtotOpO)
    tOpPtotO = np.add(tOpPtotOpP, tOpPtotOpO)
    tOtotO = np.add(tOpOtotO, tOpPtotO)
    tOpOtotP = np.add(tOpOtotPpP, tOpOtotPpO)
    tOpPtotP = np.add(tOpPtotPpP, tOpPtotPpO)
    tOtotP = np.add(tOpPtotP, tOpOtotP)
    # print(np.add(tOtotP, tOtotO))

    tPpOtotO = np.add(tPpOtotOpP, tPpOtotOpO)
    tPpPtotO = np.add(tPpPtotOpP, tPpPtotOpO)
    tPtotO = np.add(tPpOtotO, tPpPtotO)
    tPpOtotP = np.add(tPpOtotPpP, tPpOtotPpO)
    tPpPtotP = np.add(tPpPtotPpP, tPpPtotPpO)
    tPtotP = np.add(tPpPtotP, tPpOtotP)

    p_tOtotP = np.divide(tOtotP, np.add(tOtotP, tOtotO))
    p_tPtotP = np.divide(tPtotP, np.add(tPtotP, tPtotO))
    tPpPtopP = np.add(tPpPtotPpP, tPpPtotOpP)
    tPpPtopO = np.add(tPpPtotPpO, tPpPtotOpO)
    p_tPpPtopP = np.divide(tPpPtopP, np.add(tPpPtopP, tPpPtopO))

    tOpOtopP = np.add(tOpOtotOpP, tOpOtotPpP)
    tOpOtopO = np.add(tOpOtotOpO, tOpOtotPpO)
    p_tOpOtopP = np.divide(tOpOtopP, np.add(tOpOtopO, tOpOtopP))

    tOpPtopP = np.add(tOpPtotOpP, tOpPtotPpP)
    tOpPtopO = np.add(tOpPtotOpO, tOpPtotPpO)
    p_tOpPtopP = np.divide(tOpPtopP, np.add(tOpPtopP, tOpPtopO))

    var_tPpPtopP = np.divide(
        np.multiply(tPpPtopP, np.square(1 - p_tPpPtopP)) + np.multiply(tPpPtopO, np.square(-p_tPpPtopP)),
        np.add(tPpPtopP, tPpPtopO))
    var_tOpPtopP = np.divide(
        np.multiply(tOpPtopP, np.square(1 - p_tOpPtopP)) + np.multiply(tOpPtopO, np.square(-p_tOpPtopP)),
        np.add(tOpPtopP, tOpPtopO))
    var_tPtotP = np.divide(np.multiply(tPtotP, np.square(1 - p_tPtotP)) + np.multiply(tPtotO, np.square(-p_tPtotP)),
                           np.add(tPtotP, tPtotO))
    var_tOpOtopP = np.divide(
        np.multiply(tOpOtopP, np.square(1 - p_tOpOtopP)) + np.multiply(tOpOtopO, np.square(-p_tOpOtopP)),
        np.add(tOpOtopP, tOpOtopO))
    var_tOtotP = np.divide(np.multiply(tOtotP, np.square(1 - p_tOtotP)) + np.multiply(tOtotO, np.square(-p_tOtotP)),
                           np.add(tOtotP, tOtotO))

    tPpOtopP = np.add(tPpOtotOpP, tPpOtotPpP)
    tPpOtopO = np.add(tPpOtotOpO, tPpOtotPpO)
    p_tPpOtopP = np.divide(tPpOtopP, np.add(tPpOtopP, tPpOtopO))

    N_tPpP = np.add(tPpPtopO, tPpPtopP)
    N_tPpO = np.add(tPpOtopO, tPpOtopP)
    N_tOpP = np.add(tOpPtopO, tOpPtopP)
    N_tOpO = np.add(tOpOtopO, tOpOtopP)

    N_tP = np.add(N_tPpO, N_tPpP)
    N_tO = np.add(N_tOpO, N_tOpP)
    N_pP = np.add(N_tPpP, N_tOpP)
    N_pO = np.add(N_tPpO, N_tOpO)

    acc_tOpO = np.divide(N_tOpO, np.add(N_tOpO, N_tOpP))
    acc_tPpP = np.divide(N_tPpP, np.add(N_tPpP, N_tPpO))

    globals().update(locals())
    return 0

def main():
    k=4
    testset_output_allCNN()
    CM_testset()
    foldavg_heatmap_valacc_and_testTPR_TNR_BA()
    test_random_walk_trueclass(k, 20)
    return 0
if __name__ == "__main__":
    main()