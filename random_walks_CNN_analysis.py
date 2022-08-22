import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import matplotlib

nf = 20
nh = 10
k = 5

tOpOtotOpO_broken = False
Nsteps = k * k
if k > 4:
    savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\'.format(k, k)
else:
    savedir = u'.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\'.format(k, k)

def load_data(nf, nh, lr, kfold):
    dat = np.load(
        savedir + 'nf{:d}_nh{:d}_lr{:.4f}_kfold{:d}_probability_classchange_randomwalk_{:d}x{:d}_test_true_and_predicted'
                  '.npz'.format(
            int(nf), int(nh), float(lr), int(kfold), int(k), int(k)))
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

    if k == 3 or k == 4:
        datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050'
                                   '_2\\'
                                   .format(k, k))
        data = np.load(datapath + '\\data_prek_xy_train_trainraw_test.npz')
    else:
        datapath = os.path.dirname('C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050'
                                   '\\'
                                   .format(k, k))
        data = np.load(datapath + '\\data_prek_xy_train_trainraw_test_{:d}x{:d}.npz'.format(k, k))

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

def load_data_kfolds(nf, nh, lr, kp=k):
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

# load_data_kfolds(20, 2, 0.0020)
dat = np.load('..\\MetaCombi\\results\\modescaling\\probability_classchange_randomwalk_{:d}x{:d}_test.npz'.format(k, k))
OtoO = dat['OtoO']
OtoP = dat['OtoP']
PtoO = dat['PtoO']
PtoP = dat['PtoP']

p_OtoP = dat['p_OtoP']
p_PtoO = dat['p_PtoO']
p_PtoP = dat['p_PtoP']
p_OtoO = dat['p_OtoO']


# "random guesser"
# rg_tO = np.random.choice([0, 1], (Number_of_O, Nsteps+1), p=[0.9, 0.1])
# rg_tP = np.random.choice([0, 1], (Number_of_P, Nsteps+1), p=[0.9, 0.1])
# # rg_tO = np.random.randint(0, 2, (Number_of_O, Nsteps + 1))
# # rg_tP = np.random.randint(0, 2, (Number_of_P, Nsteps + 1))
# ind_tPpP = np.argwhere(rg_tP[:, 0] == 1)
# ind_tPpO = np.argwhere(rg_tP[:, 0] == 0)
# ind_tOpP = np.argwhere(rg_tO[:, 0] == 1)
# ind_tOpO = np.argwhere(rg_tO[:, 0] == 0)
# rg_tPpPtopP = np.sum(rg_tP[ind_tPpP[:, 0], 1:], axis=0)
# rg_tPpPtopO = len(ind_tPpP) - rg_tPpPtopP
# rg_tPpOtopP = np.sum(rg_tP[ind_tPpO[:, 0], 1:], axis=0)
# rg_tPpOtopO = len(ind_tPpO) - rg_tPpOtopP
# rg_tOpPtopP = np.sum(rg_tO[ind_tOpP[:, 0], 1:], axis=0)
# rg_tOpPtopO = len(ind_tOpP) - rg_tOpPtopP
# rg_tOpOtopP = np.sum(rg_tO[ind_tOpO[:, 0], 1:], axis=0)
# rg_tOpOtopO = len(ind_tOpO) - rg_tOpOtopP
#
# rg_p_tPpPtopP = np.divide(rg_tPpPtopP, np.add(rg_tPpPtopP, rg_tPpPtopO))
# rg_p_tPpOtopP = np.divide(rg_tPpOtopP, np.add(rg_tPpOtopP, rg_tPpOtopO))
# rg_p_tOpPtopP = np.divide(rg_tOpPtopP, np.add(rg_tOpPtopP, rg_tOpPtopO))
# rg_p_tOpOtopP = np.divide(rg_tOpOtopP, np.add(rg_tOpOtopP, rg_tOpOtopO))
#
def plot_panal():
    steps = np.arange(1, Nsteps + 1)
    #
    p_PtoP_anal = np.power(5/6, steps) + np.multiply(1-np.power(5/6, steps-1),
                                                     (PtoO[0] + PtoP[0]) / (OtoO[0] + OtoP[0] + PtoO[0] +PtoP[0]))
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    f, ax = plt.subplots()
    cma = plt.cm.get_cmap('tab10')
    # ax.plot(np.arange(1, Nsteps + 1), p_tPtotP, c='g', label='tP to tP')
    # ax.plot(np.arange(1, Nsteps + 1), p_tOtotP, c='r', label='tO to tP')
    # ax.errorbar(np.arange(1, Nsteps + 1), p_PtoP, yerr=np.sqrt(np.multiply(p_PtoP, 1-p_PtoP)), fmt='o', c=cma.colors[2],
    #             label='P to P')
    # ax.errorbar(np.arange(1, Nsteps + 1), p_OtoP, yerr=np.sqrt(np.multiply(p_OtoP, 1-p_OtoP)), fmt='o', c=cma.colors[0],
    #             label='O to P')
    # ax.fill_between(steps, p_PtoP - np.sqrt(np.multiply(p_PtoP, 1-p_PtoP)), p_PtoP + np.sqrt(np.multiply(p_PtoP, 1-p_PtoP)),
    #                 color=cma.colors[2], alpha=0.2)
    ax.plot(np.arange(1, Nsteps + 1), p_PtoP, '.-', c=cma.colors[2],
                label='P to P')
    # ax.fill_between(steps, p_OtoP - np.sqrt(np.multiply(p_OtoP, 1-p_OtoP)), p_OtoP + np.sqrt(np.multiply(p_OtoP, 1-p_OtoP)),
    #                 color=cma.colors[0], alpha=0.2)
    ax.plot(np.arange(1, Nsteps + 1), p_OtoP, '.-', c=cma.colors[0],
                label='O to P')
    ax.plot(steps, p_PtoP_anal, c=cma.colors[1], label='MC approx')
    ax.set_xlabel('$s$')
    ax.set_ylabel(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$')
    # ax.set_title('nf {:d} \t nh {:d}'.format(nf, nh))
    ax.set_ylim([-0.05, 1.05])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_PtoP_OtoP_MCapprox_TestSet_k{:d}.pdf'.format(k),
                facecolor=f.get_facecolor())
    plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_PtoP_OtoP_MCapprox_TestSet_k{:d}.svg'.format(k),
                facecolor=f.get_facecolor())
    plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_PtoP_OtoP_MCapprox_TestSet_k{:d}.png'.format(k), dpi=400,
                facecolor=f.get_facecolor())
    plt.show()
    plt.close()
#
#
#
# f, ax = plt.subplots()
# cma = plt.cm.get_cmap('tab20c')
# ax.fill_between(np.arange(1, Nsteps+1), p_tPpPtopP - var_tPpPtopP, p_tPpPtopP + var_tPpPtopP, alpha=0.2, color=cma.colors[0])
# ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP, c=cma.colors[0], label='tPpP to pP')
# ax.fill_between(np.arange(1, Nsteps+1), p_tPtotP - var_tPtotP, p_tPtotP + var_tPtotP, alpha=0.2, color=cma.colors[4])
# ax.plot(np.arange(1, Nsteps + 1), p_tPtotP, c=cma.colors[4], label='tP to tP')
# # ax.fill_between(np.arange(1, Nsteps+1), p_tOpOtopP - var_tOpOtopP, p_tOpOtopP + var_tOpOtopP, alpha=0.2, color=cma.colors[4])
# # ax.plot(np.arange(1, Nsteps + 1), p_tOpOtopP, c=cma.colors[4], label='tOpO to pP')
# # ax.fill_between(np.arange(1, Nsteps+1), p_tOtotP - var_tOtotP, p_tOtotP + var_tOtotP, alpha=0.2, color=cma.colors[5])
# # ax.plot(np.arange(1, Nsteps + 1), p_tOtotP, c=cma.colors[5], label='tO to tP')
# ax.fill_between(np.arange(1, Nsteps+1), p_tOpPtopP - var_tOpPtopP, p_tOpPtopP + var_tOpPtopP, alpha=0.2, color=cma.colors[8])
# ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP, c=cma.colors[8], label='tOpP to pP')
# # ax.plot(np.arange(1, Nsteps+1), rg_p_tPpPtopP, c='r', label='rg tPpPtopP')
# ax.set_xlabel('steps')
# ax.set_ylabel('p')
# ax.set_title('nf {:d} \t nh {:d}'.format(nf, nh))
# ax.set_yscale('log')
# # ax.set_ylim([-0.05, 1.05])
# plt.legend(loc='best')
#
# plt.show()
# plt.close()
#
# "analytic expressions for tPpP to pP and tOpP to pP"
steps = np.arange(1, Nsteps + 1)
#
# p_tPpPtopP_anal = np.power(p_tPpPtopP[0], steps) + np.multiply(1-np.power(p_tPpPtopP[0], steps-1),
#                                                                N_pP[0] / (N_pP[0] + N_pO[0]))
# p_tOpPtopP_anal = np.power(p_tOpPtopP[0], steps) + np.multiply(1-np.power(p_tOpPtopP[0], steps-1),
#                                                                N_pP[0] / (N_pP[0] + N_pO[0]))

# load_data(20, 10, 0.0020, 9)
def plot_ptrue():
    for k in range(3, 9):
        Nsteps = k*k
        steps = np.arange(1, Nsteps + 1)
        #
        slope_opt, slope_cov = curve_fit(slope_fit, steps, p_PtoP)
        plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
        f, ax = plt.subplots()
        cma = plt.cm.get_cmap('tab10')
        # ax.plot(np.arange(1, Nsteps + 1), p_tPtotP, c='g', label='tP to tP')
        # ax.plot(np.arange(1, Nsteps + 1), p_tOtotP, c='r', label='tO to tP')
        # ax.errorbar(np.arange(1, Nsteps + 1), p_PtoP, yerr=np.sqrt(np.multiply(p_PtoP, 1-p_PtoP)), fmt='o', c=cma.colors[2],
        #             label='P to P')
        # ax.errorbar(np.arange(1, Nsteps + 1), p_OtoP, yerr=np.sqrt(np.multiply(p_OtoP, 1-p_OtoP)), fmt='o', c=cma.colors[0],
        #             label='O to P')
        # ax.fill_between(steps, p_PtoP - np.sqrt(np.multiply(p_PtoP, 1-p_PtoP)), p_PtoP + np.sqrt(np.multiply(p_PtoP, 1-p_PtoP)),
        #                 color=cma.colors[2], alpha=0.2)
        ax.plot(np.arange(1, Nsteps + 1), p_PtoP, '.-', c=cma.colors[2],
                    label='P to P')
        # ax.fill_between(steps, p_OtoP - np.sqrt(np.multiply(p_OtoP, 1-p_OtoP)), p_OtoP + np.sqrt(np.multiply(p_OtoP, 1-p_OtoP)),
        #                 color=cma.colors[0], alpha=0.2)
        ax.plot(np.arange(1, Nsteps + 1), p_OtoP, '.-', c=cma.colors[0],
                    label='O to P')
        ax.plot(steps, p_PtoP_anal, c=cma.colors[1], label='MC approx')
        ax.set_xlabel('$s$')
        ax.set_ylabel(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$')
        # ax.set_title('nf {:d} \t nh {:d}'.format(nf, nh))
        ax.set_ylim([-0.05, 1.05])
        plt.legend(loc='best')
        plt.tight_layout()
        plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_PtoP_OtoP_MCapprox_TestSet_k{:d}.pdf'.format(k),
                    facecolor=f.get_facecolor())
        plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_PtoP_OtoP_MCapprox_TestSet_k{:d}.svg'.format(k),
                    facecolor=f.get_facecolor())
        plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_PtoP_OtoP_MCapprox_TestSet_k{:d}.png'.format(k), dpi=400,
                    facecolor=f.get_facecolor())
        plt.show()
        plt.close()

def cm_to_inch(x):
    return x / 2.54

def plot_p_kfolds():
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    if k == 5:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        lr_list = np.array([0.0020, 0.0010, 0.0030, 0.0030, 0.0030, 0.0030, 0.0050, 0.0030,
                        0.0030, 0.0020, 0.0040, 0.0050, 0.0040, 0.0050, 0.0040, 0.0020,
                        0.0030, 0.0030])
    elif k == 6:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        lr_list = np.array([0.0020, 0.0020, 0.0010, 0.0010, 0.0020, 0.0020, 0.0020, 0.0020,
                            0.0020, 0.0030, 0.0030, 0.0030, 0.0030, 0.0020, 0.0020, 0.0020,
                            0.0020, 0.00200])
    elif k == 7:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        lr_list = np.array([0.0001, 0.0030, 0.0020, 0.0030, 0.0030, 0.0020, 0.0030, 0.0010, 0.0030, 0.0020, 0.0030,
                            0.0040, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020])
    elif k == 4:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        lr_list = np.array([0.0010, 0.0030, 0.0040, 0.0040, 0.0030, 0.0030, 0.0050, 0.0040, 0.0040, 0.0050, 0.0050,
                            0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050])
    elif k == 3:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        lr_list = np.array([0.0040, 0.0050, 0.0040, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050,
                            0.0050, 0.0050, 0.0040, 0.0040, 0.0040, 0.0050, 0.0040, 0.0040, 0.0040])
    elif k == 8:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        lr_list = np.array([0.0001, 0.0001, 0.0010, 0.0010, 0.0010, 0.0010, 0.0020, 0.0010, 0.0010, 0.0010, 0.0010,
                            0.0010, 0.0010, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020])
    # kfold_list = np.array([4, 9, 3, 0, 9, 5, 8, 1, 4, 7, 1, 4, 4, 5, 5, 6, 2, 6])
    # inds = np.array([0, 4, 17])
    # nh_list = nh_list[inds]
    # lr_list = lr_list[inds]
    # kfold_list = kfold_list[inds]

    # f, ax = plt.subplots()
    cma = plt.cm.get_cmap('rainbow')
    slope_list = np.zeros((len(nh_list), 10))
    slope_var_list = np.zeros((len(nh_list), 10))
    for i, nh in enumerate(nh_list):
        load_data_kfolds(nf, nh, lr_list[i])
        for kfold in range(10):
            def slope_fit(x, a):
                return np.power(a, x) + (1 - np.power(a, x - 1)) * N_pP[kfold, 0] / (N_pP[kfold, 0] + N_pO[kfold, 0])

            def slope_asymp_fit(x, a, b):
                return np.power(a, x) + (1 - np.power(a, x - 1)) * b

            def slope_log_fit(x, a):
                return x * np.log(a) + np.log(1 - np.power(a, -1) * N_pP[kfold, 0] / (N_pP[kfold, 0] + N_pO[kfold, 0]))

            def slope_asymp_fit(x, a, b):
                return np.log(np.power(a, x) * (1 - b * np.power(a, -1)) + b)

            slope_opt, slope_cov = curve_fit(slope_fit, steps, p_tPpPtopP[kfold])
            # slope_asymp_opt, slope_asymp_cov = curve_fit(slope_asymp_fit, steps, p_tPpPtopP)
            asympt = N_pP[kfold, 0] / (N_pP[kfold, 0] + N_pO[kfold, 0])
            # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP[kfold] - asympt, c=cma(2. * i / (2 * len(nh_list))),
            #         label=r'$n_h=' + '{:d}'.format(nh) + '$')
            # ax.plot(np.arange(1, Nsteps + 1), slope_fit(steps, *slope_opt) - asympt,
            #         c=cma((2. * i + 1) / (2 * len(nh_list))), label=r'$n_h=' + '{:d}'.format(nh) + '$ fit')
            slope_list[i, kfold] = slope_opt[0]
            slope_var_list[i, kfold] = slope_cov[0, 0]
            # slope_list.append(slope_opt[0])
            # slope_var_list.append(slope_cov[0, 0])
            # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - slope_asymp_opt[1], c=cma.colors[4], label='tPpP to pP - b')
            # ax.plot(np.arange(1, Nsteps + 1), slope_asymp_fit(steps, *slope_asymp_opt) - slope_asymp_opt[1], c=cma.colors[5], label=r'$a^x + (1-a^{x-1})b$')
            # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP, c=cma.colors[8], label='tOpP to pP')
            # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP_anal, c=cma.colors[9], label='tOpP to pP analytic')

            print(*slope_opt)

        # print(*slope_asymp_opt)
    # cma = plt.cm.get_cmap('tab20c')
    # ax.plot(steps, p_PtoP - Number_of_P / (Number_of_O + Number_of_P), '-.', c=cma.colors[16], label='true')
    # ax.plot(steps, np.power(5 / 6, steps) * (1 - ((5 / 6) ** -1) * Number_of_P / (Number_of_O + Number_of_P)), '-.',
    #         c=cma.colors[17], label='true 5/6')
    #
    # ax.set_xlabel('$s$')
    # ax.set_ylabel('$p$')
    # # ax.set_title('subtract known asymptote, fit slope only')
    # ax.set_yscale('log')
    # # ax.set_ylim([-0.05, 1.05])
    # plt.legend(loc='best')
    #
    # # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_TestSet_k{:d}_nh2_nh10_nh100.pdf'.format(k),
    # #             facecolor=f.get_facecolor())
    # # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_TestSet_k{:d}_nh2_nh10_nh100.svg'.format(k),
    # #             facecolor=f.get_facecolor())
    # # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_TestSet_k{:d}_nh2_nh10_nh100.png'.format(k), dpi=400,
    # #             facecolor=f.get_facecolor())
    # plt.show()
    #
    # plt.close()

    def alpha_fit_pl(x, a, b, c):
        return np.add(a * np.power(x, -b), c)
    def alpha_fit_exp(x, a, b, c):
        return a * np.exp(-x * b) + c
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    f, ax = plt.subplots()
    slope_opt, slope_cov = curve_fit(slope_fit, steps, p_PtoP)
    print(*slope_opt)

    slope_kfoldavg = np.mean(slope_list, axis=1)
    # slope_kfoldvar = np.sum(np.square(1./10.) * slope_var_list, axis=1)
    slope_kfoldvar = np.var(slope_list, axis=1)
    np.savez(savedir + '\\alpha_fit_nf20_mean_var.npz'.format(k, k),
             mean=slope_kfoldavg,
             var=slope_kfoldvar)
    pl_fit, pl_cov = curve_fit(alpha_fit_pl, nh_list, slope_kfoldavg, sigma=np.sqrt(slope_kfoldvar))
    exp_fit, exp_cov = curve_fit(alpha_fit_exp, nh_list, slope_kfoldavg, sigma=np.sqrt(slope_kfoldvar))
    print(*exp_fit)
    ax.fill_between(nh_list, np.add(slope_kfoldavg, -1.*np.sqrt(slope_kfoldvar)),
                    np.add(slope_kfoldavg, np.sqrt(slope_kfoldvar)), alpha=0.2)
    ax.plot(nh_list, slope_kfoldavg, '.-')
    ax.plot(np.linspace(2, 100), alpha_fit_pl(np.linspace(2, 100), *pl_fit), c='r', label='power law')
    ax.plot(np.linspace(2, 100), alpha_fit_exp(np.linspace(2, 100), *exp_fit), c='b', label='exponential')
    ax.set_xlabel('$n_h$')
    ax.set_ylabel(r'$\langle \alpha \rangle$')
    # ax.set_title('slope fit only')
    plt.legend()
    plt.tight_layout()
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet_k{:d}.pdf'.format(k),
    #             facecolor=f.get_facecolor())
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet_k{:d}.svg'.format(k),
    #             facecolor=f.get_facecolor())
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet_k{:d}.png'.format(k), dpi=400,
    #             facecolor=f.get_facecolor())
    plt.show()
    plt.close()

def plot_p_foldavg():
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    # mpl.rcParams['axes.labelsize'] = 24
    # mpl.rcParams['xtick.labelsize'] = 24
    # mpl.rcParams['ytick.labelsize'] = 24
    # mpl.rcParams['figure.subplot.left'] = 0.25
    # mpl.rcParams['figure.subplot.right'] = 0.95
    # mpl.rcParams['figure.subplot.bottom'] = 0.15
    # mpl.rcParams['figure.subplot.top'] = 0.95
    # # mpl.rcParams['figure.figsize']=[ 8,8]
    # # mpl.rcParams['font.family']='Times'
    # mpl.rcParams['axes.labelsize'] = 42
    # mpl.rcParams['lines.linewidth'] = 5
    # mpl.rcParams['axes.linewidth'] = 3
    # mpl.rcParams['xtick.major.size'] = 0 * 8
    # mpl.rcParams['xtick.major.width'] = 3
    # mpl.rcParams['xtick.major.pad'] = 2
    # mpl.rcParams['ytick.major.pad'] = 2
    # mpl.rcParams['ytick.major.size'] = 0 * 8
    # mpl.rcParams['ytick.major.width'] = 3
    # mpl.rcParams['xtick.minor.size'] = 0  # 7.5
    # mpl.rcParams['xtick.minor.width'] = 0
    # mpl.rcParams['ytick.minor.size'] = 0  # 7.5
    # mpl.rcParams['ytick.minor.width'] = 0
    # mpl.rcParams['ytick.direction'] = "in"
    # mpl.rcParams['xtick.direction'] = "in"
    cm = plt.get_cmap('Greens')
    cma = plt.get_cmap('tab20c')
    c_red = 'tab:red'
    cmv = plt.get_cmap('viridis')
    cmtab20 = plt.get_cmap('tab20')
    alpha_c_list = np.zeros(6)
    # f, ax = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(4.0)))
    xlength = 3.32
    ylength = 0.3
    # f = plt.figure(1, figsize=(cm_to_inch(4.7), cm_to_inch(4.)))
    # xoffset = 0.25*(xlength / 4.7)
    # yoffset = 0.18*(xlength / 4.)
    # figfracx = (3.85 - xoffset*4.7) / 4.7
    # figfracy = figfracx*4.7 / 4.
    f = plt.figure(1, figsize=(cm_to_inch(4.3), cm_to_inch(4.3)))
    xoffset = 0.22 * (xlength / 4.3)
    yoffset = 0.18 * (xlength / 4.3)
    figfracx = (3.5 - xoffset * 4.3) / 4.3
    figfracy = figfracx * 4.3 / 4.3
    # figfracy = 0.7
    ax = f.add_axes([xoffset, yoffset, figfracx, figfracy])
    caxg = f.add_axes(
        [xoffset + figfracx + 0.01, yoffset, (0.3/4.3)-0.01, figfracy])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    klist_odd = np.array([3, 5, 7])
    klist_even = np.array([4, 6, 8])
    karange = np.linspace(3, 8, 1000)
    alpha_est_odd = np.divide((karange - 2), karange)
    alpha_est_even = np.divide((karange - 1), karange)
    ax.plot(karange, alpha_est_odd, ls='dashed', c='black', zorder=1)
    ax.plot(karange, alpha_est_even, ls='dashdot', c='black', zorder=1)
    # f3 = plt.figure(3, figsize=(cm_to_inch(3.85), cm_to_inch(3.85)))
    # xoffset3 = 0.25 * (xlength / 3.85)
    # yoffset3 = 0.18 * (xlength / 3.85)
    # figfracx3 = (3.85 - xoffset3 * 3.85) / 3.85
    # figfracy3 = figfracx3 * 3.85 / 3.85
    f3 = plt.figure(3, figsize=(cm_to_inch(8.6), cm_to_inch(8.6)))
    xoffset3 = 0.22 * (xlength / 8.6)
    yoffset3 = 0.18 * (xlength / 8.6)
    figfracx3 = (8.5 - xoffset3 * 8.6) / 8.6
    figfracy3 = figfracx3 * 8.6 / 8.6
    ax3 = f3.add_axes([xoffset3, yoffset3, figfracx3, figfracy3])
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # f3, ax3 = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(4.0)))
    ax3.set_ylabel(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$', labelpad=0)
    ax3.set_xlabel('$s$', labelpad=0)
    ax.set_xticks([3, 4, 5, 6, 7, 8])
    ax.set_xlim([2.5, 8.5])
    ax.set_ylim([0.3, 0.9])
    for k in range(3, 9):
        steps = np.arange(1, k*k + 1)
        Nsteps = k*k
        dat = np.load('..\\metacombi\\results\\modescaling\\probability_classchange_randomwalk_{:d}x{:d}_test.npz'.format(k, k))
        OtoO = dat['OtoO']
        OtoP = dat['OtoP']
        PtoO = dat['PtoO']
        PtoP = dat['PtoP']

        p_OtoP = dat['p_OtoP']
        p_PtoO = dat['p_PtoO']
        p_PtoP = dat['p_PtoP']
        p_OtoO = dat['p_OtoO']
        if k < 6:
            res = np.loadtxt(
                '..\\metacombi\\results\\modescaling\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}_fixn4.txt'.
                format(k, k),
                delimiter=',')
        elif k == 7:
            res = np.loadtxt(
                '..\\metacombi\\results\\modescaling\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended.txt'.
                    format(k, k),
                delimiter=',')

        else:
            res = np.loadtxt(
                '..\\metacombi\\results\\modescaling\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}.txt'.
                    format(k, k),
                delimiter=',')
        beta = np.sum(res[:, 1] == 1) / np.shape(res)[0]


        def slope_fit(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * (beta)


        if k % 2:
            'odd'
            p0 = np.array([(k - 2.) / k])
        else:
            'even'
            p0 = np.array([(k - 1.) / k])
        slope_opt, slope_cov = curve_fit(slope_fit, steps, p_PtoP, p0=p0)

        if k == 5:
            nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90])
            lr_list = np.array([0.0020, 0.0010, 0.0030, 0.0030, 0.0030, 0.0030, 0.0050, 0.0030,
                            0.0030, 0.0020, 0.0040, 0.0050, 0.0040, 0.0050, 0.0040, 0.0020,
                            0.0030])
        elif k == 6:
            nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            lr_list = np.array([0.0020, 0.0020, 0.0010, 0.0010, 0.0020, 0.0020, 0.0020, 0.0020,
                                0.0020, 0.0030, 0.0030, 0.0030, 0.0030, 0.0020, 0.0020, 0.0020,
                                0.0020, 0.00200])
        elif k == 7:
            nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            lr_list = np.array([0.0001, 0.0030, 0.0020, 0.0030, 0.0030, 0.0020, 0.0030, 0.0010, 0.0030, 0.0020, 0.0030,
                                0.0040, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020])
        elif k == 4:
            nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90])
            lr_list = np.array([0.0010, 0.0030, 0.0040, 0.0040, 0.0030, 0.0030, 0.0050, 0.0040, 0.0040, 0.0050, 0.0050,
                                0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050])
        elif k == 3:
            nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            # lr_list = np.array([0.0050, 0.0050, 0.0040, 0.0030, 0.0040, 0.0050, 0.0040, 0.0040, 0.0020,
            #                     0.0050, 0.0040, 0.0040, 0.0030, 0.0030, 0.0040, 0.0030, 0.0030])
            lr_list = np.array([0.0040, 0.0050, 0.0040, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050, 0.0050,
                                0.0050, 0.0050, 0.0040, 0.0040, 0.0040, 0.0050, 0.0040, 0.0040, 0.0040])

        elif k == 8:
            nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
            lr_list = np.array([0.0001, 0.0001, 0.0010, 0.0010, 0.0010, 0.0010, 0.0020, 0.0010, 0.0010, 0.0010, 0.0010,
                                0.0010, 0.0010, 0.0020, 0.0020, 0.0020, 0.0020, 0.0020])
        # kfold_list = np.array([4, 9, 3, 0, 9, 5, 8, 1, 4, 7, 1, 4, 4, 5, 5, 6, 2, 6])
        # inds = np.array([0, 4, 17])
        # nh_list = nh_list[inds]
        # lr_list = lr_list[inds]
        # kfold_list = kfold_list[inds]

        # f, ax = plt.subplots()
        if k == 3:
            nf = 20
        else:
            nf = 20

        if k % 2:
            "odd"
            p0 = (k - 2) / k
        else:
            "even"
            p0 = (k - 1) / k
        # cma = plt.cm.get_cmap('rainbow')
        slope_list = np.zeros((len(nh_list), 10))
        slope_var_list = np.zeros((len(nh_list), 10))
        p_list = []
        p_list_var = []

        # f2 = plt.figure(2, figsize=(cm_to_inch(3.9), cm_to_inch(4.)))
        # xoffset2 = 0.22*(xlength / 3.9)
        # yoffset2 = 0.18*(xlength / 4.)
        # figfracx2 = figfracy * 4. / 3.9
        # figfracy2 = figfracy
        f2 = plt.figure(2, figsize=(cm_to_inch(4.3), cm_to_inch(4.3)))
        xoffset2 = 0.22 * (xlength / 4.3)
        yoffset2 = 0.18 * (xlength / 4.3)
        figfracx2 = figfracy * 4.3 / 4.3
        figfracy2 = figfracy
        ax2 = f2.add_axes([xoffset2, yoffset2, figfracx2, figfracy2])
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_xticks([0, 10, 20, 30, 40])
        # f2, ax2 = plt.subplots(figsize=(cm_to_inch(4.3), cm_to_inch(4.3)))
        ax2.set_ylim([0.0, 1.0])
        ax2.set_xlim([0, 40])
        color = 'tab:orange'
        ax2.set_xlabel('$s$')
        # ax2.set_ylabel('$p(X_s = P | X_0 = P)$', color=color)
        # ax2.set_ylabel(r'$p_{P \rightarrow P}(s)$', color=color)
        ax2.set_ylabel(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$')
        # ax1.plot(t, data1, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        # axt2 = ax2.twinx()
        # color = 'tab:purple'
        # # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$', color=color)  # we already handled the x-label with ax1
        # axt2.set_ylabel(r'$\langle \tilde{p}_{P \rightarrow P} \rangle(s)$', color=color)
        # axt2.tick_params(axis='y', labelcolor=color)

        if k % 2:
            ax2.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
                    c=c_red, zorder=2, markersize=20, linewidth=.5)
            ax2.plot(steps, p_PtoP, marker='x', c=c_red, linestyle='None', zorder=3, markersize=2, linewidth=.5)
            # ax3.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
            #          c=cmtab20(2*(k-3)+1), zorder=2)
            # ax3.plot(steps, p_PtoP, marker=(k, 0, 0), c=cmtab20(2*(k-3)+1), linestyle='None', zorder=3, label=k)
        else:
            ax2.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
                    c=c_red, zorder=2, markersize=20, linewidth=.5)
            ax2.plot(steps, p_PtoP, marker='x', c=c_red, linestyle='None', zorder=3, markersize=2, linewidth=.5)
        ax3.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
                 c=cmtab20(2*(k-3)+1), zorder=2, markersize=2, linewidth=.5)
        ax3.plot(steps, p_PtoP, marker=(k, 0, 0), c=cmtab20(2*(k-3)), linestyle='None', zorder=3, label=k,
                 markersize=2, linewidth=.5)
        f2.tight_layout()
        f2.savefig('.\\figures\\p_tPtotP_k{:d}.pdf'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tPtotP_k{:d}.svg'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tPtotP_k{:d}.png'.format(k), dpi=400,
                   facecolor=f2.get_facecolor())

        # instantiate a second axes that shares the same x-axis
        # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$')
        for i, nh in enumerate(nh_list):
            load_data_kfolds(nf, nh, lr_list[i], kp=k)


            p_list.append(np.nanmean(p_tPpPtopP, axis=0))
            p_list_var.append(np.nanvar(p_tPpPtopP, axis=0))
            for kfold in range(10):
                def slope_fit(x, a):
                    return np.power(a, x) + (1 - np.power(a, x - 1)) * N_pP[kfold, 0] / (N_pP[kfold, 0] + N_pO[kfold, 0])



                def slope_asymp_fit(x, a, b):
                    return np.power(a, x) + (1 - np.power(a, x - 1)) * b

                def slope_log_fit(x, a):
                    return x * np.log(a) + np.log(1 - np.power(a, -1) * N_pP[kfold, 0] / (N_pP[kfold, 0] + N_pO[kfold, 0]))

                def slope_asymp_fit(x, a, b):
                    return np.log(np.power(a, x) * (1 - b * np.power(a, -1)) + b)

                slope_opt, slope_cov = curve_fit(slope_fit, steps, p_tPpPtopP[kfold])
                # slope_asymp_opt, slope_asymp_cov = curve_fit(slope_asymp_fit, steps, p_tPpPtopP)
                asympt = N_pP[kfold, 0] / (N_pP[kfold, 0] + N_pO[kfold, 0])
                # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP[kfold] - asympt, c=cma(2. * i / (2 * len(nh_list))),
                #         label=r'$n_h=' + '{:d}'.format(nh) + '$')
                # ax.plot(np.arange(1, Nsteps + 1), slope_fit(steps, *slope_opt) - asympt,
                #         c=cma((2. * i + 1) / (2 * len(nh_list))), label=r'$n_h=' + '{:d}'.format(nh) + '$ fit')
                slope_list[i, kfold] = slope_opt[0]
                slope_var_list[i, kfold] = slope_cov[0, 0]
                # slope_list.append(slope_opt[0])
                # slope_var_list.append(slope_cov[0, 0])
                # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - slope_asymp_opt[1], c=cma.colors[4], label='tPpP to pP - b')
                # ax.plot(np.arange(1, Nsteps + 1), slope_asymp_fit(steps, *slope_asymp_opt) - slope_asymp_opt[1], c=cma.colors[5], label=r'$a^x + (1-a^{x-1})b$')
                # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP, c=cma.colors[8], label='tOpP to pP')
                # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP_anal, c=cma.colors[9], label='tOpP to pP analytic')

                print(*slope_opt)

            # ax2.plot(steps, p_list[i], '.-', c=cm(1- i/len(nh_list)))
            ax2.plot(steps, p_list[i], '.', c=cmv(nh/np.amax(nh_list)), zorder=1)
            ax2.set_ylim([-0.05, 1.05])
            f2.tight_layout()
            f2.savefig('.\\figures\\p_tPpPtopP_k{:d}_uptonh{:d}.pdf'.format(k, nh), facecolor=f2.get_facecolor())
            f2.savefig('.\\figures\\p_tPpPtopP_k{:d}_uptonh{:d}.svg'.format(k, nh), facecolor=f2.get_facecolor())
            f2.savefig('.\\figures\\p_tPpPtopP_k{:d}_uptonh{:d}.png'.format(k, nh), dpi=400,
                       facecolor=f2.get_facecolor())

        ax2.set_ylim([0.0, 1.0])
        # ax2.set_xlabel('$s$')
        # ax2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$')
        f2.tight_layout()
        # plt.show()
        # ax2.set_title('{:d}x{:d}'.format(k, k))
        custom_lines = [Line2D([0], [0], color=cmv(0), lw=.5)]
        custom_lines.append(Line2D([0], [0], color='tab:red', lw=.5))
        # custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
        custom_labels = [r'$\bar{\rho}_{\mathrm{C} \rightarrow \mathrm{C}}$']
        custom_labels.append(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$')
        ax2.legend(custom_lines, custom_labels, loc='lower left')
        f2.savefig('.\\figures\\p_tPpPtopP_nh_k{:d}_nocbar.pdf'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tPpPtopP_nh_k{:d}_nocbar.svg'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tPpPtopP_nh_k{:d}_nocbar.png'.format(k), facecolor=f2.get_facecolor(), dpi=400)
        norm = mpl.colors.Normalize(vmin=0, vmax=np.amax(nh_list), clip=False)
        cb = f2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmv), ax=ax2, pad=0.03)
        cb.ax.set_title('$n_h$')
        f2.tight_layout()
        # plt.show()
        # ax2.set_title('{:d}x{:d}'.format(k, k))
        f2.savefig('.\\figures\\p_tPpPtopP_nh_k{:d}.pdf'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tPpPtopP_nh_k{:d}.svg'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tPpPtopP_nh_k{:d}.png'.format(k), facecolor=f2.get_facecolor(), dpi=400)
        f2.clear()
        def slope_fit_class(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * N_tP[0] / (N_tP[0] + N_tO[0])

        slope_opt, slope_cov = curve_fit(slope_fit_class, steps, p_tPtotP[0], p0=p0)
        print('alpha class:')
        print('alpha class:')
        print(*slope_opt)
        alpha_c = slope_opt[0]
        alpha_c_list[k-3] = alpha_c

        slope_kfoldavg = np.mean(slope_list, axis=1)
        slope_kfoldvar = np.var(slope_list, axis=1)
        # ax.scatter(np.full(len(slope_kfoldavg), k), 1-slope_kfoldavg,
        #             color=cm((slope_kfoldavg-np.amin(slope_kfoldavg))/(np.amax(slope_kfoldavg) - np.amin(slope_kfoldavg))),
        #             marker=(k, 0, 0), linestyle='None')
        ax.scatter(np.full(len(slope_kfoldavg), k), slope_kfoldavg, edgecolors=cmv(nh_list/np.amax(nh_list)),
                   facecolors='None', marker=(k, 0, 0), linestyle='None', s=20, linewidths=.5
                   )
        # ax.scatter(k, 1-alpha_c, marker=(k, 0, 0), edgecolor='tab:red',
        #             facecolor='None', linestyle='None', s=40, linewidths=1.5
        #             )
        ax.hlines(alpha_c, xmin=k - .5, xmax=k + .5, colors='tab:red', linewidths=.5)
        # ax.errorbar(k, 1 - alpha_c, yerr=np.sqrt(slope_cov[0]), marker=(k, 0, 0), edgecolor='tab:red',
        #            facecolor='None', linestyle='None', s=40, linewidths=1.5
        #            )
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$\alpha$')

    # ax.set_title('slope fit only')
    # plt.legend()
    custom_lines = [Line2D([0], [0], color=cmv(0), lw=.5)]
    custom_lines.append(Line2D([0], [0], color='tab:red', lw=.5))
    # custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
    custom_labels = [r'$\bar{\alpha}$']
    custom_labels.append(r'$\alpha$')
    ax.legend(custom_lines, custom_labels)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylim([0.3, 1.1])
    # f.tight_layout()
        # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet_k{:d}.pdf'.format(k),
        #             facecolor=f.get_facecolor())
        # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet_k{:d}.svg'.format(k),
        #             facecolor=f.get_facecolor())
        # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet_k{:d}.png'.format(k), dpi=400,
        #             facecolor=f.get_facecolor())
    f.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), cax=caxg)
    caxg.set_title('$n_h$')
    f.savefig('.\\figures\\p_tPpPtopP_slopefit_nh.pdf', facecolor=f.get_facecolor())
    f.savefig('.\\figures\\p_tPpPtopP_slopefit_nh.svg', facecolor=f.get_facecolor())
    f.savefig('.\\figures\\p_tPpPtopP_slopefit_nh.png', facecolor=f.get_facecolor(), dpi=400)
    np.save('.\\alpha_c_list.npy', np.array(alpha_c_list))

    ax3.set_xscale('log')
    ax3.legend()
    f3.tight_layout()
    f3.savefig('.\\figures\\p_tPtotP_pfit_k345678.pdf', facecolor=f.get_facecolor())
    f3.savefig('.\\figures\\p_tPtotP_pfit_k345678.png', facecolor=f.get_facecolor(), dpi=400)
    f3.savefig('.\\figures\\p_tPtotP_pfit_k345678.svg', facecolor=f.get_facecolor())
    plt.show()
    plt.close()

def plot_p_bestkfold():
    if k == 5:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90])
        lr_list = np.array([0.0020, 0.0010, 0.0030, 0.0030, 0.0030, 0.0030, 0.0050, 0.0030,
                            0.0030, 0.0020, 0.0040, 0.0050, 0.0040, 0.0050, 0.0040, 0.0020,
                            0.0030])
    elif k == 6:
        nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
        lr_list = np.array([0.0020, 0.0020, 0.0010, 0.0010, 0.0020, 0.0020, 0.0020, 0.0020,
                            0.0020, 0.0030, 0.0030, 0.0030, 0.0030, 0.0020, 0.0020, 0.0020,
                            0.0020, 0.0020])
        kfold_list = np.array([4, 9, 3, 0, 9, 5, 8, 1, 4, 7, 1, 4, 4, 5, 5, 6, 2, 6])
    inds = np.array([0, 4, 17])
    nh_list = nh_list[inds]
    lr_list = lr_list[inds]
    kfold_list = kfold_list[inds]
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    f, ax = plt.subplots()
    cma = plt.cm.get_cmap('rainbow')
    slope_list = []
    slope_var_list = []
    for i, nh in enumerate(nh_list):
        load_data(20, nh, lr_list[i], kfold_list[i])
        def slope_fit(x, a):
            return np.power(a, x) + (1-np.power(a, x-1)) * N_pP[0]/ (N_pP[0] + N_pO[0])

        def slope_asymp_fit(x, a, b):
            return np.power(a, x) + (1-np.power(a, x-1)) * b

        def slope_log_fit(x, a):
            return x * np.log(a) + np.log(1 - np.power(a, -1) * N_pP[0]/(N_pP[0] + N_pO[0]))

        def slope_asymp_fit(x, a, b):
            return np.log(np.power(a, x)*(1-b*np.power(a, -1)) + b)
        slope_opt, slope_cov = curve_fit(slope_fit, steps, p_tPpPtopP)
        # slope_asymp_opt, slope_asymp_cov = curve_fit(slope_asymp_fit, steps, p_tPpPtopP)
        asympt = N_pP[0] / (N_pP[0] + N_pO[0])
        ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - asympt, '.', c=cma(2.*i/(2*len(nh_list))), label=r'$n_h='+'{:d}'.format(nh)+'$')
        ax.plot(np.arange(1, Nsteps + 1), slope_fit(steps, *slope_opt)-asympt, c=cma((2.*i)/(2*len(nh_list))))
        slope_list.append(slope_opt[0])
        slope_var_list.append(slope_cov[0, 0])
        # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - slope_asymp_opt[1], c=cma.colors[4], label='tPpP to pP - b')
        # ax.plot(np.arange(1, Nsteps + 1), slope_asymp_fit(steps, *slope_asymp_opt) - slope_asymp_opt[1], c=cma.colors[5], label=r'$a^x + (1-a^{x-1})b$')
        # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP, c=cma.colors[8], label='tOpP to pP')
        # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP_anal, c=cma.colors[9], label='tOpP to pP analytic')


        print(*slope_opt)

        # print(*slope_asymp_opt)
    cma = plt.cm.get_cmap('tab20c')
    slope_opt, slope_cov = curve_fit(slope_fit, steps, p_PtoP)
    print(*slope_opt)
    ax.plot(steps, p_PtoP - Number_of_P / (Number_of_O + Number_of_P), '.', c=cma.colors[16], label='true')
    ax.plot(steps, np.power((k-1)/k, steps) * (1 - (((k-1)/k)**-1) * Number_of_P / (Number_of_O + Number_of_P)), '-.', c=cma.colors[17])

    ax.set_xlabel('$s$')
    ax.set_ylabel(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}-\beta$')
    # ax.set_title('subtract known asymptote, fit slope only')
    ax.set_yscale('log')
    # ax.set_ylim([-0.05, 1.05])
    plt.legend(loc='best')
    plt.tight_layout()
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_TestSet_k{:d}_nh2_nh10_nh100.pdf'.format(k),
    #             facecolor=f.get_facecolor())
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_TestSet_k{:d}_nh2_nh10_nh100.svg'.format(k),
    #             facecolor=f.get_facecolor())
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_TestSet_k{:d}_nh2_nh10_nh100.png'.format(k), dpi=400,
    #             facecolor=f.get_facecolor())
    plt.show()

    plt.close()

    f, ax = plt.subplots()
    ax.fill_between(nh_list, np.add(slope_list, np.multiply(-1., np.sqrt(slope_var_list))), np.add(slope_list,  np.sqrt(slope_var_list)), alpha=0.2)
    ax.plot(nh_list, slope_list, '.-')
    ax.set_xlabel('$n_h$')
    ax.set_ylabel(r'$\alpha$')
    # ax.set_title('slope fit only')
    plt.tight_layout()
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_slope_vs_nh_TestSet_k{:d}.pdf'.format(k),
    #             facecolor=f.get_facecolor())
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_slope_vs_nh_TestSet_k{:d}.svg'.format(k),
    #             facecolor=f.get_facecolor())
    # plt.savefig('..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_fit_slope_vs_nh_TestSet_k{:d}.png'.format(k) dpi=400,
    #             facecolor=f.get_facecolor())
    plt.show()
    plt.close()

    f, ax = plt.subplots()
    cma = plt.cm.get_cmap('rainbow')
    slope_list=[]
    slope_var_list = []
    for i, nh in enumerate(nh_list):
        load_data(20, nh, lr_list[i], kfold_list[i])
        def slope_fit(x, a):
            return np.power(a, x) + (1-np.power(a, x-1)) * N_pP[0]/ (N_pP[0] + N_pO[0])

        def slope_asymp_fit(x, a, b):
            return np.power(a, x) + (1-np.power(a, x-1)) * b
        # slope_opt, slope_cov = curve_fit(slope_fit, steps, p_tPpPtopP)
        slope_asymp_opt, slope_asymp_cov = curve_fit(slope_asymp_fit, steps, p_tPpPtopP)
        asympt = N_pP[0] / (N_pP[0] + N_pO[0])
        ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - slope_asymp_opt[1], c=cma(1.*i/len(nh_list)), label=r'nh {:d}'.format(nh))
        ax.plot(np.arange(1, Nsteps + 1), slope_asymp_fit(steps, *slope_asymp_opt)-slope_asymp_opt[1], c=cma(1.*i/len(nh_list)), label=r'nh {:d} fit'.format(nh))
        # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - slope_asymp_opt[1], c=cma.colors[4], label='tPpP to pP - b')
        # ax.plot(np.arange(1, Nsteps + 1), slope_asymp_fit(steps, *slope_asymp_opt) - slope_asymp_opt[1], c=cma.colors[5], label=r'$a^x + (1-a^{x-1})b$')
        # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP, c=cma.colors[8], label='tOpP to pP')
        # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP_anal, c=cma.colors[9], label='tOpP to pP analytic')
        slope_list.append(slope_asymp_opt[0])
        slope_var_list.append(slope_asymp_cov[0, 0])


        # print(*slope_opt)

        print(*slope_asymp_opt)
    cma = plt.cm.get_cmap('tab20c')
    plt.tight_layout()
    ax.plot(steps, p_PtoP - Number_of_P / (Number_of_O + Number_of_P), '-.', c=cma.colors[16], label='true')
    ax.plot(steps, np.power(5/6, steps) * (1 - ((5/6)**-1) * Number_of_P / (Number_of_O + Number_of_P)), '-.', c=cma.colors[17], label='true 5/6')

    ax.set_xlabel('steps')
    ax.set_ylabel('p')
    ax.set_title('subtract fitted asymptote, fit slope & asymptote')
    ax.set_yscale('log')
    # ax.set_ylim([-0.05, 1.05])
    plt.legend(loc='best')

    plt.show()

    plt.close()

    f, ax = plt.subplots()
    ax.fill_between(nh_list, np.add(slope_list, np.multiply(-1., np.sqrt(slope_var_list))), np.add(slope_list,  np.sqrt(slope_var_list)), alpha=0.2)
    ax.plot(nh_list, slope_list, '.-')

    ax.set_xlabel('$n_h$')
    ax.set_ylabel(r'$\alpha$')
    ax.set_title('slope & asymptote fit')
    plt.show()
    plt.close()

    f, ax = plt.subplots()
    cma = plt.cm.get_cmap('rainbow')
    slope_list = []
    slope_var_list = []
    for i, nh in enumerate(nh_list):
        load_data(20, nh, lr_list[i], kfold_list[i])


        def slope_fit(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * N_pP[0] / (N_pP[0] + N_pO[0])


        def slope_asymp_fit(x, a, b):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * b


        def slope_log_fit(x, a):
            return np.log(np.power(a, x) + (1 - np.power(a, x - 1)) * N_pP[0] / (N_pP[0] + N_pO[0]))


        def slope_asymp_fit(x, a, b):
            return np.log(np.power(a, x) * (1 - b * np.power(a, -1)) + b)



        # slope_asymp_opt, slope_asymp_cov = curve_fit(slope_asymp_fit, steps, p_tPpPtopP)
        asympt = N_pP[0] / (N_pP[0] + N_pO[0])
        slope_opt, slope_cov = curve_fit(slope_log_fit, steps, np.log(p_tPpPtopP))
        ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - asympt, c=cma(1. * i / len(nh_list)), label=r'nh {:d}'.format(nh))
        ax.plot(np.arange(1, Nsteps + 1), np.exp(slope_log_fit(steps, *slope_opt)), c=cma(1. * i / len(nh_list)),
                label=r'nh {:d} fit'.format(nh))
        slope_list.append(slope_opt[0])
        slope_var_list.append(slope_cov[0, 0])
        # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - slope_asymp_opt[1], c=cma.colors[4], label='tPpP to pP - b')
        # ax.plot(np.arange(1, Nsteps + 1), slope_asymp_fit(steps, *slope_asymp_opt) - slope_asymp_opt[1], c=cma.colors[5], label=r'$a^x + (1-a^{x-1})b$')
        # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP, c=cma.colors[8], label='tOpP to pP')
        # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP_anal, c=cma.colors[9], label='tOpP to pP analytic')

        print(*slope_opt)

        # print(*slope_asymp_opt)
    cma = plt.cm.get_cmap('tab20c')
    ax.plot(steps, p_PtoP - Number_of_P / (Number_of_O + Number_of_P), '-.', c=cma.colors[16], label='true')
    ax.plot(steps, np.power(5 / 6, steps) * (1 - ((5 / 6) ** -1) * Number_of_P / (Number_of_O + Number_of_P)), '-.',
            c=cma.colors[17], label='true 5/6')

    ax.set_xlabel('steps')
    ax.set_ylabel('p')
    ax.set_title('subtract known asymptote, fit slope only')
    ax.set_yscale('log')
    # ax.set_ylim([-0.05, 1.05])
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

    plt.close()

    f, ax = plt.subplots()
    ax.fill_between(nh_list, np.add(slope_list, np.multiply(-1., np.sqrt(slope_var_list))), np.add(slope_list,  np.sqrt(slope_var_list)), alpha=0.2)
    ax.plot(nh_list, slope_list, '.-')

    ax.set_xlabel('$n_h$')
    ax.set_ylabel(r'$\alpha$')
    ax.set_title('slope & asymptote fit')
    plt.show()
    plt.close()

def plot_alpha_klist(klist):
    nh_list = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    f, ax = plt.subplots()
    cma = plt.cm.get_cmap('tab20c')
    for i, k in enumerate(klist):
        Nsteps = k*k
        steps = np.arange(1, Nsteps+1)
        if k == 4:
            load_data_kfolds(20, 80, 0.0050, k)
        elif k == 3:
            load_data_kfolds(18, 80, 0.0030, k)
        else:
            load_data_kfolds(20, 80, 0.0020, k)

        def slope_fit(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * N_pP[0] / (N_pP[0] + N_pO[0])

        slope_opt, slope_cov = curve_fit(slope_fit, steps, p_tPtotP[0])
        print(*slope_opt)
        alpha_c = slope_opt[0]
        if k > 4:
            dat = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\random_walks\\alpha_fit_nf20_mean_var.npz'.
                          format(k, k))
        else:
            if k == 3:
                dat = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\alpha_fit_nf18_mean_var.npz'.
                              format(k, k))
            else:
                dat = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\alpha_fit_nf20_mean_var.npz'.
                              format(k, k))
        slope_kfoldavg = dat['mean']
        slope_kfoldvar = dat['var']
        x_dat = nh_list[:np.shape(slope_kfoldavg)[0]]
        if k % 2:
            "odd"
            ax.axhline(1 - alpha_c, 0, 1, ls='--', color=cma.colors[3-int(i/2)])
            ax.fill_between(x_dat, np.add(1-slope_kfoldavg, -1. * np.sqrt(slope_kfoldvar)),
                            np.add(1-slope_kfoldavg, np.sqrt(slope_kfoldvar)), color=cma.colors[3-int(i/2)], alpha=0.2)
            ax.plot(x_dat, 1-slope_kfoldavg, '^-', c=cma.colors[3-int(i/2)], label=k)
        else:
            'even'
            ax.axhline(1 - alpha_c, 0, 1, ls='--', color=cma.colors[7 - int(i / 2)])
            ax.fill_between(x_dat, np.add(1 - slope_kfoldavg, -1. * np.sqrt(slope_kfoldvar)),
                            np.add(1 - slope_kfoldavg, np.sqrt(slope_kfoldvar)), color=cma.colors[7 - int(i / 2)],
                            alpha=0.2)
            ax.plot(x_dat, 1 - slope_kfoldavg, 's-', c=cma.colors[7 - int(i / 2)], label=k)

    ax.set_xlabel('$n_h$')
    ax.set_ylabel(r'$\langle \alpha \rangle$')
    # ax.set_title('slope fit only')
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        '..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet.pdf'.format(
            k),
        facecolor=f.get_facecolor())
    plt.savefig(
        '..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet.svg'.format(
            k),
        facecolor=f.get_facecolor())
    plt.savefig(
        '..\\MetaCombi\\results\\modescaling\\figures\\p_tPpPtopP_kfoldavg_fit_slope_vs_nh_TestSet.png'.format(
            k), dpi=400,
        facecolor=f.get_facecolor())
    plt.show()
    plt.close()

def plot_p_foldavg_Unimodal_vs_Oligomodal_big_inc_stripmodes():
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    # mpl.rcParams['axes.labelsize'] = 24
    # mpl.rcParams['xtick.labelsize'] = 24
    # mpl.rcParams['ytick.labelsize'] = 24
    # mpl.rcParams['figure.subplot.left'] = 0.25
    # mpl.rcParams['figure.subplot.right'] = 0.95
    # mpl.rcParams['figure.subplot.bottom'] = 0.15
    # mpl.rcParams['figure.subplot.top'] = 0.95
    # # mpl.rcParams['figure.figsize']=[ 8,8]
    # # mpl.rcParams['font.family']='Times'
    # mpl.rcParams['axes.labelsize'] = 42
    # mpl.rcParams['lines.linewidth'] = 5
    # mpl.rcParams['axes.linewidth'] = 3
    # mpl.rcParams['xtick.major.size'] = 0 * 8
    # mpl.rcParams['xtick.major.width'] = 3
    # mpl.rcParams['xtick.major.pad'] = 2
    # mpl.rcParams['ytick.major.pad'] = 2
    # mpl.rcParams['ytick.major.size'] = 0 * 8
    # mpl.rcParams['ytick.major.width'] = 3
    # mpl.rcParams['xtick.minor.size'] = 0  # 7.5
    # mpl.rcParams['xtick.minor.width'] = 0
    # mpl.rcParams['ytick.minor.size'] = 0  # 7.5
    # mpl.rcParams['ytick.minor.width'] = 0
    # mpl.rcParams['ytick.direction'] = "in"
    # mpl.rcParams['xtick.direction'] = "in"
    cm = plt.get_cmap('Greens')
    cma = plt.get_cmap('tab20c')
    c_red = 'tab:red'
    cmv = plt.get_cmap('viridis')
    cmtab20 = plt.get_cmap('tab20')
    alpha_c_list = np.zeros(6)
    # f, ax = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(4.0)))
    xlength = 3.32
    ylength = 0.3
    # f = plt.figure(1, figsize=(cm_to_inch(4.7), cm_to_inch(4.)))
    # xoffset = 0.25*(xlength / 4.7)
    # yoffset = 0.18*(xlength / 4.)
    # figfracx = (3.85 - xoffset*4.7) / 4.7
    # figfracy = figfracx*4.7 / 4.
    f = plt.figure(1, figsize=(cm_to_inch(4.3), cm_to_inch(4.3)))
    xoffset = 0.22 * (xlength / 4.3)
    yoffset = 0.18 * (xlength / 4.3)
    figfracx = (3.5 - xoffset * 4.3) / 4.3
    figfracy = figfracx * 4.3 / 4.3
    # figfracy = 0.7
    ax = f.add_axes([xoffset, yoffset, figfracx, figfracy])
    caxg = f.add_axes(
        [xoffset + figfracx + 0.01, yoffset, (0.3/4.3)-0.01, figfracy])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    klist_odd = np.array([3, 5, 7])
    klist_even = np.array([4, 6, 8])
    # karange = np.linspace(3, 8, 1000)
    # alpha_est_odd = np.divide((karange - 2), karange)
    # alpha_est_even = np.divide((karange - 1), karange)
    # ax.plot(karange, alpha_est_odd, ls='dashed', c='black', zorder=1)
    # ax.plot(karange, alpha_est_even, ls='dashdot', c='black', zorder=1)
    # f3 = plt.figure(3, figsize=(cm_to_inch(3.85), cm_to_inch(3.85)))
    # xoffset3 = 0.25 * (xlength / 3.85)
    # yoffset3 = 0.18 * (xlength / 3.85)
    # figfracx3 = (3.85 - xoffset3 * 3.85) / 3.85
    # figfracy3 = figfracx3 * 3.85 / 3.85
    f3 = plt.figure(3, figsize=(cm_to_inch(8.6), cm_to_inch(8.6)))
    xoffset3 = 0.3 * (xlength / 8.6)
    yoffset3 = 0.18 * (xlength / 8.6)
    figfracx3 = (8.5 - xoffset3 * 8.6) / 8.6
    figfracy3 = figfracx3 * 8.6 / 8.6
    ax3 = f3.add_axes([xoffset3, yoffset3, figfracx3, figfracy3])
    ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # f3, ax3 = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(4.0)))
    ax3.set_ylabel(r'$\rho_{\mathrm{O} \rightarrow \mathrm{O}}$', labelpad=0)
    ax3.set_xlabel('$s$', labelpad=0)
    ax.set_xticks([4])
    ax.set_xlim([2.5, 5.5])
    ax.set_ylim([0.3, 0.9])
    for k in range(4, 5):
        steps = np.arange(1, k*k + 1)
        Nsteps = k*k
        dat = np.load('..\\metacombi\\results\\modescaling\\probability_classchange_randomwalk_Unimodal_vs_Oligomodal_'
                      'inc_stripmodes_{:d}x{:d}_n4_test.npz'.format(k, k))
        UtoU = dat['UtoU']
        UtoO = dat['UtoO']
        OtoU = dat['OtoU']
        OtoO = dat['OtoO']

        p_UtoO = dat['p_UtoO']
        p_OtoU = dat['p_OtoU']
        p_OtoO = dat['p_OtoO']
        p_UtoU = dat['p_UtoU']
        if k < 6:
            res = np.loadtxt(
                    '..\\metacombi\\results\\modescaling\\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_Scen_'
                    'slope_modes_M1k_{:d}x{:d}_fixn4.txt'.
                    format(k, k),
                    delimiter=',')
        else:
            res = np.loadtxt(
                '..\\metacombi\\results\\modescaling\\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_Scen_'
                'slope_M_M1k_{:d}x{:d}.txt'.
                    format(k, k),
                delimiter=',')
        data = np.load(
            'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\data_unimodal_vs_oligomodal_inc_stripmodes_train_trainraw_test_'
            '{:d}x{:d}.npz'.format(k, k))
        # nanind = np.argwhere(np.isnan(raw_results[:, 2]))
        # raw_data = np.delete(raw_data, nanind[:, 0], axis=0)
        # raw_results = np.delete(raw_results, nanind[:, 0], axis=0)
        y_test = data['y_test']
        x_test = data['x_test'][:, :, :, 0]
        # beta = (OtoU[0] + OtoO[0]) / (OtoU[0] + OtoO[0] + UtoO[0] + UtoU[0])
        beta = np.sum(res[:, 3] > 1) / np.shape(res)[0]
        beta = np.sum(y_test == 1) / np.shape(y_test)[0]


        def slope_fit(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * (beta)


        slope_opt, slope_cov = curve_fit(slope_fit, steps, p_OtoO)

        nf1 = 20
        nf2 = 80
        nf3 = 160
        nh1 = 1000
        lr = 0.0005
        # cma = plt.cm.get_cmap('rainbow')
        slope_list = np.zeros((10))
        slope_var_list = np.zeros((10))
        p_list = []
        p_list_var = []

        # f2 = plt.figure(2, figsize=(cm_to_inch(3.9), cm_to_inch(4.)))
        # xoffset2 = 0.22*(xlength / 3.9)
        # yoffset2 = 0.18*(xlength / 4.)
        # figfracx2 = figfracy * 4. / 3.9
        # figfracy2 = figfracy
        f2 = plt.figure(2, figsize=(cm_to_inch(4.3), cm_to_inch(4.3)))
        xoffset2 = 0.22 * (xlength / 4.3)
        yoffset2 = 0.18 * (xlength / 4.3)
        figfracx2 = figfracy * 4.3 / 4.3
        figfracy2 = figfracy
        ax2 = f2.add_axes([xoffset2, yoffset2, figfracx2, figfracy2])
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
        ax2.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax2.set_xticks([0, 10, 20, 30, 40])
        # f2, ax2 = plt.subplots(figsize=(cm_to_inch(4.3), cm_to_inch(4.3)))
        ax2.set_ylim([0.0, 1.0])
        ax2.set_xlim([0, 20])
        color = 'tab:orange'
        ax2.set_xlabel('$s$')
        # ax2.set_ylabel('$p(X_s = P | X_0 = P)$', color=color)
        # ax2.set_ylabel(r'$p_{P \rightarrow P}(s)$', color=color)
        ax2.set_ylabel(r'$\rho_{\mathrm{O} \rightarrow \mathrm{O}}$')
        # ax1.plot(t, data1, color=color)
        # ax2.tick_params(axis='y', labelcolor=color)
        # axt2 = ax2.twinx()
        # color = 'tab:purple'
        # # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$', color=color)  # we already handled the x-label with ax1
        # axt2.set_ylabel(r'$\langle \tilde{p}_{P \rightarrow P} \rangle(s)$', color=color)
        # axt2.tick_params(axis='y', labelcolor=color)

        if k % 2:
            ax2.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
                    c=c_red, zorder=2, markersize=20, linewidth=.5)
            ax2.plot(steps, p_OtoO, marker='x', c=c_red, linestyle='None', zorder=3, markersize=2, linewidth=.5)
            # ax3.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
            #          c=cmtab20(2*(k-3)+1), zorder=2)
            # ax3.plot(steps, p_PtoP, marker=(k, 0, 0), c=cmtab20(2*(k-3)+1), linestyle='None', zorder=3, label=k)
        else:
            ax2.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
                    c=c_red, zorder=2, markersize=20, linewidth=.5)
            ax2.plot(steps, p_OtoO, marker='x', c=c_red, linestyle='None', zorder=3, markersize=2, linewidth=.5)
        ax3.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
                 c=cmtab20(2*(k-3)+1), zorder=2, markersize=2, linewidth=.5)
        ax3.plot(steps, p_OtoO, marker=(k, 0, 0), c=cmtab20(2*(k-3)), linestyle='None', zorder=3, label=k,
                 markersize=2, linewidth=.5)
        f2.tight_layout()
        f2.savefig('.\\figures\\p_tOtotO_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.pdf'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOtotO_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.svg'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOtotO_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.png'.format(k), dpi=400,
                   facecolor=f2.get_facecolor())

        # instantiate a second axes that shares the same x-axis
        # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$')

        load_data_kfolds_U_vs_O_big_inc_stripmodes(nf1, nf2, nf3, nh1, lr, kp=k)

        def slope_fit_class(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * beta
        p_list.append(np.nanmean(p_tOpOtopO, axis=0))
        p_list_var.append(np.nanvar(p_tOpOtopO, axis=0))
        for kfold in range(10):
            def slope_fit(x, a):
                return np.power(a, x) + (1 - np.power(a, x - 1)) * N_pO[kfold, 0] / (N_pO[kfold, 0] + N_pU[kfold, 0])



            def slope_asymp_fit(x, a, b):
                return np.power(a, x) + (1 - np.power(a, x - 1)) * b

            def slope_log_fit(x, a):
                return x * np.log(a) + np.log(1 - np.power(a, -1) * N_pO[kfold, 0] / (N_pO[kfold, 0] + N_pU[kfold, 0]))

            def slope_asymp_fit(x, a, b):
                return np.log(np.power(a, x) * (1 - b * np.power(a, -1)) + b)

            slope_opt, slope_cov = curve_fit(slope_fit, steps, p_tOpOtopO[kfold])
            # slope_asymp_opt, slope_asymp_cov = curve_fit(slope_asymp_fit, steps, p_tPpPtopP)
            asympt = N_pO[kfold, 0] / (N_pO[kfold, 0] + N_pU[kfold, 0])
            # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP[kfold] - asympt, c=cma(2. * i / (2 * len(nh_list))),
            #         label=r'$n_h=' + '{:d}'.format(nh) + '$')
            # ax.plot(np.arange(1, Nsteps + 1), slope_fit(steps, *slope_opt) - asympt,
            #         c=cma((2. * i + 1) / (2 * len(nh_list))), label=r'$n_h=' + '{:d}'.format(nh) + '$ fit')
            slope_list[kfold] = slope_opt[0]
            slope_var_list[kfold] = slope_cov[0, 0]
            # slope_list.append(slope_opt[0])
            # slope_var_list.append(slope_cov[0, 0])
            # ax.plot(np.arange(1, Nsteps + 1), p_tPpPtopP - slope_asymp_opt[1], c=cma.colors[4], label='tPpP to pP - b')
            # ax.plot(np.arange(1, Nsteps + 1), slope_asymp_fit(steps, *slope_asymp_opt) - slope_asymp_opt[1], c=cma.colors[5], label=r'$a^x + (1-a^{x-1})b$')
            # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP, c=cma.colors[8], label='tOpP to pP')
            # ax.plot(np.arange(1, Nsteps + 1), p_tOpPtopP_anal, c=cma.colors[9], label='tOpP to pP analytic')

            print(*slope_opt)

        # ax2.plot(steps, p_list[i], '.-', c=cm(1- i/len(nh_list)))
        ax2.plot(steps, p_list[0], '.', c=cmv(1), zorder=1)
        ax2.set_ylim([-0.05, 1.05])
        f2.tight_layout()
        f2.savefig('.\\figures\\p_tOpOtopO_k{:d}_uptonh{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.pdf'.format(k, nh), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOpOtopO_k{:d}_uptonh{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.svg'.format(k, nh), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOpOtopO_k{:d}_uptonh{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.png'.format(k, nh), dpi=400,
                   facecolor=f2.get_facecolor())

        ax2.set_ylim([0.0, 1.0])
        # ax2.set_xlabel('$s$')
        # ax2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$')
        f2.tight_layout()
        # plt.show()
        # ax2.set_title('{:d}x{:d}'.format(k, k))
        custom_lines = [Line2D([0], [0], color=cmv(0), lw=.5)]
        custom_lines.append(Line2D([0], [0], color='tab:red', lw=.5))
        # custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
        custom_labels = [r'$\bar{\rho}_{\mathrm{O} \rightarrow \mathrm{O}}$']
        custom_labels.append(r'$\rho_{\mathrm{O} \rightarrow \mathrm{O}}$')
        ax2.legend(custom_lines, custom_labels, loc='lower left')
        f2.savefig('.\\figures\\p_tOpOtopO_nh_k{:d}_nocbar_unimodal_vs_oligomodal_big_inc_stripmodes.pdf'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOpOtopO_nh_k{:d}_nocbar_unimodal_vs_oligomodal_big_inc_stripmodes.svg'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOpOtopO_nh_k{:d}_nocbar_unimodal_vs_oligomodal_big_inc_stripmodes.png'.format(k), facecolor=f2.get_facecolor(), dpi=400)
        norm = mpl.colors.Normalize(vmin=0, vmax=1, clip=False)
        cb = f2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmv), ax=ax2, pad=0.03)
        cb.ax.set_title('$n_h$')
        f2.tight_layout()
        # plt.show()
        # ax2.set_title('{:d}x{:d}'.format(k, k))
        f2.savefig('.\\figures\\p_tOpOtopO_nh_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.pdf'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOpOtopO_nh_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.svg'.format(k), facecolor=f2.get_facecolor())
        f2.savefig('.\\figures\\p_tOpOtopO_nh_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.png'.format(k), facecolor=f2.get_facecolor(), dpi=400)
        f2.clear()


        slope_opt, slope_cov = curve_fit(slope_fit_class, steps, p_OtoO)
        print('alpha class:')
        print('alpha class:')
        print(*slope_opt)
        alpha_c = slope_opt[0]
        alpha_c_list[k-3] = alpha_c

        slope_kfoldavg = np.mean(slope_list, axis=0)
        slope_kfoldvar = np.var(slope_list, axis=0)
        np.savez('.\\alpha_kfold_UvsO_big_inc_stripmodes.npz', avg=slope_kfoldavg, var=slope_kfoldvar)
        # ax.scatter(np.full(len(slope_kfoldavg), k), 1-slope_kfoldavg,
        #             color=cm((slope_kfoldavg-np.amin(slope_kfoldavg))/(np.amax(slope_kfoldavg) - np.amin(slope_kfoldavg))),
        #             marker=(k, 0, 0), linestyle='None')
        ax.scatter(k, slope_kfoldavg, edgecolors=cmv(1),
                   facecolors='None', marker=(k, 0, 0), linestyle='None', s=20, linewidths=.5
                   )
        # ax.scatter(k, 1-alpha_c, marker=(k, 0, 0), edgecolor='tab:red',
        #             facecolor='None', linestyle='None', s=40, linewidths=1.5
        #             )
        ax.hlines(alpha_c, xmin=k - .5, xmax=k + .5, colors='tab:red', linewidths=.5)
        # ax.errorbar(k, 1 - alpha_c, yerr=np.sqrt(slope_cov[0]), marker=(k, 0, 0), edgecolor='tab:red',
        #            facecolor='None', linestyle='None', s=40, linewidths=1.5
        #            )
    ax.set_xlabel('$k$')
    ax.set_ylabel(r'$\alpha$')

    # ax.set_title('slope fit only')
    # plt.legend()
    custom_lines = [Line2D([0], [0], color=cmv(0), lw=.5)]
    custom_lines.append(Line2D([0], [0], color='tab:red', lw=.5))
    # custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
    custom_labels = [r'$\bar{\alpha}$']
    custom_labels.append(r'$\alpha$')
    ax.legend(custom_lines, custom_labels)
    ax.set_yticks([0.4, 0.6, 0.8, 1.0])
    ax.set_ylim([0.3, 1.1])
    f.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), cax=caxg)
    caxg.set_title('$n_h$')
    f.savefig('.\\figures\\p_tOpOtopO_slopefit_nh_unimodal_vs_oligomodal_big_inc_stripmodes.pdf', facecolor=f.get_facecolor())
    f.savefig('.\\figures\\p_tOpOtopO_slopefit_nh_unimodal_vs_oligomodal_big_inc_stripmodes.svg', facecolor=f.get_facecolor())
    f.savefig('.\\figures\\p_tOpOtopO_slopefit_nh_unimodal_vs_oligomodal_big_inc_stripmodes.png', facecolor=f.get_facecolor(), dpi=400)
    np.save('.\\alpha_c_list_unimodal_vs_oligomodal_big_inc_stripmodes.npy', np.array(alpha_c_list))

    ax3.set_xscale('log')
    ax3.legend()
    f3.tight_layout()
    f3.savefig('.\\figures\\p_tOtotO_pfit_k345678_unimodal_vs_oligomodal_big_inc_stripmodes.pdf', facecolor=f.get_facecolor())
    f3.savefig('.\\figures\\p_tOtotO_pfit_k345678_unimodal_vs_oligomodal_big_inc_stripmodes.png', facecolor=f.get_facecolor(), dpi=400)
    f3.savefig('.\\figures\\p_tOtotO_pfit_k345678_unimodal_vs_oligomodal_big_inc_stripmodes.svg', facecolor=f.get_facecolor())
    plt.show()
    plt.close()

def plot_p_foldavg_Unimodal_vs_Oligomodal_big_inc_stripmodes_fig2(k):
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    # mpl.rcParams['axes.labelsize'] = 24
    # mpl.rcParams['xtick.labelsize'] = 24
    # mpl.rcParams['ytick.labelsize'] = 24
    # mpl.rcParams['figure.subplot.left'] = 0.25
    # mpl.rcParams['figure.subplot.right'] = 0.95
    # mpl.rcParams['figure.subplot.bottom'] = 0.15
    # mpl.rcParams['figure.subplot.top'] = 0.95
    # # mpl.rcParams['figure.figsize']=[ 8,8]
    # # mpl.rcParams['font.family']='Times'
    # mpl.rcParams['axes.labelsize'] = 42
    # mpl.rcParams['lines.linewidth'] = 5
    # mpl.rcParams['axes.linewidth'] = 3
    # mpl.rcParams['xtick.major.size'] = 0 * 8
    # mpl.rcParams['xtick.major.width'] = 3
    # mpl.rcParams['xtick.major.pad'] = 2
    # mpl.rcParams['ytick.major.pad'] = 2
    # mpl.rcParams['ytick.major.size'] = 0 * 8
    # mpl.rcParams['ytick.major.width'] = 3
    # mpl.rcParams['xtick.minor.size'] = 0  # 7.5
    # mpl.rcParams['xtick.minor.width'] = 0
    # mpl.rcParams['ytick.minor.size'] = 0  # 7.5
    # mpl.rcParams['ytick.minor.width'] = 0
    # mpl.rcParams['ytick.direction'] = "in"
    # mpl.rcParams['xtick.direction'] = "in"
    cm = plt.get_cmap('Greens')
    cma = plt.get_cmap('tab20c')
    c_red = 'tab:red'
    cmv = plt.get_cmap('viridis')
    cmtab20 = plt.get_cmap('tab20')
    alpha_c_list = np.zeros(6)
    # f, ax = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(4.0)))
    xlength = 3.32
    ylength = 0.3
    # f = plt.figure(1, figsize=(cm_to_inch(4.7), cm_to_inch(4.)))
    # xoffset = 0.25*(xlength / 4.7)
    # yoffset = 0.18*(xlength / 4.)
    # figfracx = (3.85 - xoffset*4.7) / 4.7
    # figfracy = figfracx*4.7 / 4.
    f = plt.figure(1, figsize=(cm_to_inch(2.8667), cm_to_inch(2.8667)))
    xoffset = 0.22 * (xlength / 2.8667)
    yoffset = 0.18 * (xlength / 2.8667)
    figfracx = (2.8667-0.1 - xoffset * 2.8667) / 2.8667
    figfracy = figfracx * 2.8667 / 2.8667
    # figfracy = 0.7
    ax = f.add_axes([xoffset, yoffset, figfracx, figfracy])
    # caxg = f.add_axes(
    #     [xoffset + figfracx + 0.01, yoffset, (0.3/4.3)-0.01, figfracy])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$', labelpad=0)
    ax.set_xlabel(r'$s$', labelpad=0)
    # klist_odd = np.array([3, 5, 7])
    # klist_even = np.array([4, 6, 8])
    # karange = np.linspace(3, 8, 1000)
    # alpha_est_odd = np.divide((karange - 2), karange)
    # alpha_est_even = np.divide((karange - 1), karange)
    # ax.plot(karange, alpha_est_odd, ls='dashed', c='black', zorder=1)
    # ax.plot(karange, alpha_est_even, ls='dashdot', c='black', zorder=1)
    # f3 = plt.figure(3, figsize=(cm_to_inch(3.85), cm_to_inch(3.85)))
    # xoffset3 = 0.25 * (xlength / 3.85)
    # yoffset3 = 0.18 * (xlength / 3.85)
    # figfracx3 = (3.85 - xoffset3 * 3.85) / 3.85
    # figfracy3 = figfracx3 * 3.85 / 3.85
    # f3 = plt.figure(3, figsize=(cm_to_inch(8.6), cm_to_inch(8.6)))
    # xoffset3 = 0.3 * (xlength / 8.6)
    # yoffset3 = 0.18 * (xlength / 8.6)
    # figfracx3 = (8.5 - xoffset3 * 8.6) / 8.6
    # figfracy3 = figfracx3 * 8.6 / 8.6
    # ax3 = f3.add_axes([xoffset3, yoffset3, figfracx3, figfracy3])
    # ax3.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # # f3, ax3 = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(4.0)))
    # ax3.set_ylabel(r'$\rho_{\mathrm{O} \rightarrow \mathrm{O}}$', labelpad=0)
    # ax3.set_xlabel('$s$', labelpad=0)
    # ax.set_xticks([4])
    # ax.set_xlim([2.5, 5.5])
    # ax.set_ylim([0.3, 0.9])
    # for k in range(4, 5):
    steps = np.arange(1, k*k + 1)
    Nsteps = k*k
    dat = np.load('..\\metacombi\\results\\modescaling\\probability_classchange_randomwalk_Unimodal_vs_Oligomodal_'
                  'inc_stripmodes_{:d}x{:d}_n4_test.npz'.format(k, k))
    UtoU = dat['UtoU']
    UtoO = dat['UtoO']
    OtoU = dat['OtoU']
    OtoO = dat['OtoO']

    p_UtoO = dat['p_UtoO']
    p_OtoU = dat['p_OtoU']
    p_OtoO = dat['p_OtoO']
    p_UtoU = dat['p_UtoU']
    if k < 6:
        res = np.loadtxt(
                '..\\metacombi\\results\\modescaling\\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_Scen_'
                'slope_modes_M1k_{:d}x{:d}_fixn4.txt'.
                format(k, k),
                delimiter=',')
    else:
        res = np.loadtxt(
            '..\\metacombi\\results\\modescaling\\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_Scen_'
            'slope_M_M1k_{:d}x{:d}.txt'.
                format(k, k),
            delimiter=',')
    data = np.load(
        'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\data_unimodal_vs_oligomodal_inc_stripmodes_train_trainraw_test_'
        '{:d}x{:d}.npz'.format(k, k))
    # nanind = np.argwhere(np.isnan(raw_results[:, 2]))
    # raw_data = np.delete(raw_data, nanind[:, 0], axis=0)
    # raw_results = np.delete(raw_results, nanind[:, 0], axis=0)
    y_test = data['y_test']
    x_test = data['x_test'][:, :, :, 0]
    # beta = (OtoU[0] + OtoO[0]) / (OtoU[0] + OtoO[0] + UtoO[0] + UtoU[0])
    beta = np.sum(res[:, 3] > 1) / np.shape(res)[0]
    # beta = np.sum(y_test == 1) / np.shape(y_test)[0]


    def slope_fit(x, a):
        return np.power(a, x) + (1 - np.power(a, x - 1)) * (beta)


    slope_opt, slope_cov = curve_fit(slope_fit, steps, p_OtoO)

    nf1 = 20
    nf2 = 80
    nf3 = 160
    nh1 = 1000
    lr = 0.0005
    # cma = plt.cm.get_cmap('rainbow')
    slope_list = np.zeros((10))
    slope_var_list = np.zeros((10))
    p_list = []
    p_list_var = []

    # f2 = plt.figure(2, figsize=(cm_to_inch(3.9), cm_to_inch(4.)))
    # xoffset2 = 0.22*(xlength / 3.9)
    # yoffset2 = 0.18*(xlength / 4.)
    # figfracx2 = figfracy * 4. / 3.9
    # figfracy2 = figfracy
    # ax1.plot(t, data1, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # axt2 = ax2.twinx()
    # color = 'tab:purple'
    # # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$', color=color)  # we already handled the x-label with ax1
    # axt2.set_ylabel(r'$\langle \tilde{p}_{P \rightarrow P} \rangle(s)$', color=color)
    # axt2.tick_params(axis='y', labelcolor=color)


    ax.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
            c=c_red, zorder=2, markersize=20, linewidth=.5)
    ax.plot(steps, p_OtoO, marker='x', c=c_red, linestyle='None', zorder=3, markersize=2, linewidth=.5)

    # print('test = {:.4f}'.format(0.421042190))
    print('alpha = {:.9f} pm {:.9f}'.format(slope_opt[0], np.sqrt(slope_cov[0, 0])))

    # instantiate a second axes that shares the same x-axis
    # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$')

    load_data_kfolds_U_vs_O_big_inc_stripmodes(nf1, nf2, nf3, nh1, lr, kp=k)

    def slope_fit_class(x, a):
        return np.power(a, x) + (1 - np.power(a, x - 1)) * beta
    p_list.append(np.nanmean(p_tOpOtopO, axis=0))
    p_list_var.append(np.nanvar(p_tOpOtopO, axis=0))

    # ax2.plot(steps, p_list[i], '.-', c=cm(1- i/len(nh_list)))
    ax.plot(steps, p_list[0], '.', c=cmv(1), zorder=1)
    ax.fill_between(steps, p_list[0] - np.sqrt(p_list_var[0]), p_list[0] + np.sqrt(p_list_var[0]), alpha=0.2,
                    color=cmv(1), zorder=1)
    ax.set_ylim([-0.05, 1.05])

    custom_lines = [Line2D([0], [0], color=cmv(0), lw=.5)]
    custom_lines.append(Line2D([0], [0], color='tab:red', lw=.5))
    # custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
    custom_labels = [r'$\bar{\rho}_{\mathrm{C} \rightarrow \mathrm{C}}$']
    custom_labels.append(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$')
    ax.legend(custom_lines, custom_labels, loc='upper right')
    # f.savefig('.\\figures\\p_tOpOtopO_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.pdf'.format(k),
    #           facecolor=f.get_facecolor())
    # f.savefig('.\\figures\\p_tOpOtopO_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.svg'.format(k),
    #           facecolor=f.get_facecolor())
    # f.savefig('.\\figures\\p_tOpOtopO_k{:d}_unimodal_vs_oligomodal_big_inc_stripmodes.png'.format(k),
    #           facecolor=f.get_facecolor(), dpi=400)
    plt.show()
    plt.close()

    alphas =[]
    alphavars = []
    for f in range(10):
        beta_bar = N_pO[f] / (N_pO[f] + N_pU[f])
        def slope_fit_bar(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * beta_bar
        slope_opt, slope_cov = curve_fit(slope_fit_bar, steps, p_tOpOtopO[f])
        alphas.append(slope_opt)
        alphavars.append(slope_cov[0, 0])
    alphabar = np.mean(alphas)
    alphabar_std = np.sqrt(np.sum(alphavars)) / len(alphas)
    print('alpha bar = {:.9f} \t pm {:.9f}'.format(alphabar, alphabar_std))

def plot_p_foldavg_fig2(k):
    if k == 4:
        strings = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_5to9',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_9to13',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_13to17',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_17to18']
    elif k==5:
        strings = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_0to2',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_2to4',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_4to6',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_6to8',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_8to10',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_10to12',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_12to14',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_5x5_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_5x5_AB_5050_nh_14to17'
        ]
    for parts in range(len(strings)):
        resultstring = strings[parts]
        if parts == 0:
            KFres = np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t')
            # KFres = KFres[np.newaxis, ...]
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'), axis=0)

    "select nf nh lr combi with lowest val loss"
    indnf20 = np.argwhere(KFres[:, 0].astype(int) == 20)
    inds = np.nanargmax(KFres[indnf20[:, 0], 3])
    nf = KFres[indnf20[inds, 0], 0]
    nh = KFres[indnf20[inds, 0], 1]
    lr = KFres[indnf20[inds, 0], 2]


    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
    # mpl.rcParams['axes.labelsize'] = 24
    # mpl.rcParams['xtick.labelsize'] = 24
    # mpl.rcParams['ytick.labelsize'] = 24
    # mpl.rcParams['figure.subplot.left'] = 0.25
    # mpl.rcParams['figure.subplot.right'] = 0.95
    # mpl.rcParams['figure.subplot.bottom'] = 0.15
    # mpl.rcParams['figure.subplot.top'] = 0.95
    # # mpl.rcParams['figure.figsize']=[ 8,8]
    # # mpl.rcParams['font.family']='Times'
    # mpl.rcParams['axes.labelsize'] = 42
    # mpl.rcParams['lines.linewidth'] = 5
    # mpl.rcParams['axes.linewidth'] = 3
    # mpl.rcParams['xtick.major.size'] = 0 * 8
    # mpl.rcParams['xtick.major.width'] = 3
    # mpl.rcParams['xtick.major.pad'] = 2
    # mpl.rcParams['ytick.major.pad'] = 2
    # mpl.rcParams['ytick.major.size'] = 0 * 8
    # mpl.rcParams['ytick.major.width'] = 3
    # mpl.rcParams['xtick.minor.size'] = 0  # 7.5
    # mpl.rcParams['xtick.minor.width'] = 0
    # mpl.rcParams['ytick.minor.size'] = 0  # 7.5
    # mpl.rcParams['ytick.minor.width'] = 0
    # mpl.rcParams['ytick.direction'] = "in"
    # mpl.rcParams['xtick.direction'] = "in"
    cm = plt.get_cmap('Greens')
    cma = plt.get_cmap('tab20c')
    c_red = 'tab:red'
    cmv = plt.get_cmap('viridis')
    cmtab20 = plt.get_cmap('tab20')
    alpha_c_list = np.zeros(6)
    # f, ax = plt.subplots(figsize=(cm_to_inch(4.5), cm_to_inch(4.0)))
    xlength = 3.32
    ylength = 0.3
    # f = plt.figure(1, figsize=(cm_to_inch(4.7), cm_to_inch(4.)))
    # xoffset = 0.25*(xlength / 4.7)
    # yoffset = 0.18*(xlength / 4.)
    # figfracx = (3.85 - xoffset*4.7) / 4.7
    # figfracy = figfracx*4.7 / 4.
    f = plt.figure(1, figsize=(cm_to_inch(2.8667), cm_to_inch(2.8667)))
    xoffset = 0.22 * (xlength / 2.8667)
    yoffset = 0.18 * (xlength / 2.8667)
    figfracx = (2.8667-0.1 - xoffset * 2.8667) / 2.8667
    figfracy = figfracx * 2.8667 / 2.8667
    # figfracy = 0.7
    ax = f.add_axes([xoffset, yoffset, figfracx, figfracy])
    # caxg = f.add_axes(
    #     [xoffset + figfracx + 0.01, yoffset, (0.3/4.3)-0.01, figfracy])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    ax.set_ylabel(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$', labelpad=0)
    ax.set_xlabel(r'$s$', labelpad=0)
    steps = np.arange(1, k*k + 1)
    Nsteps = k*k
    dat = np.load('..\\metacombi\\results\\modescaling\\probability_classchange_randomwalk_{:d}x{:d}_test.npz'.
                  format(k, k))
    PtoP = dat['PtoP']
    OtoP = dat['OtoP']
    PtoO = dat['PtoO']
    PtoP = dat['PtoP']

    p_OtoP = dat['p_OtoP']
    p_PtoO = dat['p_PtoO']
    p_OtoO = dat['p_OtoO']
    p_PtoP = dat['p_PtoP']
    if k < 6:
        res = np.loadtxt(
            '..\\metacombi\\results\\modescaling\\results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}_fixn4.txt'.
                format(k, k),
            delimiter=',')
    elif k == 7:
        res = np.loadtxt(
            '..\\metacombi\\results\\modescaling\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}_extended.txt'.
                format(k, k),
            delimiter=',')

    else:
        res = np.loadtxt(
            '..\\metacombi\\results\\modescaling\\results_analysis_new_rrQR_i_Scen_slope_M1k_{:d}x{:d}.txt'.
                format(k, k),
            delimiter=',')
    beta = np.sum(res[:, 1] == 1) / np.shape(res)[0]


    def slope_fit(x, a):
        return np.power(a, x) + (1 - np.power(a, x - 1)) * (beta)


    slope_opt, slope_cov = curve_fit(slope_fit, steps, p_PtoP)

    # cma = plt.cm.get_cmap('rainbow')
    slope_list = np.zeros((10))
    slope_var_list = np.zeros((10))
    p_list = []
    p_list_var = []

    # f2 = plt.figure(2, figsize=(cm_to_inch(3.9), cm_to_inch(4.)))
    # xoffset2 = 0.22*(xlength / 3.9)
    # yoffset2 = 0.18*(xlength / 4.)
    # figfracx2 = figfracy * 4. / 3.9
    # figfracy2 = figfracy
    # ax1.plot(t, data1, color=color)
    # ax2.tick_params(axis='y', labelcolor=color)
    # axt2 = ax2.twinx()
    # color = 'tab:purple'
    # # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$', color=color)  # we already handled the x-label with ax1
    # axt2.set_ylabel(r'$\langle \tilde{p}_{P \rightarrow P} \rangle(s)$', color=color)
    # axt2.tick_params(axis='y', labelcolor=color)


    ax.plot(np.linspace(1, Nsteps + 1, 100), slope_fit(np.linspace(1, Nsteps + 1, 100), *slope_opt),
            c=c_red, zorder=2, markersize=20, linewidth=.5)
    ax.plot(steps, p_PtoP, marker='x', c=c_red, linestyle='None', zorder=3, markersize=2, linewidth=.5)

    # instantiate a second axes that shares the same x-axis
    # axt2.set_ylabel('$p(X_s \simeq P | X_0 = P, X_0 \simeq P)$')
    print('alpha = {:.9f} \t pm {:.9f}'.format(slope_opt[0], np.sqrt(slope_cov[0, 0])))
    load_data_kfolds(nf, nh, lr, kp=k)



    p_list.append(np.nanmean(p_tPpPtopP, axis=0))
    p_list_var.append(np.nanvar(p_tPpPtopP, axis=0))

    # ax2.plot(steps, p_list[i], '.-', c=cm(1- i/len(nh_list)))
    ax.plot(steps, p_list[0], '.', c=cmv(1), zorder=1)
    ax.fill_between(steps, p_list[0] - np.sqrt(p_list_var[0]), p_list[0] + np.sqrt(p_list_var[0]), alpha=0.2,
                    color=cmv(1), zorder=1)
    ax.set_ylim([-0.05, 1.05])

    custom_lines = [Line2D([0], [0], color=cmv(0), lw=.5)]
    custom_lines.append(Line2D([0], [0], color='tab:red', lw=.5))
    # custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
    custom_labels = [r'$\bar{\rho}_{\mathrm{C} \rightarrow \mathrm{C}}$']
    custom_labels.append(r'$\rho_{\mathrm{C} \rightarrow \mathrm{C}}$')
    ax.legend(custom_lines, custom_labels, loc='upper right')
    # f.savefig('.\\figures\\p_tPpPtopP_k{:d}_nf20_bestvalacc.pdf'.format(k),
    #           facecolor=f.get_facecolor())
    # f.savefig('.\\figures\\p_tPpPtopP_k{:d}_nf20_bestvalacc.svg'.format(k),
    #           facecolor=f.get_facecolor())
    # f.savefig('.\\figures\\p_tPpPtopP_k{:d}_nf20_bestvalacc.png'.format(k),
    #           facecolor=f.get_facecolor(), dpi=400)
    plt.show()
    plt.close()

    alphas = []
    alphavars = []
    for f in range(10):
        beta_bar = N_pP[f] / (N_pP[f] + N_pO[f])
        def slope_fit_bar(x, a):
            return np.power(a, x) + (1 - np.power(a, x - 1)) * beta_bar
        slope_opt, slope_cov = curve_fit(slope_fit_bar, steps, p_tPpPtopP[f])
        alphas.append(slope_opt)
        alphavars.append(slope_cov)
    alphabar = np.mean(alphas)
    alphabar_std = np.sqrt(np.sum(alphavars)) / len(alphas)
    print('alpha bar = {:.9f} \t pm {:.9f}'.format(alphabar, alphabar_std))
# plot_panal()
# plot_p_bestkfold()
# plot_p_kfolds()
# plot_alpha_klist(np.array([3, 4, 5, 6, 7, 8]))
# k = 5
# plot_p_kfolds()
plot_p_foldavg()