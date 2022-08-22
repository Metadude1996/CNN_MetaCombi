import numpy as np
import matplotlib.pyplot as plt

def cm_to_inch(x):
    return x / 2.54
plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')

"plot 3 confusion matrices"
f = plt.figure(1, figsize=(cm_to_inch(4.3), cm_to_inch(4.3)))
xlength = 3.32
ylength = 0.3
# f = plt.figure(1, figsize=(cm_to_inch(4.45), cm_to_inch(4.15)))
xoffset = 0.2 * (xlength / 4.3)
yoffset = 0.2 * (xlength / 4.3)
# xoffset = 0.25
# yoffset =
frac_pad_x = 0.01 * (2. / 3.)
frac_pad_y = frac_pad_x * 4.3 / 4.3
figfracx = (4.3 - xoffset * 4.3 - 3 * frac_pad_x * 4.3) / 4.3
figfracy = figfracx * 4.3 / (4.3)

PM_index = 3
k = 5
i=0
"select best (nh, nf, lr, kfold)"

"strip modes"
dat = np.load('.\\testPM_nfnh_{:d}x{:d}.npy'.format(k, k))
nhlist = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
nflist = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])

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
    strings = [u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_4x4_AB_5050_2\\scratch\\output_dir\\cnniter_HP_GS_SKF_4x4_AB_5050_nh_0to5',
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
    res_kfolds[kfold] = np.loadtxt(string+u'kfold={:d}'.format(kfold)+u'\\results.txt', delimiter='\t')
    "results.txt: val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1"
bestkfold = np.argmax(res_kfolds[:, 1])
nf_arg = np.argwhere(CMs[i][:, 0, 0, 0, 0] == nf)[0, 0]
nh_arg = np.argwhere(CMs[i][0, :, 0, 0, 1] == nh)[0, 0]
lr_arg = np.argwhere(CMs[i][0, 0, :, 0, 2] == lr)[0, 0]
fold_arg = np.argwhere(CMs[i][0, 0, 0, :, 3] == bestkfold)[0, 0]
"CM = [nf, nh, lr, kfold, TP, FP, TN, FN]"
CM = np.zeros((2, 2), dtype=int)
CM[0, 0] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 5]
CM[1, 0] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 4]
CM[0, 1] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 6]
CM[1, 1] = CMs[i][nf_arg, nh_arg, lr_arg, fold_arg, 7]

CMfolds = CMs[i][nf_arg, nh_arg, lr_arg]
BA_stripmode = 0.5*(CMfolds[:, 4] / (CMfolds[:, 4] + CMfolds[:, 7]) + CMfolds[:, 6] / (CMfolds[:, 6] + CMfolds[:, 5]))

vmax = np.amax(CMs[i][nf_arg, nh_arg, lr_arg, fold_arg])
colors = ['black', 'white']
# ax = f.add_axes([xoffset + int((k - 3) / 2) * (frac_pad_x + figfracx / 3.),
#                  yoffset + (k % 2) * (frac_pad_y + figfracy / 3.),
#                  figfracx / 3., figfracy / 3.])
ax = f.add_axes([xoffset + frac_pad_x + figfracx/3., yoffset + 0.5, figfracx / 3., figfracy / 3.])
ax.imshow(CM, cmap='Blues', vmin=0, vmax=vmax, origin='lower')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
# ax.text(-0.44, -0.1, np.format_float_scientific(CM[0, 0], precision=2, unique=True, exp_digits=1, min_digits=2),
#         c='white', fontsize=8)
ax.text(-0, 0, '{:d}'.format(CM[0, 0]), c=colors[int(2*CM[0, 0]/(vmax+1))], fontsize=6, ha='center', va='center')
ax.text(-0, 1, '{:d}'.format(CM[1, 0]), c=colors[int(2*CM[1, 0]/(vmax+1))], fontsize=6, ha='center', va='center')
ax.text(1, -0, '{:d}'.format(CM[0, 1]), c=colors[int(2*CM[0, 1]/(vmax+1))], fontsize=6, ha='center', va='center')
ax.text(1, 1, '{:d}'.format(CM[1, 1]), c=colors[int(2*CM[1, 1]/(vmax+1))], fontsize=6, ha='center', va='center')
# ax.text(0.56, 0.9, np.format_float_scientific(CM[1, 1], precision=2, unique=True, exp_digits=1, min_digits=2),
#         c='white', fontsize=8)
ax.set_title(r'${:d} \times {:d}$'.format(k, k), fontsize=8, pad=0.01)

ax.set_xticklabels(['C', 'I'])
# ax.set_ylabel('actual')
ax.set_xlabel('predicted', labelpad=0)

ax.set_yticklabels([])

"smiley cube"
kx = 5
ky = 3
datpath = '.\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}'.format(kx, ky)
kfold_avg_val_results = np.loadtxt(datpath + '\\kfold_avg_val_results.txt', delimiter='\t')
if len(np.shape(kfold_avg_val_results)) == 1:
    kfold_avg_val_results = kfold_avg_val_results[np.newaxis, :]
arg_best = np.argmin(kfold_avg_val_results[:, 5])
nf = kfold_avg_val_results[arg_best, 0].astype(int)
nh = kfold_avg_val_results[arg_best, 1].astype(int)
lr = kfold_avg_val_results[arg_best, 2]
"select best kfold"
kfold_val_acc = np.load(datpath + '\\nf{:d}_nh{:d}_lr{:.4f}_kfold_val_acc_loss_prec_rec_f1.npz'
                        .format(nf, nh, lr))['val_loss_list']
bestkfold = np.nanargmin(kfold_val_acc)
ConMat = np.load(r'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                 r'testset_nf={:d}_nh={:d}'
                 r'_lr={:.4f}_TP_FP_TN_FN.npy'.format(kx, ky, nf, nh, lr))
"CM = [kfold, TP, FP, TN, FN]"
CM = np.zeros((2, 2), dtype=int)
CM[0, 0] = ConMat[bestkfold, 2]
CM[1, 0] = ConMat[bestkfold, 1]
CM[0, 1] = ConMat[bestkfold, 3]
CM[1, 1] = ConMat[bestkfold, 4]

BA_smiley = 0.5*(ConMat[:, 1] / (ConMat[:, 1] + ConMat[:, 4]) + ConMat[:, 3] / (ConMat[:, 3] + ConMat[:, 2]))

vmax = np.amax(ConMat[bestkfold, :])
colors = ['black', 'white']
# ax = f.add_axes([xoffset + int((k - 3) / 2) * (frac_pad_x + figfracx / 3.),
#                  yoffset + (k % 2) * (frac_pad_y + figfracy / 3.),
#                  figfracx / 3., figfracy / 3.])
ax = f.add_axes([xoffset,
                 yoffset + 0.5, figfracx / 3., figfracy / 3.])
ax.imshow(CM, cmap='Blues', vmin=0, vmax=vmax, origin='lower')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
# ax.text(-0.44, -0.1, np.format_float_scientific(CM[0, 0], precision=2, unique=True, exp_digits=1, min_digits=2),
#         c='white', fontsize=8)
ax.text(-0, 0, '{:d}'.format(CM[0, 0]), c=colors[int(2*CM[0, 0]/(vmax+1))], fontsize=6,
        ha='center', va='center')
ax.text(-0, 1, '{:d}'.format(CM[1, 0]), c=colors[int(2*CM[1, 0]/(vmax+1))], fontsize=6,
        ha='center', va='center')
ax.text(1, -0, '{:d}'.format(CM[0, 1]), c=colors[int(2*CM[0, 1]/(vmax+1))], fontsize=6,
        ha='center', va='center')
ax.text(1, 1, '{:d}'.format(CM[1, 1]), c=colors[int(2*CM[1, 1]/(vmax+1))], fontsize=6,
        ha='center', va='center')
# ax.text(0.56, 0.9, np.format_float_scientific(CM[1, 1], precision=2, unique=True, exp_digits=1, min_digits=2),
#         c='white', fontsize=8)
ax.set_title(r'${:d} \times {:d}$'.format(kx, ky), fontsize=8, pad=0.01)

ax.set_xticklabels(['C', 'I'])

ax.set_ylabel('actual', labelpad=0)
ax.set_yticklabels(['I', 'C'])

"unimodal vs oligomodal"
nf1 = 20
nf2 = 80
nf3 = 160
nh1 = 1000
lr = 0.0005

k = 5

resultstring = u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_' \
               u'{:d}x{:d}\\kfold_val_acc_loss_prec_rec_f1.npz'.format(k, k)
KFres = np.load(resultstring)['val_loss_list']
bestkfold = np.nanargmin(KFres)

ConMat = np.load(
    r'D:\\data\cnn_Unimodal_vs_Oligomodal_big_inc_stripmodes_{:d}x{:d}\\testset_nf1={:d}_nf2={:d}_nf3={:d}_nh1={:d}'
    r'_lr={:.4f}_TP_FP_TN_FN.npy'.format(k, k, nf1, nf2, nf3, nh1, lr))
"CM = [kfold, TP, FP, TN, FN]"
CM = np.zeros((2, 2), dtype=int)
CM[0, 0] = ConMat[bestkfold, 2]
CM[1, 0] = ConMat[bestkfold, 1]
CM[0, 1] = ConMat[bestkfold, 3]
CM[1, 1] = ConMat[bestkfold, 4]

BA_oligo = 0.5*(ConMat[:, 1] / (ConMat[:, 1] + ConMat[:, 4]) + ConMat[:, 3] / (ConMat[:, 3] + ConMat[:, 2]))
vmax = np.amax(ConMat[bestkfold, :])
colors = ['black', 'white']
# ax = f.add_axes([xoffset + int((k - 3) / 2) * (frac_pad_x + figfracx / 3.),
#                  yoffset + (k % 2) * (frac_pad_y + figfracy / 3.),
#                  figfracx / 3., figfracy / 3.])
ax = f.add_axes([xoffset+2*(figfracx/3. + frac_pad_x), yoffset+0.5, figfracx / 3., figfracy / 3.])
ax.imshow(CM, cmap='Blues', vmin=0, vmax=vmax, origin='lower')
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
# ax.text(-0.44, -0.1, np.format_float_scientific(CM[0, 0], precision=2, unique=True, exp_digits=1, min_digits=2),
#         c='white', fontsize=8)
ax.text(-0, 0, '{:d}'.format(CM[0, 0]), c=colors[int(2*CM[0, 0]/(vmax+1))], fontsize=6, ha='center', va='center')
ax.text(-0, 1, '{:d}'.format(CM[1, 0]), c=colors[int(2*CM[1, 0]/(vmax+1))], fontsize=6, ha='center', va='center')
ax.text(1, -0, '{:d}'.format(CM[0, 1]), c=colors[int(2*CM[0, 1]/(vmax+1))], fontsize=6, ha='center', va='center')
ax.text(1, 1, '{:d}'.format(CM[1, 1]), c=colors[int(2*CM[1, 1]/(vmax+1))], fontsize=6, ha='center', va='center')
# ax.text(0.56, 0.9, np.format_float_scientific(CM[1, 1], precision=2, unique=True, exp_digits=1, min_digits=2),
#         c='white', fontsize=8)
ax.set_title(r'${:d} \times {:d}$'.format(k, k), fontsize=8, pad=0.01)

ax.set_xticklabels(['C', 'I'])

# ax.set_ylabel('actual')
ax.set_yticklabels([])

# plt.show()

"plot BA results"
ax = f.add_axes([xoffset, yoffset, figfracx + 2*frac_pad_x, figfracy / 3.])
ax.boxplot([BA_smiley, BA_stripmode, BA_oligo])
ax.set_xticklabels(['i', 'ii', 'iii'])
ax.set_yticks([1.0, 0.8, 0.6])
ax.set_ylabel('$\mathrm{BA}$', labelpad=0)
ax.set_xlabel('classification', labelpad=0)


f.savefig('.\\figures\\ConfusionMatrices_BA_smileycube_stripmodes_oligomodal_{:d}x{:d}.pdf'.format(k, k),
          facecolor=f.get_facecolor())
f.savefig('.\\figures\\ConfusionMatrices_BA_smileycube_stripmodes_oligomodal_{:d}x{:d}.svg'.format(k, k),
          facecolor=f.get_facecolor())
f.savefig('.\\figures\\ConfusionMatrices_BA_smileycube_stripmodes_oligomodal_{:d}x{:d}.png'.format(k, k),
          facecolor=f.get_facecolor(), dpi=400)
plt.show()
plt.close()