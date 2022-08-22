import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(0)

def cm_to_inch(x):
    return x / 2.54

def testset_output_allCNN(kxlist, kylist):
    for kx in kxlist:
        for ky in kylist:
            "load in pre-k train and test data"
            dataprek = np.load(u'.\\data_smiley_cube_train_trainraw_test_{:d}x{:d}.npz'.format(kx, ky))
            x_test = dataprek['x_test'].astype(np.float32)
            y_test = dataprek['y_test'].astype(np.float32)
            datpath = '.\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}'.format(kx, ky)
            kfold_avg_val_results = np.loadtxt(datpath+ '\\kfold_avg_val_results.txt', delimiter='\t')
            if len(np.shape(kfold_avg_val_results)) == 1:
                kfold_avg_val_results = kfold_avg_val_results[np.newaxis, :]
            arg_best = np.argmin(kfold_avg_val_results[:, 5])
            nf = kfold_avg_val_results[arg_best, 0].astype(int)
            nh = kfold_avg_val_results[arg_best, 1].astype(int)
            lr = kfold_avg_val_results[arg_best, 2]
            N_kfolds = 10
            if not os.path.exists(u'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'.format(kx, ky)):
                os.makedirs(u'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'.format(kx, ky))
            for kfold in range(N_kfolds):
                filepath = datpath + u"\\saves\\nf={:d}_nh={:d}_lr={:.4f}" \
                           u"\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
                if os.path.isfile(
                        'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                        'testres_nf={:d}_nh={:d}_lr={:.4f}'
                        '_kfold{:d}.npy'.
                                format(kx, ky, nf, nh, lr, kfold)):
                    print('skip nf{:d} nh{:d} lr{:.4f} kfold{:d}'.format(nf, nh, lr, kfold))
                    continue
                if os.path.isfile(filepath + 'variables\\variables.data-00001-of-00002') and os.path.isfile(
                        filepath + 'variables\\variables.data-00000-of-00002'):
                    print('loading model')
                    model = tf.keras.models.load_model(
                        filepath, custom_objects=None, compile=False
                    )
                    model.summary()
                    outval = model.predict(x_test)
                    np.save(
                        'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                        'testres_nf={:d}_nh={:d}_lr={:.4f}'
                        '_kfold{:d}.npy'.format(
                            kx, ky, nf, nh, lr, kfold), outval)
                    tf.keras.backend.clear_session()
                elif os.path.isfile(filepath + 'variables\\variables.data-00000-of-00001'):
                    print('loading model')
                    model = tf.keras.models.load_model(
                        filepath, custom_objects=None, compile=False
                    )
                    model.summary()
                    outval = model.predict(x_test)
                    np.save(
                        'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                        'testres_nf={:d}_nh={:d}_lr={:.4f}'
                        '_kfold{:d}.npy'.format(
                            kx, ky, nf, nh, lr, kfold), outval)
                    tf.keras.backend.clear_session()
    return 0

def CM_testset(kxlist, kylist):
    for kx in kxlist:
        for ky in kylist:
            dataprek = np.load(u'.\\data_smiley_cube_train_trainraw_test_{:d}x{:d}.npz'.format(kx, ky))
            x_test = dataprek['x_test']
            y_test = dataprek['y_test']

            datpath = '.\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}'.format(kx, ky)
            kfold_avg_val_results = np.loadtxt(datpath + '\\kfold_avg_val_results.txt', delimiter='\t')
            if len(np.shape(kfold_avg_val_results)) == 1:
                kfold_avg_val_results = kfold_avg_val_results[np.newaxis, :]
            arg_best = np.argmin(kfold_avg_val_results[:, 5])
            nf = kfold_avg_val_results[arg_best, 0].astype(int)
            nh = kfold_avg_val_results[arg_best, 1].astype(int)
            lr = kfold_avg_val_results[arg_best, 2]
            CM_mat = np.zeros((10, 5), dtype=int)
            for kfold in range(10):
                outval = np.load(
                    'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                    'testres_nf={:d}_nh={:d}_lr={:.4f}'
                    '_kfold{:d}.npy'.format(
                        kx, ky, nf, nh, lr, kfold))
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
            np.save('D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                    'testset_nf={:d}_nh={:d}_lr={:.4f}'
                    '_TP_FP_TN_FN.npy'.format(kx, ky, nf, nh, lr), CM_mat)
    return 0

def plot_BA_Volume(kxlist, kylist):
    for kx in kxlist:
        beta_class_list = []
        beta_bar_list = []
        BA_list = []
        for ky in kylist:
            datpath = '.\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}'.format(kx, ky)
            kfold_avg_val_results = np.loadtxt(datpath + '\\kfold_avg_val_results.txt', delimiter='\t')
            if len(np.shape(kfold_avg_val_results)) == 1:
                kfold_avg_val_results = kfold_avg_val_results[np.newaxis, :]
            arg_best = np.argmin(kfold_avg_val_results[:, 5])
            nf = kfold_avg_val_results[arg_best, 0].astype(int)
            nh = kfold_avg_val_results[arg_best, 1].astype(int)
            lr = kfold_avg_val_results[arg_best, 2]
            ConMat = np.load(r'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                             r'testset_nf={:d}_nh={:d}'
                             r'_lr={:.4f}_TP_FP_TN_FN.npy'.format(kx, ky, nf, nh, lr))
            total_pred = np.sum(ConMat[-1, :])
            beta_class = np.divide(ConMat[:, 1] + ConMat[:, 4], total_pred)
            beta_bar = np.divide(ConMat[:, 1] + ConMat[:, 2], total_pred)

            BA = 0.5*((ConMat[:, 1] / (ConMat[:, 1] + ConMat[:, 4])) + (ConMat[:, 3] / (ConMat[:, 3] + ConMat[:, 2])))

            beta_class_list.append(beta_class)
            beta_bar_list.append(beta_bar)
            BA_list.append(BA)

        "plot BA as function of ky"
        BA_list = np.array(BA_list)
        plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
        f, ax = plt.subplots()
        ax.plot(kylist, np.mean(BA_list, axis=1), '.-', c='tab:blue')
        ax.fill_between(kylist, np.mean(BA_list, axis=1)-np.sqrt(np.var(BA_list, axis=1)),
                        np.mean(BA_list, axis=1) + np.sqrt(np.var(BA_list, axis=1)), alpha=0.3, color='tab:blue')
        ax.set_xlabel('$k_y$')
        ax.set_ylabel('$\mathrm{BA}$')
        # ax.set_yscale('log')
        ax.set_title('$k_x = {:d}$'.format(kx))
        # ax.yaxis.set_major_locator(MaxNLocator(5))
        # ax.minorticks_off()
        # plt.legend()
        f.tight_layout()
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\BalAcc_vs_ky_kx{:d}.pdf'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\BalAcc_vs_ky_kx{:d}.svg'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\BalAcc_vs_ky_kx{:d}.png'.format(kx), dpi=400, facecolor=f.get_facecolor())
        plt.show()
        plt.close()

        "plot beta as a function of ky"
        beta_class_list = np.array(beta_class_list)
        beta_bar_list = np.array(beta_bar_list)
        plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
        f, ax = plt.subplots()
        ax.plot(kylist, np.mean(beta_bar_list, axis=1) , '.-', c='tab:blue', label=r'$\bar{\beta}$')
        ax.fill_between(kylist, np.mean(beta_bar_list, axis=1) - np.sqrt(np.var(beta_bar_list, axis=1)),
                        np.mean(beta_bar_list, axis=1) + np.sqrt(np.var(beta_bar_list, axis=1)),
                        alpha=0.3, color='tab:blue')
        ax.plot(kylist, np.mean(beta_class_list, axis=1), '.-', c='tab:red', label=r'$\beta$')
        ax.set_xlabel('$k_y$')
        ax.set_ylabel(r'$\beta$')
        ax.set_yscale('log')
        ax.set_title('$k_x = {:d}$'.format(kx))
        # ax.yaxis.set_major_locator(MaxNLocator(5))
        # ax.minorticks_off()
        plt.legend()
        f.tight_layout()
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\beta_vs_ky_kx{:d}.pdf'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\beta_vs_ky_kx{:d}.svg'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\beta_vs_ky_kx{:d}.png'.format(kx), dpi=400, facecolor=f.get_facecolor())
        plt.show()
        plt.close()
    return 0

def plot_BA_Volume_bestkfold(kxlist, kylist):
    for kx in kxlist:
        beta_class_list = []
        beta_bar_list = []
        BA_list = []
        for ky in kylist:
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
                          .format(nf, nh , lr))['val_loss_list']
            kfold = np.nanargmin(kfold_val_acc)
            ConMat = np.load(r'D:\\data\\cnn_Smiley_Cube_kx_ky\\cnn_Smiley_Cube_{:d}x{:d}\\'
                             r'testset_nf={:d}_nh={:d}'
                             r'_lr={:.4f}_TP_FP_TN_FN.npy'.format(kx, ky, nf, nh, lr))
            total_pred = np.sum(ConMat[-1, :])
            beta_class = np.divide(ConMat[kfold, 1] + ConMat[kfold, 4], total_pred)
            beta_bar = np.divide(ConMat[kfold, 1] + ConMat[kfold, 2], total_pred)

            BA = 0.5 * ((ConMat[kfold, 1] / (ConMat[kfold, 1] + ConMat[kfold, 4])) + (ConMat[kfold, 3] /
                                                                                      (ConMat[kfold, 3] +
                                                                                       ConMat[kfold, 2])))

            beta_class_list.append(beta_class)
            beta_bar_list.append(beta_bar)
            BA_list.append(BA)

        "plot BA as function of ky"
        BA_list = np.array(BA_list)
        plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
        f, ax = plt.subplots()
        ax.plot(kylist, BA_list, '.-', c='tab:blue')
        ax.set_xlabel('$k_y$')
        ax.set_ylabel('$\mathrm{BA}$')
        # ax.set_yscale('log')
        ax.set_title('$k_x = {:d}$'.format(kx))
        # ax.yaxis.set_major_locator(MaxNLocator(5))
        # ax.minorticks_off()
        # plt.legend()
        f.tight_layout()
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\BalAcc_vs_ky_kx{:d}_bestkfold.pdf'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\BalAcc_vs_ky_kx{:d}_bestkfold.svg'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\BalAcc_vs_ky_kx{:d}_bestkfold.png'.format(kx), dpi=400, facecolor=f.get_facecolor())
        plt.show()
        plt.close()

        "plot beta as a function of ky"
        beta_class_list = np.array(beta_class_list)
        beta_bar_list = np.array(beta_bar_list)
        plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
        f, ax = plt.subplots()
        ax.plot(kylist, beta_bar_list, '.-', c='tab:blue', label=r'$\bar{\beta}$')
        ax.plot(kylist, beta_class_list, '.-', c='tab:red', label=r'$\beta$')
        ax.set_xlabel('$k_y$')
        ax.set_ylabel(r'$\beta$')
        ax.set_yscale('log')
        ax.set_title('$k_x = {:d}$'.format(kx))
        # ax.yaxis.set_major_locator(MaxNLocator(5))
        # ax.minorticks_off()
        plt.legend()
        f.tight_layout()
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\beta_vs_ky_kx{:d}_bestkfold.pdf'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\beta_vs_ky_kx{:d}_bestkfold.svg'.format(kx), facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\beta_vs_ky_kx{:d}_bestkfold.png'.format(kx), dpi=400, facecolor=f.get_facecolor())
        plt.show()
        plt.close()
    return 0

def plot_CM_bestvalloss(kxlist, kylist):
    f = plt.figure(1, figsize=(cm_to_inch(8.6), cm_to_inch(6.1)))
    xlength = 3.32
    ylength = 0.3
    # f = plt.figure(1, figsize=(cm_to_inch(4.45), cm_to_inch(4.15)))
    xoffset = 0.3 * (xlength / 8.6)
    yoffset = 0.35 * (xlength / 6.1)
    # xoffset = 0.25
    # yoffset =
    frac_pad_x = 0.07 * (2. / 3.)
    frac_pad_y = frac_pad_x * 8.6 / 6.1
    figfracx = (8.5 - xoffset * 8.6 - 2 * frac_pad_x * 8.6) / 8.6
    figfracy = figfracx * 8.6 / (6.1)
    for kx in kxlist:
        for ky in kylist:
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

            vmax = np.amax(ConMat[bestkfold, :])
            colors = ['black', 'white']
            # ax = f.add_axes([xoffset + int((k - 3) / 2) * (frac_pad_x + figfracx / 3.),
            #                  yoffset + (k % 2) * (frac_pad_y + figfracy / 3.),
            #                  figfracx / 3., figfracy / 3.])
            ax = f.add_axes([xoffset + int(ky-3)*(frac_pad_x+figfracx/3.),
                             yoffset, figfracx / 3., figfracy / 3.])
            ax.imshow(CM, cmap='Blues', vmin=0, vmax=vmax, origin='lower')
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            # ax.text(-0.44, -0.1, np.format_float_scientific(CM[0, 0], precision=2, unique=True, exp_digits=1, min_digits=2),
            #         c='white', fontsize=8)
            ax.text(-0, 0, '{:d}'.format(CM[0, 0]), c=colors[int(2*CM[0, 0]/(vmax+1))], fontsize=7,
                    ha='center', va='center')
            ax.text(-0, 1, '{:d}'.format(CM[1, 0]), c=colors[int(2*CM[1, 0]/(vmax+1))], fontsize=7,
                    ha='center', va='center')
            ax.text(1, -0, '{:d}'.format(CM[0, 1]), c=colors[int(2*CM[0, 1]/(vmax+1))], fontsize=7,
                    ha='center', va='center')
            ax.text(1, 1, '{:d}'.format(CM[1, 1]), c=colors[int(2*CM[1, 1]/(vmax+1))], fontsize=7,
                    ha='center', va='center')
            # ax.text(0.56, 0.9, np.format_float_scientific(CM[1, 1], precision=2, unique=True, exp_digits=1, min_digits=2),
            #         c='white', fontsize=8)
            ax.set_title(r'${:d} \times {:d}$'.format(kx, ky), fontsize=8, pad=0.01)
            if ky == 4:
                ax.set_xlabel('predicted')
            ax.set_xticklabels(['C', 'I'])
            if ky == 3:
                ax.set_ylabel('actual')
            if ky==3:
                ax.set_yticklabels(['I', 'C'])
            else:
                ax.set_yticklabels([])
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\ConfusionMatrix_bestvalloss_kx{:d}.pdf'.format(kx),
                  facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\ConfusionMatrix_bestvalloss_kx{:d}.svg'.format(kx),
                  facecolor=f.get_facecolor())
        f.savefig('.\\cnn_Smiley_Cube_kx_ky\\ConfusionMatrix_bestvalloss_kx{:d}.png'.format(kx),
                  facecolor=f.get_facecolor(), dpi=400)
        plt.show()
        plt.close()
    return 0

# testset_output_allCNN(np.array([5]), np.array([3, 4, 5]))
# CM_testset(np.array([5]), np.array([3, 4, 5]))
# plot_BA_Volume_bestkfold(np.array([5]), np.array([3, 4, 5]))
plot_CM_bestvalloss(np.array([5]), np.array([3, 4, 5]))