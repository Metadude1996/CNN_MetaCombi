import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import colorcet as cc

plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')
def cm_to_inch(x):
    return x / 2.54
'load data'
dat = []
for k in range(3, 9):
    dat.append(np.load('.\\testPM_nfnh_{:d}x{:d}.npy'.format(k, k)))
nhlist = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100])
nflist = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20])
def plot_BAheatmaps():
    spaceratio = 1./(2.5)
    Lxratio = (3+2*spaceratio)/(1+1*spaceratio+3+2*spaceratio)
    Lxlength = 8.6 * Lxratio
    Lylength = (2+3*spaceratio)/(3+2*spaceratio) * Lxlength

    # fg = plt.figure(1, figsize=(cm_to_inch(Lxlength), cm_to_inch(Lylength)))
    # xoffset = spaceratio*1/(3+2*spaceratio)
    # yoffset = spaceratio*1/(2+3*spaceratio)
    # figfracx = 1/(3+2*spaceratio)
    # figfracy = 1/(2+3*spaceratio)

    xlength = 3.32
    ylength = 0.3
    fg = plt.figure(1, figsize=(cm_to_inch(3/2*xlength+ylength), cm_to_inch(xlength)))
    xoffset = 0.25 * (xlength/((3./2.)*xlength+ylength))
    yoffset = 0.18*(xlength/xlength)
    figfracy = 0.7 * (xlength / xlength)
    figfracx = 0.7 * (xlength / ((3./2.) * xlength + ylength))
    axg1 = fg.add_axes([xoffset, yoffset+figfracy/2., figfracx/2., figfracy/2.])
    axg2 = fg.add_axes([xoffset, yoffset, figfracx/2., figfracy/2.])
    axg3 = fg.add_axes([xoffset + figfracx/2., yoffset+figfracy/2., figfracx/2., figfracy/2.])
    axg4 = fg.add_axes([xoffset+figfracx/2., yoffset, figfracx/2., figfracy/2.])
    axg5 = fg.add_axes([xoffset+2*figfracx/2., yoffset+figfracy/2., figfracx/2., figfracy/2.])
    axg6 = fg.add_axes([xoffset+2*figfracx/2., yoffset, figfracx/2., figfracy/2.])
    # ax1 = fg.add_axis([1/11., 1/11., 3./11., 3./11.])

    # axg1 = fg.add_axes([xoffset, 2*yoffset+figfracy, figfracx, figfracy])
    # axg2 = fg.add_axes([xoffset, yoffset, figfracx, figfracy])
    # axg3 = fg.add_axes([xoffset + figfracx, 2*yoffset+figfracy, figfracx, figfracy])
    # axg4 = fg.add_axes([xoffset+figfracx, yoffset, figfracx, figfracy])
    # axg5 = fg.add_axes([xoffset+2*figfracx, 2*yoffset+figfracy, figfracx, figfracy])
    # axg6 = fg.add_axes([xoffset+2*figfracx, yoffset, figfracx, figfracy])
    axg1.set_xticks(np.arange(len(nflist))[::4])
    axg2.set_xticks(np.arange(len(nflist))[::4])
    axg3.set_xticks(np.arange(len(nflist))[::4])
    axg4.set_xticks(np.arange(len(nflist))[::4])
    axg5.set_xticks(np.arange(len(nflist))[::4])
    axg6.set_xticks(np.arange(len(nflist))[::4])
    axg1.set_yticks(np.arange(len(nhlist))[::5])
    axg2.set_yticks(np.arange(len(nhlist))[::5])
    axg3.set_yticks(np.arange(len(nhlist))[::5])
    axg4.set_yticks(np.arange(len(nhlist))[::5])
    axg5.set_yticks(np.arange(len(nhlist))[::5])
    axg6.set_yticks(np.arange(len(nhlist))[::5])
    axg1.set_xticklabels(nflist[::4], visible=False)
    axg1.set_yticklabels(nhlist[::5])
    axg2.set_xticklabels(nflist[::4])
    axg2.set_yticklabels(nhlist[::5])
    axg3.set_xticklabels(nflist[::4], visible=False)
    axg3.set_yticklabels(nhlist[::5], visible=False)
    axg4.set_xticklabels(nflist[::4])
    axg4.set_yticklabels(nhlist[::5], visible=False)
    axg5.set_xticklabels(nflist[::4], visible=False)
    axg5.set_yticklabels(nhlist[::5], visible=False)
    axg6.set_xticklabels(nflist[::4])
    axg6.set_yticklabels(nhlist[::5], visible=False)

    axg1.set_ylabel('$n_h$', labelpad=0)
    axg2.set_ylabel('$n_h$',  labelpad=0)
    axg2.set_xlabel('$n_f$',  labelpad=0)
    axg4.set_xlabel('$n_f$',  labelpad=0)
    axg6.set_xlabel('$n_f$', labelpad=0)

    caxg = fg.add_axes([xoffset+3*figfracx/2.+0.01, yoffset, (1-(figfracx+3*xoffset/2.))/12., figfracy])
    axs = [axg1, axg2, axg3, axg4, axg5, axg6]
    for i in range(len(axs)):
        aspect = np.shape(dat[i][:, :, 2, 0].T)[1] / np.shape(dat[i][:, :, 2, 0].T)[0]
        axs[i].imshow(dat[i][:, :, 2, 0].T, origin='lower', cmap=cc.cm["diverging_bwr_20_95_c54"], vmin=0, vmax=1,
                      aspect=aspect)
        # axs[i].set_title(r'${:d}\times{:d}$'.format(i+3, i+3))
        # axs[i].legend([r'${:d}\times {:d}$'.format(i+3, i+3)], loc='upper left')
        axs[i].axhline(9., 0, 1, ls='--', lw=0.5, c='black')
        # axs[i].annotate(r'${:d}\times {:d}$'.format(i+3, i+3), xy=(0.58, 0.8), xycoords='axes fraction',
        #              fontsize=6, color='white')
        axs[i].annotate(r'$k = {:d}$'.format(i + 3), xy=(0.58, 0.8), xycoords='axes fraction',
                        fontsize=6, color='white')
        axs[i].set_xticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[1]), minor=True)
        axs[i].set_xticks(np.arange(9)[::3]+1, minor=False)
        axs[i].set_xticklabels([4, 10, 16], minor=False)
        axs[i].set_yticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[0]), minor=True)
        axs[i].set_yticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[0]-1)[::4] + 1, minor=False)
        if i+3 == 4 or i+3 == 5:
            axs[i].set_yticklabels([4, 12, 20, 60, 100], minor=False)
        else:
            axs[i].set_yticklabels(['', 12, 20, 60, 100], minor=False)


    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
    fg.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cc.cm["diverging_bwr_20_95_c54"]), cax=caxg)
    caxg.set_title('$\mathrm{BA}$')

    fg.savefig('.\\figures\\BA_heatmaps_grid.pdf', facecolor=fg.get_facecolor())
    fg.savefig('.\\figures\\BA_heatmaps_grid.svg', facecolor=fg.get_facecolor())
    fg.savefig('.\\figures\\BA_heatmaps_grid.png', facecolor=fg.get_facecolor(), dpi=400)
    plt.show()
    plt.close()

def plot_TCRheatmaps():
    spaceratio = 1. / (2.5)
    Lxratio = (3 + 2 * spaceratio) / (1 + 1 * spaceratio + 3 + 2 * spaceratio)
    Lxlength = 8.6 * Lxratio
    Lylength = (2 + 3 * spaceratio) / (3 + 2 * spaceratio) * Lxlength

    # fg = plt.figure(1, figsize=(cm_to_inch(Lxlength), cm_to_inch(Lylength)))
    # xoffset = spaceratio*1/(3+2*spaceratio)
    # yoffset = spaceratio*1/(2+3*spaceratio)
    # figfracx = 1/(3+2*spaceratio)
    # figfracy = 1/(2+3*spaceratio)

    xlength = 3.32
    ylength = 0.3
    fg = plt.figure(1, figsize=(cm_to_inch(8.6), cm_to_inch(8.)))
    xoffset = 0.25 * (xlength / 8.6)
    yoffset = 0.18 * (xlength / 8.)
    figfracx = (7.7 - xoffset * 8.6)/8.6
    figfracy = figfracx * 8.6/8.
    # figfracx = 0.7 * (8. / 8.6)
    axg1 = fg.add_axes([xoffset, yoffset + figfracy / 3., figfracx / 3., figfracy / 3.])
    axg2 = fg.add_axes([xoffset, yoffset, figfracx / 3., figfracy / 3.])
    axg3 = fg.add_axes([xoffset + figfracx / 3., yoffset + figfracy / 3., figfracx / 3., figfracy / 3.])
    axg4 = fg.add_axes([xoffset + figfracx / 3., yoffset, figfracx / 3., figfracy / 3.])
    axg5 = fg.add_axes([xoffset + 2 * figfracx / 3., yoffset + figfracy / 3., figfracx / 3., figfracy / 3.])
    axg6 = fg.add_axes([xoffset + 2 * figfracx / 3., yoffset, figfracx / 3., figfracy / 3.])
    # ax1 = fg.add_axis([1/11., 1/11., 3./11., 3./11.])

    # axg1 = fg.add_axes([xoffset, 2*yoffset+figfracy, figfracx, figfracy])
    # axg2 = fg.add_axes([xoffset, yoffset, figfracx, figfracy])
    # axg3 = fg.add_axes([xoffset + figfracx, 2*yoffset+figfracy, figfracx, figfracy])
    # axg4 = fg.add_axes([xoffset+figfracx, yoffset, figfracx, figfracy])
    # axg5 = fg.add_axes([xoffset+2*figfracx, 2*yoffset+figfracy, figfracx, figfracy])
    # axg6 = fg.add_axes([xoffset+2*figfracx, yoffset, figfracx, figfracy])
    axg1.set_xticks(np.arange(len(nflist))[::4])
    axg2.set_xticks(np.arange(len(nflist))[::4])
    axg3.set_xticks(np.arange(len(nflist))[::4])
    axg4.set_xticks(np.arange(len(nflist))[::4])
    axg5.set_xticks(np.arange(len(nflist))[::4])
    axg6.set_xticks(np.arange(len(nflist))[::4])
    axg1.set_yticks(np.arange(len(nhlist))[::5])
    axg2.set_yticks(np.arange(len(nhlist))[::5])
    axg3.set_yticks(np.arange(len(nhlist))[::5])
    axg4.set_yticks(np.arange(len(nhlist))[::5])
    axg5.set_yticks(np.arange(len(nhlist))[::5])
    axg6.set_yticks(np.arange(len(nhlist))[::5])
    axg1.set_xticklabels(nflist[::4], visible=False)
    axg1.set_yticklabels(nhlist[::5])
    axg2.set_xticklabels(nflist[::4])
    axg2.set_yticklabels(nhlist[::5])
    axg3.set_xticklabels(nflist[::4], visible=False)
    axg3.set_yticklabels(nhlist[::5], visible=False)
    axg4.set_xticklabels(nflist[::4])
    axg4.set_yticklabels(nhlist[::5], visible=False)
    axg5.set_xticklabels(nflist[::4], visible=False)
    axg5.set_yticklabels(nhlist[::5], visible=False)
    axg6.set_xticklabels(nflist[::4])
    axg6.set_yticklabels(nhlist[::5], visible=False)

    axg1.set_ylabel('$n_h$', labelpad=0)
    axg2.set_ylabel('$n_h$', labelpad=0)
    axg2.set_xlabel('$n_f$', labelpad=0)
    axg4.set_xlabel('$n_f$', labelpad=0)
    axg6.set_xlabel('$n_f$', labelpad=0)

    caxg = fg.add_axes(
        [xoffset + 3 * figfracx / 3. + 0.01, yoffset, (1 - (figfracx + xoffset)) / 3., figfracy*(2./3.)])
    axs = [axg1, axg2, axg3, axg4, axg5, axg6]
    for i in range(len(axs)):
        aspect = np.shape(dat[i][:, :, 0, 0].T)[1] / np.shape(dat[i][:, :, 0, 0].T)[0]
        axs[i].imshow(dat[i][:, :, 0, 0].T, origin='lower', cmap=cc.cm["diverging_bwr_20_95_c54"], vmin=0, vmax=1,
                      aspect=aspect)
        # axs[i].set_title(r'${:d}\times{:d}$'.format(i+3, i+3))
        # axs[i].legend([r'${:d}\times {:d}$'.format(i+3, i+3)], loc='upper left')
        axs[i].axhline(9., 0, 1, ls='--', lw=0.5, c='black')
        # axs[i].annotate(r'${:d}\times {:d}$'.format(i + 3, i + 3), xy=(0.7, 0.9), xycoords='axes fraction',
        #                 fontsize=6, color='white')
        axs[i].annotate(r'$k = {:d}$'.format(i + 3), xy=(0.7, 0.9), xycoords='axes fraction',
                        fontsize=6, color='white')
        axs[i].set_xticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[1]), minor=True)
        axs[i].set_xticks(np.arange(9)[::3] + 1, minor=False)
        axs[i].set_xticklabels([4, 10, 16], minor=False)
        axs[i].set_yticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[0]), minor=True)
        axs[i].set_yticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[0] - 1)[::4] + 1, minor=False)
        if i + 3 == 4 or i + 3 == 5:
            axs[i].set_yticklabels([4, 12, 20, 60, 100], minor=False)
        else:
            axs[i].set_yticklabels(['', 12, 20, 60, 100], minor=False)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
    fg.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cc.cm["diverging_bwr_20_95_c54"]), cax=caxg)
    caxg.set_title(r'$\langle \mathrm{TCR} \rangle$')

    fg.savefig('.\\figures\\TCR_heatmaps_grid.pdf', facecolor=fg.get_facecolor())
    fg.savefig('.\\figures\\TCR_heatmaps_grid.svg', facecolor=fg.get_facecolor())
    fg.savefig('.\\figures\\TCR_heatmaps_grid.png', facecolor=fg.get_facecolor(), dpi=400)
    plt.show()
    plt.close()

def plot_TIRheatmaps():
    spaceratio = 1. / (2.5)
    Lxratio = (3 + 2 * spaceratio) / (1 + 1 * spaceratio + 3 + 2 * spaceratio)
    Lxlength = 8.6 * Lxratio
    Lylength = (2 + 3 * spaceratio) / (3 + 2 * spaceratio) * Lxlength

    # fg = plt.figure(1, figsize=(cm_to_inch(Lxlength), cm_to_inch(Lylength)))
    # xoffset = spaceratio*1/(3+2*spaceratio)
    # yoffset = spaceratio*1/(2+3*spaceratio)
    # figfracx = 1/(3+2*spaceratio)
    # figfracy = 1/(2+3*spaceratio)

    xlength = 3.32
    ylength = 0.3
    fg = plt.figure(1, figsize=(cm_to_inch(8.6), cm_to_inch(8.)))
    xoffset = 0.25 * (xlength / 8.6)
    yoffset = 0.18 * (xlength / 8.)
    figfracx = (7.7 - xoffset * 8.6)/8.6
    figfracy = figfracx * 8.6/8.
    # figfracx = 0.7 * (8. / 8.6)
    axg1 = fg.add_axes([xoffset, yoffset + figfracy / 3., figfracx / 3., figfracy / 3.])
    axg2 = fg.add_axes([xoffset, yoffset, figfracx / 3., figfracy / 3.])
    axg3 = fg.add_axes([xoffset + figfracx / 3., yoffset + figfracy / 3., figfracx / 3., figfracy / 3.])
    axg4 = fg.add_axes([xoffset + figfracx / 3., yoffset, figfracx / 3., figfracy / 3.])
    axg5 = fg.add_axes([xoffset + 2 * figfracx / 3., yoffset + figfracy / 3., figfracx / 3., figfracy / 3.])
    axg6 = fg.add_axes([xoffset + 2 * figfracx / 3., yoffset, figfracx / 3., figfracy / 3.])
    # ax1 = fg.add_axis([1/11., 1/11., 3./11., 3./11.])

    # axg1 = fg.add_axes([xoffset, 2*yoffset+figfracy, figfracx, figfracy])
    # axg2 = fg.add_axes([xoffset, yoffset, figfracx, figfracy])
    # axg3 = fg.add_axes([xoffset + figfracx, 2*yoffset+figfracy, figfracx, figfracy])
    # axg4 = fg.add_axes([xoffset+figfracx, yoffset, figfracx, figfracy])
    # axg5 = fg.add_axes([xoffset+2*figfracx, 2*yoffset+figfracy, figfracx, figfracy])
    # axg6 = fg.add_axes([xoffset+2*figfracx, yoffset, figfracx, figfracy])
    axg1.set_xticks(np.arange(len(nflist))[::4])
    axg2.set_xticks(np.arange(len(nflist))[::4])
    axg3.set_xticks(np.arange(len(nflist))[::4])
    axg4.set_xticks(np.arange(len(nflist))[::4])
    axg5.set_xticks(np.arange(len(nflist))[::4])
    axg6.set_xticks(np.arange(len(nflist))[::4])
    axg1.set_yticks(np.arange(len(nhlist))[::5])
    axg2.set_yticks(np.arange(len(nhlist))[::5])
    axg3.set_yticks(np.arange(len(nhlist))[::5])
    axg4.set_yticks(np.arange(len(nhlist))[::5])
    axg5.set_yticks(np.arange(len(nhlist))[::5])
    axg6.set_yticks(np.arange(len(nhlist))[::5])
    axg1.set_xticklabels(nflist[::4], visible=False)
    axg1.set_yticklabels(nhlist[::5])
    axg2.set_xticklabels(nflist[::4])
    axg2.set_yticklabels(nhlist[::5])
    axg3.set_xticklabels(nflist[::4], visible=False)
    axg3.set_yticklabels(nhlist[::5], visible=False)
    axg4.set_xticklabels(nflist[::4])
    axg4.set_yticklabels(nhlist[::5], visible=False)
    axg5.set_xticklabels(nflist[::4], visible=False)
    axg5.set_yticklabels(nhlist[::5], visible=False)
    axg6.set_xticklabels(nflist[::4])
    axg6.set_yticklabels(nhlist[::5], visible=False)

    axg1.set_ylabel('$n_h$', labelpad=0)
    axg2.set_ylabel('$n_h$', labelpad=0)
    axg2.set_xlabel('$n_f$', labelpad=0)
    axg4.set_xlabel('$n_f$', labelpad=0)
    axg6.set_xlabel('$n_f$', labelpad=0)

    caxg = fg.add_axes(
        [xoffset + 3 * figfracx / 3. + 0.01, yoffset, (1 - (figfracx + xoffset)) / 3., figfracy*(2./3.)])
    axs = [axg1, axg2, axg3, axg4, axg5, axg6]
    for i in range(len(axs)):
        aspect = np.shape(dat[i][:, :, 1, 0].T)[1] / np.shape(dat[i][:, :, 1, 0].T)[0]
        axs[i].imshow(dat[i][:, :, 1, 0].T, origin='lower', cmap=cc.cm["diverging_bwr_20_95_c54"], vmin=0, vmax=1,
                      aspect=aspect)
        # axs[i].set_title(r'${:d}\times{:d}$'.format(i+3, i+3))
        # axs[i].legend([r'${:d}\times {:d}$'.format(i+3, i+3)], loc='upper left')
        axs[i].axhline(9., 0, 1, ls='--', lw=0.5, c='black')
        # axs[i].annotate(r'${:d}\times {:d}$'.format(i + 3, i + 3), xy=(0.7, 0.9), xycoords='axes fraction',
        #                 fontsize=6, color='white')
        axs[i].annotate(r'$k = {:d}$'.format(i + 3), xy=(0.7, 0.9), xycoords='axes fraction',
                        fontsize=6, color='white')
        axs[i].set_xticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[1]), minor=True)
        axs[i].set_xticks(np.arange(9)[::3] + 1, minor=False)
        axs[i].set_xticklabels([4, 10, 16], minor=False)
        axs[i].set_yticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[0]), minor=True)
        axs[i].set_yticks(np.arange(np.shape(dat[i][:, :, 2, 0].T)[0] - 1)[::4] + 1, minor=False)
        if i + 3 == 4 or i + 3 == 5:
            axs[i].set_yticklabels([4, 12, 20, 60, 100], minor=False)
        else:
            axs[i].set_yticklabels(['', 12, 20, 60, 100], minor=False)

    norm = matplotlib.colors.Normalize(vmin=0, vmax=1, clip=False)
    fg.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cc.cm["diverging_bwr_20_95_c54"]), cax=caxg)
    caxg.set_title(r'$\langle \mathrm{TIR} \rangle$')

    fg.savefig('.\\figures\\TIR_heatmaps_grid.pdf', facecolor=fg.get_facecolor())
    fg.savefig('.\\figures\\TIR_heatmaps_grid.svg', facecolor=fg.get_facecolor())
    fg.savefig('.\\figures\\TIR_heatmaps_grid.png', facecolor=fg.get_facecolor(), dpi=400)
    plt.show()
    plt.close()

plot_BAheatmaps()
plot_TCRheatmaps()
plot_TIRheatmaps()