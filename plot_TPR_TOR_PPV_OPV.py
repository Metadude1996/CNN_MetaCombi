"Ryan van Mastrigt, 31.01.2022"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.lines import Line2D
from scipy import stats
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter

def cm_to_inch(x):
    return x / 2.54

"plot P_ratio, TPR, TOR, PPV, OPV"
hor_length = 5
ver_length = 4

plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')
f, ax = plt.subplots(figsize=(cm_to_inch(hor_length), cm_to_inch(ver_length)))
color = 'tab:green'
ax.set_xlabel('$n_h$')
ax.set_ylabel('TPR', color=color)
# ax1.plot(t, data1, color=color)
ax.tick_params(axis='y', labelcolor=color)

axt = ax.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:red'
axt.set_ylabel('TOR', color=color)  # we already handled the x-label with ax1
axt.tick_params(axis='y', labelcolor=color)
# ax.set_yscale('log')
# axt.set_yscale('log')
# ax.set_ylim([10**(-5), 1])
# axt.set_ylim([10**(-5), 1])
ax.set_ylim([0.45, 1.05])
axt.set_ylim([0.45, 1.05])
plt.tight_layout()

f2, ax2 = plt.subplots(figsize=(cm_to_inch(hor_length), cm_to_inch(ver_length)))
ax2.set_xlabel('$n_h$')
color = 'tab:blue'
ax2.set_xlabel('$n_h$')
ax2.set_ylabel('OPV', color=color)
# ax1.plot(t, data1, color=color)
ax2.tick_params(axis='y', labelcolor=color)

axt2 = ax2.twinx()  # instantiate a second axes that shares the same x-axis

color = 'tab:green'
axt2.set_ylabel('PPV', color=color)  # we already handled the x-label with ax1
axt2.tick_params(axis='y', labelcolor=color)

# ax2.set_yscale('log')
# axt2.set_yscale('log')
# ax2.set_ylim([10**(-5), 1])
# axt2.set_ylim([10**(-5), 1])
ax2.set_ylim([-0.05, 1.05])
axt2.set_ylim([-0.05, 1.05])
plt.tight_layout()

# matplotlib.rcParams['legend.handlelength'] = 0
# matplotlib.rcParams['legend.markerscale'] = 0
f3, ax3 = plt.subplots(figsize=(cm_to_inch(hor_length), cm_to_inch(ver_length)))
color = 'tab:purple'
ax3.set_xlabel('$n_h$')
ax3.set_ylabel(r'$\beta_p - \beta_t$', color=color)
# ax1.plot(t, data1, color=color)
ax3.tick_params(axis='y', labelcolor=color)
ax3.set_xlim([-10, 110])
ax3.set_ylim([-0.01, 0.35])

f4, ax4 = plt.subplots(figsize=(cm_to_inch(hor_length), cm_to_inch(ver_length)))
color = 'tab:purple'
ax4.set_xlabel('$n_h$')
ax4.set_ylabel(r'1-OPV')
ax4.set_yscale('log')

f5, ax5 = plt.subplots(figsize=(cm_to_inch(hor_length), cm_to_inch(ver_length)))
color = 'tab:purple'
ax5.set_xlabel('$n_h$')
ax5.set_ylabel(r'$\frac{\beta_p}{\beta_t}$', color=color)
# ax1.plot(t, data1, color=color)

ax5.tick_params(axis='y', labelcolor=color)
ax5.set_yscale('log')
ax5.axhline(y=1, xmin=0, xmax=1, ls='--', c='y')
ax5.set_yticks([1, 10, 100, 1000])
ax5.set_ylim([0.66, 2000])
ax5.set_xlim([-2.5, 105])

f7, ax7 = plt.subplots(figsize=(cm_to_inch(9.75), cm_to_inch(5.5)))
color='tab:red'
ax7.set_xlabel('$k$')
ax7.set_ylabel(r'$\beta$')
# ax7.tick_params(axis='y', labelcolor=color)
# ax7.set_ylim([-0.02, 0.38])
ax7.set_yscale('log')
# axt7 = ax7.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:purple'
# axt7.set_ylabel(r'$\langle \beta \rangle$', color=color)  # we already handled the x-label with ax1
# axt7.tick_params(axis='y', labelcolor=color)
# axt7.set_ylim([-0.02, 0.38])
# ax7.set_yscale('log')
# ax1.plot(t, data1, color=color)
# ax3.tick_params(axis='y', labelcolor=color)
#
#
# axt = ax.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:red'
# axt.set_ylabel('TOR', color=color)  # we already handled the x-label with ax1
# axt.tick_params(axis='y', labelcolor=color)
# ax.set_yscale('log')
# axt.set_yscale('log')
# ax.set_ylim([10**(-5), 1])
# axt.set_ylim([10**(-5), 1])
# ax.set_yscale('log')
# axt.set_ylim([0.45, 1.05])

f8, ax8 = plt.subplots(figsize=(cm_to_inch(9.75), cm_to_inch(5.5)))
color='tab:red'
ax8.set_xlabel('$k$')
ax8.set_ylabel(r'$\beta$')
# ax7.tick_params(axis='y', labelcolor=color)
ax8.set_ylim([-0.02, 0.38])

f9, ax9 = plt.subplots(figsize=(cm_to_inch(9.75), cm_to_inch(5.5)))
color='tab:red'
ax9.set_xlabel('$k$')
ax9.set_ylabel(r'$\beta$')
# ax7.tick_params(axis='y', labelcolor=color)
ax9.set_ylim([-0.02, 0.38])



figs, axs = plt.subplots(2, 3, figsize=(cm_to_inch(4.5), cm_to_inch(4.0)), gridspec_kw={'wspace':0.1, 'hspace':0.3})
# plt.subplots_adjust(wspace=0.001, hspace=0.001)

for axi in axs.flat:
    axi.set_xscale('log')
    axi.set_yscale('log')
    axi.set_xticks([10**-1, 10**-2])
    axi.set_yticks([10 ** -1, 10 ** -2])
    # axi.xaxis.set_major_locator(plt.MaxNLocator(4))
    # axi.yaxis.set_major_locator(plt.MaxNLocator(4))
    axi.set(xlabel=r'$1-\mathrm{BA}$', ylabel=r'$\bar{\beta} - \beta$')
    axi.label_outer()

# fg = plt.figure(25, figsize=(cm_to_inch(3./2.*3.44), cm_to_inch(3.44)))
# xoffset = 0.26*2./3.
# yoffset = 0.19
# figfracy = 0.7
# figfracx = 0.7*2./3.
xlength = 3.32
ylength = 0.3
fg = plt.figure(25, figsize=(cm_to_inch(3/2*xlength+ylength), cm_to_inch(xlength)))
xoffset = 0.26 *(xlength/((3./2.)*xlength+ylength))
yoffset = 0.18*(xlength/xlength)
figfracy = 0.7 * (xlength / xlength)
figfracx = 0.7 * (xlength / ((3./2.) * xlength + ylength))
axg1 = fg.add_axes([xoffset, yoffset+figfracy/2., figfracx/2., figfracy/2.])
axg2 = fg.add_axes([xoffset, yoffset, figfracx/2., figfracy/2.])
axg3 = fg.add_axes([xoffset + figfracx/2., yoffset+figfracy/2., figfracx/2., figfracy/2.])
axg4 = fg.add_axes([xoffset+figfracx/2., yoffset, figfracx/2., figfracy/2.])
axg5 = fg.add_axes([xoffset+2*figfracx/2., yoffset+figfracy/2., figfracx/2., figfracy/2.])
axg6 = fg.add_axes([xoffset+2*figfracx/2., yoffset, figfracx/2., figfracy/2.])
# axg1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axg2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axg2.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axg4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axg6.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
axg1.set_xscale('log')
axg2.set_xscale('log')
axg3.set_xscale('log')
axg4.set_xscale('log')
axg5.set_xscale('log')
axg6.set_xscale('log')
axg1.set_yscale('log')
axg2.set_yscale('log')
axg3.set_yscale('log')
axg4.set_yscale('log')
axg5.set_yscale('log')
axg6.set_yscale('log')

caxg = fg.add_axes([xoffset+3*figfracx/2.+0.01, yoffset, (1-(figfracx+3*xoffset/2.))/12., figfracy])

yticklist = [10**-3, 10**-1]
xticklist = [10**-3, 10**-1]
ylimlist = [10**-4, 10**0]
xlimlist = [10**-4, 10**0]
axg1.set_yticks(yticklist)
axg1.set_xticks(xticklist)
axg1.set_ylim(ylimlist)
axg1.set_xlim(xlimlist)
axg1.set_xticklabels([])
axg1.set_ylabel(r'$\bar{\beta}-\beta$', labelpad=0.00001)
axg1.set_xlabel('', visible=False, labelpad=0.00001)
axg2.set_yticks(yticklist)
axg2.set_xticks(xticklist)
axg2.set_ylim(ylimlist)
axg2.set_xlim(xlimlist)
# axg2.set_xticklabels([])
axg2.set_ylabel(r'$\bar{\beta}-\beta$', labelpad=0.00001)
axg2.set_xlabel(r'$1-\mathrm{BA}$', labelpad=0.00001)
axg3.set_yticks(yticklist)
axg3.set_xticks(xticklist)
axg3.set_ylim(ylimlist)
axg3.set_xlim(xlimlist)
axg3.set_xticklabels([])
axg3.set_yticklabels([])
axg3.set_ylabel('', visible=False, labelpad=0.00001)
axg3.set_xlabel('', visible=False, labelpad=0.00001)
axg4.set_yticks(yticklist)
axg4.set_xticks(xticklist)
axg4.set_ylim(ylimlist)
axg4.set_xlim(xlimlist)
# axg4.set_xticklabels([])
axg4.set_yticklabels([])
axg4.set_ylabel('', visible=False, labelpad=0.00001)
axg4.set_xlabel(r'$1-\mathrm{BA}$', labelpad=0.00001)
axg5.set_yticks(yticklist)
axg5.set_xticks(xticklist)
axg5.set_ylim(ylimlist)
axg5.set_xlim(xlimlist)
axg5.set_xticklabels([])
axg5.set_yticklabels([])
axg5.set_ylabel('', visible=False, labelpad=0.00001)
axg5.set_xlabel('', visible=False, labelpad=0.00001)
axg6.set_yticks(yticklist)
axg6.set_xticks(xticklist)
axg6.set_ylim(ylimlist)
axg6.set_xlim(xlimlist)
# axg4.set_xticklabels([])
axg6.set_yticklabels([])
axg6.set_ylabel('', visible=False, labelpad=0.00001)
axg6.set_xlabel(r'$1-\mathrm{BA}$', labelpad=0.00001)
axsg = [axg1, axg2, axg3, axg4, axg5, axg6]
# plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')

# f11, ax11 = plt.subplots(figsize=(cm_to_inch(4.0), cm_to_inch(4.3)))
f11 = plt.figure(11, figsize=(cm_to_inch(xlength), cm_to_inch(xlength)))
ax11 = f11.add_axes([0.25, 0.18, 0.7, 0.7])
color='tab:red'
ax11.set_xlabel('$k$', labelpad=0.00001)
ax11.set_ylabel(r'$\beta$', labelpad=0.00001)
ax11.set_yscale('log')
# ax7.tick_params(axis='y', labelcolor=color)
ax11.set_ylim([10**-4, 10**0])
ax11.set_yticks([10**-4, 10**-3, 10**-2, 10**-1, 10**0])

plt.tight_layout()
beta_class_list=[]
for k in range(3, 9):
    if k == 3:
        fix_nf = 20
    else:
        fix_nf = 20


    if k == 5:
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
        strings_old = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_0to1',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_1to2',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_2to3',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_3to4',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_4to5',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_5to6',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_6to7',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_7to8',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_8to9',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_9to10',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_10to11',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_11to12',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_12to13',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_13to14',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_14to15',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_15to16',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_16to17',
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_7x7_AB_5050\\scratch\\output_dir\\cnniter_HP_GS_SKF_7x7_AB_5050_nh_17to18'
        ]
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
    elif k == 3:
        strings = [
            u'C:\\Users\\ryanv\\PycharmProjects\\MetaCombiNN\\cnniter_HP_GS_SKF_3x3_AB_5050_2']
    elif k == 4:
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
        else:
            KFres = np.append(KFres, np.loadtxt(resultstring + u'\\kfold_avg_val_results.txt', delimiter='\t'), axis=0)
    "fold-averaged val_acc as performance measure"
    plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')

    PM_index = 3
    nf_list = np.unique(KFres[:, 0])
    nh_list = np.unique(KFres[:, 1])

    len_nh_list = len(nh_list)

    # random noise, periodic humps via sin
    if k > 4:
        if k == 7:
            ConMat = np.load(r'E:\\data\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\testset_nfnhlrkfold_TP_FP_TN_FN.npy')
        else:
            ConMat = np.load(r'D:\\data\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\testset_nfnhlrkfold_TP_FP_TN_FN.npy'.format(k, k))
        results = np.load(
            r'D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\linemode_defects_testsetB_k{:d}_lmin_lmout_original.npz'.
                format(k, k, k))
        rw_slope = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\random_walks\\alpha_fit_nf20_mean_var.npz'.format(k, k))
    else:
        ConMat = np.load(r'D:\\data\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\testset_nfnhlrkfold_TP_FP_TN_FN.npy'.
                         format(k, k))
        results = np.load(
            r'D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\linemode_defects_testsetB_k{:d}_lmin_lmout_original.npz'.
                format(k, k, k))
        if k == 3:
            rw_slope = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\alpha_fit_nf18_mean_var.npz'.
                               format(k, k))
        else:
            rw_slope = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\alpha_fit_nf20_mean_var.npz'.
                               format(k, k))
    PM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 2))
    testPM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 7, 2))
    acc_width_nfnhfold = results['accuracy']

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

            PPV = np.divide(ConMat[nf_arg, nh_arg, lr_arg, :, 4], ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 5],
                            out=np.zeros_like(ConMat[nf_arg, nh_arg, lr_arg, :, 4]),
                            where=ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 5]!=0)
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

    nf_index = np.argwhere(nf_list == fix_nf)[0, 0]
    beta_class = np.divide(ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 7], total_pred)
    beta_class = beta_class[0]
    beta_class_list.append(beta_class)
    foldavg_acc_lmin = np.nanmean(acc_width_nfnhfold[0], axis=2)
    foldavg_acc_lmin_var = np.nanvar(acc_width_nfnhfold[0], axis=2)
    # x=np.array(range(L))

    f6, ax6 = plt.subplots(figsize=(cm_to_inch(hor_length), cm_to_inch(ver_length)))
    color = 'tab:green'
    ax6.set_xlabel('$n_h$')
    ax6.set_ylabel('TPR', color=color)
    # ax1.plot(t, data1, color=color)
    ax6.tick_params(axis='y', labelcolor=color)

    axt6 = ax6.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    axt6.set_ylabel('TOR', color=color)  # we already handled the x-label with ax1
    axt6.tick_params(axis='y', labelcolor=color)
    # ax.set_yscale('log')
    # axt.set_yscale('log')
    # ax.set_ylim([10**(-5), 1])
    # axt.set_ylim([10**(-5), 1])
    ax6.set_ylim([0.45, 1.05])
    axt6.set_ylim([0.45, 1.05])
    f6.tight_layout()

    cmg = plt.get_cmap('Greens')
    cmr = plt.get_cmap('Reds')
    cmb = plt.get_cmap('Blues')
    cmo = plt.get_cmap('Oranges')
    cmp = plt.get_cmap('Purples')
    cmv = plt.get_cmap('viridis')

    ax.errorbar(nh_list, testPM_nfnh[nf_index, :, 0, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 0, 1]),
                marker=(k, 0, 0), c=cmg((k)/9.))

    # axt.errorbar(nh_list, testPM_nfnh[nf_index, :, 1, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 1, 1]),
    #             marker=(k, 0, 0), c=cmr((k) / 9.))
    axt.errorbar(nh_list, testPM_nfnh[nf_index, :, 1, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 3, 1]),
                             marker=(k, 0, 0), c=cmr((k) / 9.))

    ax2.errorbar(nh_list, testPM_nfnh[nf_index, :, 4, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 4, 1]),
                marker=(k, 0, 0), c=cmb((k) / 9.))

    axt2.errorbar(nh_list, testPM_nfnh[nf_index, :, 3, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 3, 1]),
                 marker=(k, 0, 0), c=cmg((k) / 9.))
    # ax.plot(nh_list, testPM_nfnh[nf_index, :, 3], c=cmb((k - 2) / 7.), label='PPV')
    #
    # ax.plot(nh_list, testPM_nfnh[nf_index, :, 4], c=cmo((k - 2) / 7.), label='OPV')
    #
    # ax.plot(nh_list, testPM_nfnh[nf_index, :, 5], c=cmb((k - 2) / 7.), label=r'$\langle \beta \rangle$')
    ax3.errorbar(nh_list, testPM_nfnh[nf_index, :, 5, 0] - beta_class, yerr=np.sqrt(testPM_nfnh[nf_index, :, 5, 1]),
                 marker=(k, 0, 0), c=cmp((k) / 9.), label=k)
    ax4.errorbar(nh_list, 1-testPM_nfnh[nf_index, :, 4, 0], yerr=testPM_nfnh[nf_index, :, 4, 1], marker=(k, 0, 0),
                 c=cmb(k/ 9.), label=k)
    ax5.errorbar(nh_list, np.divide(testPM_nfnh[nf_index, :, 5, 0], beta_class),
                 yerr=np.sqrt(testPM_nfnh[nf_index, :, 5, 1]) / beta_class,
                 marker=(k, 0, 0), c=cmp((k) / 9.), label=k)

    ax6.errorbar(nh_list, testPM_nfnh[nf_index, :, 0, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 0, 1]),
                marker=(k, 0, 0), c=cmg((k) / 9.))

    # axt.errorbar(nh_list, testPM_nfnh[nf_index, :, 1, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 1, 1]),
    #             marker=(k, 0, 0), c=cmr((k) / 9.))
    axt6.errorbar(nh_list, testPM_nfnh[nf_index, :, 1, 0], yerr=np.sqrt(testPM_nfnh[nf_index, :, 3, 1]),
                 marker=(k, 0, 0), c=cmr((k) / 9.))


    ax7.scatter(np.full(len(nh_list), k), testPM_nfnh[nf_index, :, 5, 0], edgecolors=cmv(nh_list / 100.),
                marker=(k, 0, 0), facecolors='None', linestyle='None', s=40, linewidths=1.)
    # ax7.scatter(k, beta_class, edgecolors='tab:red', marker=(k, 0, 0), facecolors='None', linestyle='None', s=40,
    #             linewidths=1.5)
    ax7.hlines(beta_class, xmin=k - .5, xmax=k + .5, colors='tab:red', linewidths=1.5)
    # axt7.scatter(k, beta_class, edgecolors='tab:red', marker=(k, 0, 0), facecolors='None', linestyle='None')

    ax8.bar(np.full(len(nh_list), k), testPM_nfnh[nf_index, :, 5, 0], color=cmv(nh_list / 100.), width=1)
    ax8.bar(k, beta_class, color='tab:red', width=1)

    ax9.hlines(testPM_nfnh[nf_index, :, 5, 0], xmin=k-.5, xmax=k+.5, colors=cmv(nh_list / 100.))
    ax9.hlines(beta_class, xmin=k-.5, xmax=k+.5, colors='tab:red')

    ax11.scatter(np.full(len(nh_list), k), testPM_nfnh[nf_index, :, 5, 0], edgecolors=cmv(nh_list / 100.),
                marker=(k, 0, 0), facecolors='None', linestyle='None', s=20, linewidths=.5)
    ax11.hlines(beta_class, xmin=k - .5, xmax=k + .5, colors='tab:red')

    f10, ax10 = plt.subplots(figsize=(3.376 / 3., 3.375 / 3.))
    # color='tab:red'
    # ax10.set_xlabel('$BA$')
    # ax10.set_ylabel(r'$\bar{\beta}$')
    # ax7.tick_params(axis='y', labelcolor=color)
    # ax10.set_ylim([-0.02, 0.38])
    ax10.set_yscale('log')
    ax10.set_xscale('log')
    # ax10.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    # ax10.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    # ax10.xaxis.set_major_locator(plt.MaxNLocator(2))
    # ax10.yaxis.set_major_locator(plt.MaxNLocator(2))
    cmv = plt.get_cmap('viridis')
    ax10.scatter(1-testPM_nfnh[nf_index, :, 2, 0], testPM_nfnh[nf_index, :, 5, 0]-beta_class, marker='x', c=cmv(nh_list/100.))
    res = stats.linregress(np.log(1-testPM_nfnh[nf_index, :, 2, 0]), np.log(testPM_nfnh[nf_index, :, 5, 0]-beta_class))
    axs[(k+1) % 2, int((k-1)/2.)-1].scatter(1-testPM_nfnh[nf_index, :, 2, 0], testPM_nfnh[nf_index, :, 5, 0]-beta_class,
                                            edgecolors=cmv(nh_list / 100.),
                                            marker=(k, 0, 0), facecolors='None', linestyle='None', s=20, linewidths=.5)
    axs[(k+1)%2, int((k-1)/2.)-1].set_title(r'${:d} \times {:d}$'.format(k, k))
    print('BA vs beta slope: {:.4f}, {:.4f}, {:.4f}'.format(res.slope, res.intercept, res.rvalue))
    axsg[k-3].scatter(1-testPM_nfnh[nf_index, :, 2, 0], testPM_nfnh[nf_index, :, 5, 0]-beta_class,
                                            edgecolors=cmv(nh_list / 100.),
                                            marker=(k, 0, 0), facecolors='None', linestyle='None', s=20, linewidths=.5)
    f3.tight_layout()
    f3.savefig('.\\figures\\beta_vs_nh_uptok{:d}.pdf'.format(k), facecolor=f3.get_facecolor())
    f3.savefig('.\\figures\\beta_vs_nh_uptok{:d}.svg'.format(k), facecolor=f3.get_facecolor())
    f3.savefig('.\\figures\\beta_vs_nh_uptok{:d}.png'.format(k), facecolor=f3.get_facecolor(), dpi=400)
    f5.tight_layout()
    f5.savefig('.\\figures\\beta_frac_uptok{:d}.pdf'.format(k), facecolor=f5.get_facecolor())
    f5.savefig('.\\figures\\beta_frac_uptok{:d}.svg'.format(k), facecolor=f5.get_facecolor())
    f5.savefig('.\\figures\\beta_frac_uptok{:d}.png'.format(k), facecolor=f5.get_facecolor(), dpi=400)
    f6.tight_layout()
    f6.savefig('.\\figures\\TPR_TOR_vs_nh_k{:d}.pdf'.format(k), facecolor=f6.get_facecolor())
    f6.savefig('.\\figures\\TPR_TOR_vs_nh_k{:d}.svg'.format(k), facecolor=f6.get_facecolor())
    f6.savefig('.\\figures\\TPR_TOR_vs_nh_k{:d}.png'.format(k), facecolor=f6.get_facecolor(), dpi=400)
    f10.tight_layout()
    f10.savefig('.\\figures\\BA_vs_beta_k{:d}.pdf'.format(k), facecolor=f10.get_facecolor())
    f10.savefig('.\\figures\\BA_vs_beta_k{:d}.svg'.format(k), facecolor=f10.get_facecolor())
    f10.savefig('.\\figures\\BA_vs_beta_k{:d}.png'.format(k), facecolor=f10.get_facecolor(), dpi=400)

# ax.set_xlabel('$n_h$')
# plt.legend()
# ax.set_ylim([-0.05, 1.05])
# axt.set_ylim([-0.05, 1.05])
# ax2.set_ylim([-0.05, 1.05])
# axt2.set_ylim([-0.05, 1.05])

# plt.tight_layout()
# handles, labels = ax3.get_legend_handles_labels()
# for h in handles:
#     h.set_linestyle("")
# ax3.legend(handles, labels)
handles, labels = ax3.get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
for h in handles:
    h.set_linestyle("")
# use them in the legend
ax3.legend(handles, labels, loc='best', numpoints=1, frameon=False)
for h in handles:
    h.set_linestyle("-")

# figs.tight_layout(pad=0.3)
figs.subplots_adjust(right=0.9)
cbar_ax = figs.add_axes([0.92, 0.11, 0.02, 0.77])
cbar_ax.set_title(r'$n_h$')
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.amax(nh_list), clip=False)
figs.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), cax=cbar_ax)
# figs.savefig('.\\figures\\BA_vs_beta_grid.pdf', facecolor=figs.get_facecolor())
# figs.savefig('.\\figures\\BA_vs_beta_grid.svg', facecolor=figs.get_facecolor())
# figs.savefig('.\\figures\\BA_vs_beta_grid.png', facecolor=figs.get_facecolor(), dpi=400)

fg.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), cax=caxg)
caxg.set_title('$n_h$')

custom_lines = []
custom_labels = []
for k in range(3, 9):
    custom_lines.append(Line2D([0], [0], marker=(k, 0, 0), markeredgecolor=cmv(2. / 100.), markerfacecolor='None', linestyle='None',
                                                       markersize=4,
                                                       markeredgewidth=.5))
    custom_labels.append('{:d}'.format(k))
# custom_lines.append(Line2D([0], [0], color='tab:red', lw=1))
# custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
# custom_labels = [r'$\bar{\beta}$']
# custom_labels.append(r'$\beta$')
# for i in range(len(custom_lines)):
#     axsg[i].legend([custom_lines[i]], [custom_labels[i]], loc=(0.02, 0.7), frameon=True, borderpad=0.1, handletextpad=0.2)

for i in range(6):
    axsg[i].annotate('$k={:d}$'.format(i+3), xy=(0.05, 0.8), fontsize=6, xycoords='axes fraction')
# axg6.legend(custom_lines, custom_labels)
fg.savefig('.\\figures\\BA_vs_beta_grid.pdf', facecolor=figs.get_facecolor())
fg.savefig('.\\figures\\BA_vs_beta_grid.svg', facecolor=figs.get_facecolor())
fg.savefig('.\\figures\\BA_vs_beta_grid.png', facecolor=figs.get_facecolor(), dpi=400)

f.savefig('.\\figures\\TPR_TOR_vs_nh.pdf', facecolor=f.get_facecolor())
f.savefig('.\\figures\\TPR_TOR_vs_nh.svg', facecolor=f.get_facecolor())
f.savefig('.\\figures\\TPR_TOR_vs_nh.png', facecolor=f.get_facecolor(), dpi=400)

f2.savefig('.\\figures\\PPV_OPV_vs_nh.pdf', facecolor=f2.get_facecolor())
f2.savefig('.\\figures\\PPV_OPV_vs_nh.svg', facecolor=f2.get_facecolor())
f2.savefig('.\\figures\\PPV_OPV_vs_nh.png', facecolor=f2.get_facecolor(), dpi=400)
#
f3.savefig('.\\figures\\beta_vs_nh.pdf', facecolor=f3.get_facecolor())
f3.savefig('.\\figures\\beta_vs_nh.svg', facecolor=f3.get_facecolor())
f3.savefig('.\\figures\\beta_vs_nh.png', facecolor=f3.get_facecolor(), dpi=400)

# handles, labels = ax4.get_legend_handles_labels()
# # remove the errorbars
# handles = [h[0] for h in handles]
# for h in handles:
#     h.set_linestyle("")
# # use them in the legend
# ax4.legend(handles, labels, loc='best', numpoints=1, frameon=False)
# for h in handles:
#     h.set_linestyle("-")

f4.savefig('.\\figures\\OPV_vs_nh.pdf', facecolor=f4.get_facecolor())
f4.savefig('.\\figures\\OPV_vs_nh.svg', facecolor=f4.get_facecolor())
f4.savefig('.\\figures\\OPV_vs_nh.png', facecolor=f4.get_facecolor(), dpi=400)

handles, labels = ax5.get_legend_handles_labels()
# remove the errorbars
handles = [h[0] for h in handles]
for h in handles:
    h.set_linestyle("")
# use them in the legend
ax5.legend(handles, labels, loc='best', numpoints=1, frameon=False)
for h in handles:
    h.set_linestyle("-")

ax5.set_yticks([1, 10, 100, 1000])
# ax5.set_yticklabels([10**0, 10**1, 10**2, 10**3])
f5.tight_layout()
f5.savefig('.\\figures\\beta_frac_vs_nh.pdf', facecolor=f5.get_facecolor())
f5.savefig('.\\figures\\beta_frac_vs_nh.svg', facecolor=f5.get_facecolor())
f5.savefig('.\\figures\\beta_frac_vs_nh.png', facecolor=f5.get_facecolor(), dpi=400)

# ax7.scatter(np.arange(3, 9, 2), beta_class_list[0::2], c='tab:red', marker=(k, 0, 0), linestyle='None')
# ax7.scatter(np.arange(4, 9, 2), beta_class_list[1::2], c='tab:red', marker=(k, 0, 0), linestyle='None')
ax7.set_xlabel('$k$')
# ax7.set_ylabel(r'$\alpha$')
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.amax(nh_list), clip=False)
# cax7 = f7.add_axis()
cb = f7.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), ax=ax7, pad=0.03)

# custom_lines = [Line2D([0], [0], color=cmv(i/len(nh_list)), lw=0.5) for i in range(len(nh_list))]
custom_lines = [Line2D([0], [0], color=cmv(0), lw=1.)]
custom_lines.append(Line2D([0], [0], color='tab:red', lw=1))
# custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
custom_labels = [r'$\bar{\beta}$']
custom_labels.append(r'$\beta$')
ax7.legend(custom_lines, custom_labels)

cb.ax.set_title('$n_h$')
f7.tight_layout()
f7.savefig('.\\figures\\beta_c_beta_vs_nh.pdf', facecolor=f7.get_facecolor())
f7.savefig('.\\figures\\beta_c_beta_vs_nh.svg', facecolor=f7.get_facecolor())
f7.savefig('.\\figures\\beta_c_beta_vs_nh.png', facecolor=f7.get_facecolor(), dpi=400)
# plt.show()
# plt.close()

ax8.set_xlabel('$k$')
# ax7.set_ylabel(r'$\alpha$')
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.amax(nh_list), clip=False)
# cax7 = f7.add_axis()
cb8 = f8.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), ax=ax8, pad=0.05)

# custom_lines = [Line2D([0], [0], color=cmv(i/len(nh_list)), lw=0.5) for i in range(len(nh_list))]
custom_lines = [Line2D([0], [0], color=cmv(0), lw=1.)]
custom_lines.append(Line2D([0], [0], color='tab:red', lw=1))
# custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
custom_labels = [r'$\bar{\beta}$']
custom_labels.append(r'$\beta$')
ax8.legend(custom_lines, custom_labels)

cb8.ax.set_title('$n_h$')
f8.tight_layout()
f8.savefig('.\\figures\\beta_c_beta_vs_nh_barplot.pdf', facecolor=f7.get_facecolor())
f8.savefig('.\\figures\\beta_c_beta_vs_nh_barplot.svg', facecolor=f7.get_facecolor())
f8.savefig('.\\figures\\beta_c_beta_vs_nh_barplot.png', facecolor=f7.get_facecolor(), dpi=400)

ax9.set_xlabel('$k$')
# ax7.set_ylabel(r'$\alpha$')
norm = matplotlib.colors.Normalize(vmin=0, vmax=np.amax(nh_list), clip=False)
# cax7 = f7.add_axis()
cb9 = f9.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), ax=ax9, pad=0.05)

# custom_lines = [Line2D([0], [0], color=cmv(i/len(nh_list)), lw=0.5) for i in range(len(nh_list))]
custom_lines = [Line2D([0], [0], color=cmv(0), lw=1.)]
custom_lines.append(Line2D([0], [0], color='tab:red', lw=1))
# custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
custom_labels = [r'$\bar{\beta}$']
custom_labels.append(r'$\beta$')
ax9.legend(custom_lines, custom_labels)

cb9.ax.set_title('$n_h$')
f9.tight_layout()
f9.savefig('.\\figures\\beta_c_beta_vs_nh_hlines.pdf', facecolor=f9.get_facecolor())
f9.savefig('.\\figures\\beta_c_beta_vs_nh_hlines.svg', facecolor=f9.get_facecolor())
f9.savefig('.\\figures\\beta_c_beta_vs_nh_hlines.png', facecolor=f9.get_facecolor(), dpi=400)

ax11.set_xticks([3, 4, 5, 6, 7, 8])
ax11.legend(custom_lines, custom_labels)

# f11.tight_layout()
f11.savefig('.\\figures\\beta_c_beta_vs_nh_small.pdf', facecolor=f11.get_facecolor())
f11.savefig('.\\figures\\beta_c_beta_vs_nh_small.svg', facecolor=f11.get_facecolor())
f11.savefig('.\\figures\\beta_c_beta_vs_nh_small.png', facecolor=f11.get_facecolor(), dpi=400)
plt.show()
plt.close()