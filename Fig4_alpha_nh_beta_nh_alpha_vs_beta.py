"Ryan van Mastrigt, 29.07.2022"
"This script plots beta, alpha and alpha vs beta for k=5"

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import math
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import matplotlib
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D

def cm_to_inch(x):
    return x / 2.54

plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')
# from matplotlib.figure import Figure
alpha_klist = []
alpha_var_klist = []
beta_klist = []
beta_var_klist = []
# plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')
# figs, axs = plt.subplots(2, 3, figsize=(cm_to_inch(8.8), cm_to_inch(5.5)), gridspec_kw={'wspace':0.15, 'hspace':0.3})
ylengthfig = 2.8
fg = plt.figure(3, figsize=(cm_to_inch(8.6), cm_to_inch(ylengthfig)))
xlength = 3.32
ylength = 0.3
# f = plt.figure(1, figsize=(cm_to_inch(4.45), cm_to_inch(4.15)))
xoffset = 0.25*(xlength / 8.6)
yoffset = 0.2*(xlength / ylengthfig)
# xoffset = 0.25
# yoffset =
figfracx = (7.8 - 3*xoffset*8.6) / 8.6
figfracy = figfracx*8.6 / (ylengthfig)
# xoffset = 0.2 *
# yoffset = 3./2.*0.09-0.025
# figfrac = 0.825
ax1 = fg.add_axes([xoffset, yoffset, figfracx/3., figfracy/3.])
ax2 = fg.add_axes([2*xoffset+figfracx/3., yoffset, figfracx/3., figfracy/3.])
ax3 = fg.add_axes([3*xoffset+2*figfracx/3., yoffset, figfracx/3., figfracy/3.])
# axg1 = fg.add_axes([xoffset, yoffset+figfracy/3., figfracx/3., figfracy/3.])
# axg2 = fg.add_axes([xoffset, yoffset, figfracx/3., figfracy/3.])
# axg3 = fg.add_axes([xoffset + figfracx/3., yoffset+figfracy/3., figfracx/3., figfracy/3.])
# axg4 = fg.add_axes([xoffset+figfracx/3., yoffset, figfracx/3., figfracy/3.])
# axg5 = fg.add_axes([xoffset+2*figfracx/3., yoffset+figfracy/3., figfracx/3., figfracy/3.])
# axg6 = fg.add_axes([xoffset+2*figfracx/3., yoffset, figfracx/3., figfracy/3.])
ax1.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axg4.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# axg6.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# caxg = fg.add_axes(
#         [xoffset + figfracx + 0.01, yoffset, (0.3/8.6)-0.01, (2./3.)*figfracy])
# caxg = fg.add_axes([xoffset+figfrac+0.009, yoffset, (1-(figfrac+xoffset))/4., figfrac])

xticklist = [0, 50, 100]
yticklist = [0.0, 0.1, 0.2]
ylimlist = [-0.05, 0.29]
# xlimlist = [-0.05, 0.25]
ax1.set_yticks(yticklist)
ax1.set_xticks(xticklist)
ax1.set_ylim(ylimlist)
# axg1.set_xlim(xlimlist)
# axg1.set_xticklabels([])
ax1.set_ylabel(r'$\bar{\beta}-\beta$', labelpad=0)
ax1.set_xlabel('$n_h$', labelpad=0)
ax2.set_yticks(yticklist)
ax2.set_xticks(xticklist)
ax2.set_ylim(ylimlist)
# axg2.set_xlim(xlimlist)
# axg2.set_xticklabels([])
ax2.set_ylabel(r'$\bar{\alpha}-\alpha$', labelpad=0)
ax2.set_xlabel(r'$n_h$', labelpad=0)
ax3.set_yticks(yticklist)
ax3.set_xticks(yticklist)
ax3.set_ylim(ylimlist)
ax3.set_xlim(ylimlist)
# axg3.set_xticklabels([])
# axg3.set_yticklabels([])
ax3.set_ylabel(r'$\bar{\beta}-\beta$', labelpad=0)
ax3.set_xlabel(r'$\bar{\alpha}-\alpha$', labelpad=0)
# axg4.set_yticks(yticklist)
# axg4.set_xticks(xticklist)
# axg4.set_ylim(ylimlist)
# axg4.set_xlim(xlimlist)
# # axg4.set_xticklabels([])
# axg4.set_yticklabels([])
# axg4.set_ylabel('', visible=False, labelpad=0)
# axg4.set_xlabel(r'$\bar{\alpha}-\alpha$', labelpad=0)
# axg5.set_yticks(yticklist)
# axg5.set_xticks(xticklist)
# axg5.set_ylim(ylimlist)
# axg5.set_xlim(xlimlist)
# axg5.set_xticklabels([])
# axg5.set_yticklabels([])
# axg5.set_ylabel('', visible=False, labelpad=0)
# axg5.set_xlabel('', visible=False, labelpad=0)
# axg6.set_yticks(yticklist)
# axg6.set_xticks(xticklist)
# axg6.set_ylim(ylimlist)
# axg6.set_xlim(xlimlist)
# # axg4.set_xticklabels([])
# axg6.set_yticklabels([])
# axg6.set_ylabel('', visible=False, labelpad=0)
# axg6.set_xlabel(r'$\bar{\alpha}-\alpha$', labelpad=0)
# axsg = [axg1, axg2, axg3, axg4, axg5, axg6]
# plt.subplots_adjust(wspace=0.001, hspace=0.001)

# for axi in axs.flat:
#     # axi.set_xscale('log')
#     # axi.set_yscale('log')
#     # axi.set_xticks([10**-1, 10**-2])
#     # axi.set_yticks([10 ** -1, 10 ** -2])
#     axi.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
#     axi.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
#     axi.xaxis.set_major_locator(plt.MaxNLocator(2))
#     axi.yaxis.set_major_locator(plt.MaxNLocator(2))
#     axi.set(xlabel=r'$\bar{\alpha} - \alpha$', ylabel=r'$\bar{\beta} - \beta$')
#     axi.label_outer()

k = 5
# for k in range(3, 9):
    # if k == 3:
    #     fix_nf = 18
    # else:
    #     fix_nf = 20
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
plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')

PM_index = 3
nf_list = np.unique(KFres[:, 0])
nh_list = np.unique(KFres[:, 1])

# random noise, periodic humps via sin
if k > 4:
    if k == 7:
        ConMat = np.load(r'E:\\data\cnniter_HP_GS_SKF_7x7_AB_5050_extended\\testset_nfnhlrkfold_TP_FP_TN_FN.npy')
        results = np.load(
            r'D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\linemode_defects_testsetB_k{:d}_lmin_lmout_original.npz'.
                format(k, k, k))
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
        rw_slope = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\alpha_fit_nf20_mean_var.npz'.
                           format(k, k))
    else:
        rw_slope = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\random_walks\\alpha_fit_nf20_mean_var.npz'.
                           format(k, k))
PM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 2))
if k >3:
    testPM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 7, 2))
else:
    testPM_nfnh = np.zeros((np.shape(nf_list)[0], np.shape(nh_list)[0], 7, 2))
acc_width_nfnhfold = results['accuracy']

for i in range(np.shape(nf_list)[0]):
    # if k == 3 and nf_list[i] == 20:
    #     continue
    for j in range(np.shape(nh_list)[0]):
        # if k == 3 and nh_list[j] == 100:
        #     continue
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
        PM_nfnh[i, j, 1] = KFres[nfnhargs[arg, 0], PM_index + 1]

        TPR = ConMat[nf_arg, nh_arg, lr_arg, :, 4] / (
                    ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 7])
        TNR = ConMat[nf_arg, nh_arg, lr_arg, :, 6] / (
                    ConMat[nf_arg, nh_arg, lr_arg, :, 6] + ConMat[nf_arg, nh_arg, lr_arg, :, 5])
        BA = (TPR + TNR) / 2.

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

nh_index = np.argwhere(nf_list == fix_nf)
beta_class = np.divide(ConMat[nf_arg, nh_arg, lr_arg, :, 4] + ConMat[nf_arg, nh_arg, lr_arg, :, 7], total_pred)
beta_class = beta_class[0]
foldavg_acc_lmin = np.nanmean(acc_width_nfnhfold[0], axis=2)
foldavg_acc_lmin_var = np.nanvar(acc_width_nfnhfold[0], axis=2)
# x=np.array(range(L))
alpha_mean = rw_slope['mean']
alpha_var = rw_slope['var']
alpha_class = np.load('.\\alpha_c_list.npy')

# if k == 3:
#     nh_list = nh_list[:-1]
# plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onethird.mplstyle')
# matplotlib.rcParams['figure.figsize'] = 3.375 / 3., 3.375 / 3.
# f, ax = plt.subplots()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.xaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
cmv = plt.get_cmap('viridis')
# ax.scatter(1-alpha_mean, testPM_nfnh[nh_index[0, 0], :, 5, 0], marker='x', color=cmv(nh_list / 100.))

# alpha_klist.append(alpha_mean)
# alpha_var_klist.append(alpha_var)
# beta_klist.append(testPM_nfnh[nh_index[0, 0], :, 5, 0])
# beta_var_klist.append(testPM_nfnh[nh_index[0, 0], :, 5, 1])

# ax.set_xlabel(r'$\langle \alpha \rangle$')
# ax.set_ylabel(r'$\langle \beta \rangle$')
# f.tight_layout()
# plt.savefig('.\\figures\\alpha_vs_beta_k{:d}.pdf'.format(k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\alpha_vs_beta_k{:d}.svg'.format(k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\alpha_vs_beta_k{:d}.png'.format(k), dpi=400, facecolor=f.get_facecolor())
# # plt.show()
# plt.close()

# f, ax = plt.subplots()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.xaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# # cmv = plt.get_cmap('viridis')
ax1.plot(nh_list, testPM_nfnh[nh_index[0, 0], :, 5, 0]-beta_class, '.-', color='tab:purple')
ax1.fill_between(nh_list, testPM_nfnh[nh_index[0, 0], :, 5, 0]-beta_class + np.sqrt(testPM_nfnh[nh_index[0, 0], :, 5, 1]),
                testPM_nfnh[nh_index[0, 0], :, 5, 0]-beta_class - np.sqrt(testPM_nfnh[nh_index[0, 0], :, 5, 1]),
                alpha=0.4, color='tab:purple')
#
# ax.set_xlabel(r'$n_h$')
# ax.set_ylabel(r'$\bar{\beta} - \beta$')
# f.tight_layout()
# plt.savefig('.\\figures\\beta_min_beta_c_vs_nh_nf{:d}_k{:d}.pdf'.format(fix_nf, k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\beta_min_beta_c_vs_nh_nf{:d}_k{:d}.svg'.format(fix_nf, k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\beta_min_beta_c_vs_nh_nf{:d}_k{:d}.png'.format(fix_nf, k), dpi=400, facecolor=f.get_facecolor())
# # plt.show()
# plt.close()
#
# f, ax = plt.subplots()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.xaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# # cmv = plt.get_cmap('viridis')
ax2.plot(nh_list, alpha_mean - alpha_class[k-3], '.-', color='tab:purple')
ax2.fill_between(nh_list,
                alpha_mean - alpha_class[k-3] + np.sqrt(alpha_var),
                alpha_mean - alpha_class[k-3] - np.sqrt(alpha_var),
                alpha=0.4, color='tab:purple')
#
# ax.set_xlabel(r'$n_h$')
# ax.set_ylabel(r'$\bar{\alpha} - \alpha$')
# f.tight_layout()
# plt.savefig('.\\figures\\alpha_min_alpha_c_vs_nh_nf{:d}_k{:d}.pdf'.format(fix_nf, k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\alpha_min_alpha_c_vs_nh_nf{:d}_k{:d}.svg'.format(fix_nf, k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\alpha_min_alpha_c_vs_nh_nf{:d}_k{:d}.png'.format(fix_nf, k), dpi=400,
#             facecolor=f.get_facecolor())
# # plt.show()
# plt.close()
#
# f, ax = plt.subplots()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.xaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# cmv = plt.get_cmap('viridis')
# ax.scatter(np.add(1-alpha_mean, -(1-alpha_class[k-3])), np.add(testPM_nfnh[nh_index[0, 0], :, 5, 0], -beta_class), marker='x',
#            color=cmv(nh_list / 100.))
# axs[(k - 1) % 2, int((k - 1) / 2) - 1].scatter(np.add(alpha_mean, -(alpha_class[k - 3])),
#                                                np.add(testPM_nfnh[nh_index[0, 0], :, 5, 0], -beta_class),
#                                                marker=(k, 0, 0),
#                                                edgecolors=cmv(nh_list / 100.), facecolors='None', linestyle='None',
#                                                s=20,
#                                                linewidths=.5
#                                                )
# axs[(k + 1) % 2, int((k - 1) / 2.) - 1].set_title(r'${:d} \times {:d}$'.format(k, k))
ax3.scatter(np.add(alpha_mean, -(alpha_class[k - 3])),
                                               np.add(testPM_nfnh[nh_index[0, 0], :, 5, 0], -beta_class),
                                               marker=(k, 0, 0),
                                               edgecolors=cmv(nh_list / 100.), facecolors='None', linestyle='None',
                                               s=20,
                                               linewidths=.5
                                               )
#
# # ax.set_xlabel(r'$\alpha_d - \alpha_c$')
# # ax.set_ylabel(r'$\beta_d - \beta_c$')
# f.tight_layout()
# plt.savefig('.\\figures\\alpha_min_alpha_c_vs_beta_min_beta_c_k{:d}.pdf'.format(k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\alpha_min_alpha_c_vs_beta_min_beta_c_k{:d}.svg'.format(k), facecolor=f.get_facecolor())
# plt.savefig('.\\figures\\alpha_min_alpha_c_vs_beta_min_beta_c_k{:d}.png'.format(k), dpi=400, facecolor=f.get_facecolor())
# # plt.show()
# plt.close()
#
# f, ax = plt.subplots()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.xaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# for i in range(len(alpha_mean)):
#     ax.errorbar(alpha_mean[i], testPM_nfnh[nh_index[0, 0], :, 5, 0][i], np.sqrt(alpha_var[i]) / np.sqrt(10.),
#                 np.sqrt(testPM_nfnh[nh_index[0, 0], :, 5, 1][i]) / np.sqrt(10.), marker='x',
#                 color=cmv(nh_list[i] / 100.))
# # ax.set_xlabel(r'$\langle \alpha \rangle$')
# # ax.set_ylabel(r'$\langle \beta \rangle$')
# plt.tight_layout()
# # plt.savefig('.\\figures\\alpha_vs_beta_k{:d}_errorbar.pdf'.format(k), facecolor=f.get_facecolor())
# # plt.savefig('.\\figures\\alpha_vs_beta_k{:d}_errorbar.svg'.format(k), facecolor=f.get_facecolor())
# # plt.savefig('.\\figures\\alpha_vs_beta_k{:d}_errorbar.png'.format(k), dpi=400, facecolor=f.get_facecolor())
# # plt.show()
# plt.close()
#
# np.savez('.\\alpha_foldavg_var_klist.npz', alpha=alpha_klist, alpha_var=alpha_var_klist)
# np.savez('.\\beta_foldavg_var_klist.npz', beta=beta_klist, beta_var=beta_var_klist)

# figs.subplots_adjust(right=0.9)

# ax1.plot(nh_list, beta)
cbar_ax = fg.add_axes([3*xoffset+figfracx+0.01, yoffset, 0.02, figfracy/3])
cbar_ax.set_title(r'$n_h$')

norm = matplotlib.colors.Normalize(vmin=0, vmax=np.amax(nh_list), clip=False)
fg.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), cax=cbar_ax)
# figs.savefig('.\\figures\\alpha_vs_beta_grid.pdf', facecolor=figs.get_facecolor(), bbox_inches='tight')
# figs.savefig('.\\figures\\alpha_vs_beta_grid.svg', facecolor=figs.get_facecolor(), bbox_inches='tight')
# figs.savefig('.\\figures\\alpha_vs_beta_grid.png', facecolor=figs.get_facecolor(), dpi=400, bbox_inches='tight')
#
# fg.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmv), cax=caxg)
# caxg.set_title('$n_h$')
#
# custom_lines = []
# custom_labels = []
# for k in range(3, 9):
#     custom_lines.append(Line2D([0], [0], marker=(k, 0, 0), markeredgecolor=cmv(2. / 100.), markerfacecolor='None', linestyle='None',
#                                                        markersize=4,
#                                                        markeredgewidth=.5))
#     custom_labels.append('{:d}'.format(k))
# # for i in range(len(custom_lines)):
# #     axsg[i].legend([custom_lines[i]], [custom_labels[i]], loc=(0.02, 0.8), frameon=True, borderpad=0.1,
# #                    handletextpad=0.2)
#
# for i in range(len(custom_lines)):
#     axsg[i].annotate('$k={:d}$'.format(i+3), xy=(0.1, 0.85), xycoords='axes fraction', fontsize=6)
#
# # custom_lines.append(Line2D([0], [0], color='tab:red', lw=1))
# # custom_labels = [r'$\beta_d$ $n_h$ {:d}'.format(i.astype(int)) for i in nh_list]
# # custom_labels = [r'$\bar{\beta}$']
# # custom_labels.append(r'$\beta$')
# # axg6.legend(custom_lines, custom_labels)
fg.savefig('.\\figures\\beta_alpha_beta_vs_alpha_k{:d}.pdf'.format(k), facecolor=fg.get_facecolor())
fg.savefig('.\\figures\\beta_alpha_beta_vs_alpha_k{:d}.svg'.format(k), facecolor=fg.get_facecolor())
fg.savefig('.\\figures\\beta_alpha_beta_vs_alpha_k{:d}.png'.format(k), facecolor=fg.get_facecolor(), dpi=400)
#
# plt.style.use(r'C:/Users/ryanv/PycharmProjects/Matplotlib styles/paper-onehalf.mplstyle')
# f, ax = plt.subplots()
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
# ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.xaxis.set_major_locator(plt.MaxNLocator(2))
# ax.yaxis.set_major_locator(plt.MaxNLocator(2))
# for i in range(len(alpha_klist)):
#     ax.scatter(alpha_klist[i], beta_klist[i], marker=(k, 0, 0), c=cmv(nh_list[:len(alpha_klist[i])] / 100.))
plt.show()
plt.close()
