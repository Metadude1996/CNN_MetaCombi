"Ryan van Mastrigt, 29.07.2022"
"This script preprocesses the pentodal metamaterial data for neural network training for classification (ii)"

import numpy as np
SEED = 0
np.random.seed(SEED)
# tf.random.set_seed(SEED)
# os.environ['PYTHONHASHSEED'] = str(SEED)

def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def translate_config(config, k):
    "input: config(2k, 2k) & k, translate configuration in x and y direction"
    configtile = np.tile(config, (2, 2))
    translconfs = np.zeros((k*k, np.shape(config)[0], np.shape(config)[1]), dtype=int)
    for x in range(k):
        for y in range(k):
            index = x*k + y
            translconfs[index] = configtile[2*x:2*x+2*k, 2*y:2*y+2*k]
    return translconfs[1::]

def transrotconfigs(pixreps, resultstotal, k, extended=False):
    # pixreps = np.loadtxt('results\\modescaling\\PixelRep_{:d}x{:d}.txt'.format(k, k), delimiter=',')
    # resultstotal = np.loadtxt('results\\modescaling\\results_analysis_i_Scen_slope_offset_M1k_{:d}x{:d}_fixn4.txt'.format(k, k), delimiter=',')
    indB = np.argwhere(resultstotal[:, 1]==1)[:, 0]
    pixreps = np.reshape(pixreps, (-1, 2*k, 2*k))
    pixrepappend = pixreps.copy()
    restotalappend = resultstotal.copy()
    for i, ind in enumerate(indB):
        config = pixreps[ind]
        tconfs = translate_config(config, k)
        pixrepappend = np.append(pixrepappend, tconfs, axis=0)
        restotalappend = np.append(restotalappend, np.tile(restotalappend[ind], (np.shape(tconfs)[0], 1)), axis=0)
    "rotate the configs"
    indBapp = np.argwhere(restotalappend[:, 1]==1)[:, 0]
    # rotconfs = pixrepappend[indBapp]
    for r in range(1, 4):
        rotconfs = np.rot90(pixrepappend[indBapp], k=r, axes=(1, 2))
        pixrepappend = np.append(pixrepappend, rotconfs, axis=0)
        restotalappend = np.append(restotalappend, restotalappend[indBapp], axis=0)
    "remove duplicate configs"
    pixrepappend = np.reshape(pixrepappend, (-1, 2*k*2*k))
    pixrepfinal, indices = np.unique(pixrepappend, axis=0, return_index=True)
    resfinal = restotalappend[indices]
    if extended:
        np.save('cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\PixelRep_{:d}x{:d}_xrest_rottrans_extended.npy'.format(k, k, k, k)
                , pixrepfinal)
        if k == 5:
            np.save(
                'cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\results_analysis_i_Scen_slope_offset_M1k_{:d}x{:d}_xrest_fixn4_'
                'rottrans_extended.npy'.format(
                    k, k, k, k), resfinal)
        else:
            np.save(
                'cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\results_analysis_i_Scen_slope_M1k_{:d}x{:d}_xrest_rottrans_'
                'extended.npy'.format(
                    k, k, k, k), resfinal)
    else:
        np.save('cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\PixelRep_{:d}x{:d}_xrest_rottrans.npy'.format(k, k, k, k),
                pixrepfinal)
        if k==5:
            np.save('cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\results_analysis_i_Scen_slope_offset_M1k_{:d}x{:d}_xrest_fixn4_rottrans.npy'.format(k, k, k, k), resfinal)
        else:
            np.save(
                'cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\results_analysis_i_Scen_slope_M1k_{:d}x{:d}_xrest_rottrans.npy'.format(
                    k, k, k, k), resfinal)
    return pixrepfinal, resfinal

extended=False
k=5
filter_size = (2, 2)
stride = (2, 2)

if k < 5:
    datastring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\PixelRep_{:d}x{:d}.txt'.format(k, k)
else:
    datastring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\PixelRep_{:d}x{:d}.npy'.format(k, k)
datastring_extended = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\PixelRep_{:d}x{:d}_extended.npy'.\
    format(k, k)
if k==5:
    resultsstring=r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_Scen_slope_modes_M1k_5x5_fixn4.txt'
elif k>6:
    resultsstring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_Scen_slope_modes_M1k_{:d}x{:d}_fixn4.txt'.format(
        k, k)
    resultsstring_extended = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_vs_oligomodal_vs_plurimodal_i_' \
                             r'Scen_slope_modes_M1k_{:d}x{:d}_extended.txt'.format(
        k, k)
else:
    resultsstring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\results\modescaling\results_analysis_unimodal_vs_' \
                    r'oligomodal_vs_plurimodal_i_Scen_slope_modes_M1k_{:d}x{:d}_fixn4.txt'.format(k, k)

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
if k>6:
    results = np.loadtxt(resultsstring, delimiter=',')
    if extended:
        results = np.append(results, np.loadtxt(resultsstring_extended, delimiter=','), axis=0)
else:
    results = np.loadtxt(resultsstring, delimiter=',')
data = np.reshape(data, (-1, 2*k, 2*k))
x_total = data

"select only unimodal vs oligomodal (include stripmodes)"
ind_U = np.argwhere(results[:, 3].astype(int) == 1)
ind_O = np.argwhere(results[:, 3].astype(int) > 1)
ind_UO = np.append(ind_U, ind_O, axis=0)
x_total = x_total[ind_UO[:, 0]]
y_total = np.full_like(ind_U[:, 0], 0)
y_total = np.append(y_total, np.full_like(ind_O[:, 0], 1))
# y_total = results[ind_UO[:, 0], 1].astype(int)

"create a split of the data"
test_frac = 0.15
rest_frac = 1-test_frac

"create test set from original data (no rot + transl append)"
test_ind = np.arange(np.shape(x_total)[0])
np.random.shuffle(test_ind)
test_ind = test_ind[:int(test_frac*np.shape(x_total)[0])]
test_set = x_total[test_ind]
test_y = y_total[test_ind].astype(int)
rest_set = np.delete(x_total, test_ind, axis=0)
rest_y = np.delete(y_total, test_ind, axis=0)

"remove the C configs"
# indCt = np.argwhere(test_y==2)[:, 0]
# indCr = np.argwhere(rest_results[:, 1]==2)[:, 0]
# test_set = np.delete(test_set, indCt, axis=0)
# test_y = np.delete(test_y, indCt, axis=0)
#
# rest_set = np.delete(rest_set, indCr, axis=0)
# rest_results = np.delete(rest_results, indCr, axis=0)

"add translations + rotations to rest set"
# rest_set, rest_results = transrotconfigs(rest_set, rest_results, k, extended=extended)
# # rest_set = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\PixelRep_{:d}x{:d}_xrest_rottrans_extended.npy'
# #                    .format(k, k, k, k))
# # rest_results = np.load('.\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\results_analysis_i_Scen_slope_M1k_{:d}x{:d}_xrest'
# #                        '_rottrans_extended.npy'.format(k, k, k, k))
#
# rest_set = np.reshape(rest_set, (-1, 2*k, 2*k))
# rest_y = rest_results[:, 1].astype(int)

"create periodic padding (make sure to set conv2d padding to valid!)"
# x_rest = np.tile(rest_set, (1, 3, 3))
# x_test = np.tile(test_set, (1, 3, 3))
restshape = np.shape(rest_set)
testshape = np.shape(test_set)

ind_horizontal = np.arange(0, restshape[1])
ind_horizontal = np.tile(ind_horizontal, 3)
ind_vertical = np.arange(0, restshape[2])
ind_vertical = np.tile(ind_vertical, 3)

# x_rest = np.broadcast_to(rest_set, (restshape[0], 3*restshape[1], 3*testshape[2]))
# x_test = np.broadcast_to(test_set, (testshape[0], 3*restshape[1], 3*testshape[2]))
# check how far the filter reaches with given stride and filter size over standard increased image of original
# size + filter_size. Add this length to the image if the extra length is not a multiple of k
x_over = ((filter_size[0]+np.ceil((2*k)/stride[0])*stride[0]) % (2*k+filter_size[0])) % k
y_over = ((filter_size[1]+np.ceil((2*k)/stride[1])*stride[1]) % (2*k+filter_size[1])) % k
x_left = np.floor(2*k-0.5*filter_size[0]).astype(int)
x_right = np.ceil(4*k+0.5*filter_size[0]+x_over).astype(int)
y_down = np.floor(2*k-0.5*filter_size[0]).astype(int)
y_up = np.ceil(4*k + 0.5*filter_size[0]+y_over).astype(int)

ind_horizontal = ind_horizontal[x_left:x_right]
ind_vertical = ind_vertical[y_down:y_up]

# x_rest = x_rest[:, x_left:x_right, y_down:y_up]
# x_test = x_test[:, x_left:x_right, y_down:y_up]
x_rest = np.zeros((restshape[0], restshape[1]+2, restshape[2] + 2))
x_test = np.zeros((testshape[0], testshape[1] + 2, testshape[2] + 2))
for i, ind in enumerate(ind_horizontal):
    x_rest[:, i, :] = rest_set[:, ind, ind_vertical]
    x_test[:, i, :] = test_set[:, ind, ind_vertical]
# x_rest = rest_set[:, ind_horizontal, ind_vertical]
# x_test = test_set[:, ind_horizontal, ind_vertical]
rest_set = x_rest.copy()

"make 5050 distribution rest set"
x_rest, rest_y = unison_shuffled_copies(x_rest, rest_y)
ind_U = np.argwhere(rest_y == 0)
ind_O = np.argwhere(rest_y == 1)
ind_U_short = ind_U[:int(ind_O.shape[0]), 0]
ind_O_short = ind_O[:int(ind_U.shape[0]), 0]
indices = np.append(ind_U_short, ind_O_short)
x_rest, y_rest = x_rest[indices], rest_y[indices]

"add channel axis"
x_rest = x_rest[..., np.newaxis]
rest_set = rest_set[..., np.newaxis]
x_test = x_test[..., np.newaxis]

"save the data"
if extended:
    np.savez('data_unimodal_vs_oligomodal_inc_stripmodes_train_trainraw_test_{:d}x{:d}_extended.npz'.format(k, k,
                                                                                                                   k, k)
             , x_rest=x_rest, y_rest=y_rest, x_rest_raw=rest_set, y_rest_raw=rest_y, x_test=x_test, y_test=test_y)
else:
    np.savez('data_unimodal_vs_oligomodal_inc_stripmodes_train_trainraw_test_{:d}x{:d}.npz'.format(k, k, k, k),
             x_rest=x_rest, y_rest=y_rest, x_rest_raw = rest_set, y_rest_raw = rest_y, x_test = x_test, y_test=test_y)
# np.savez(os.path.join(masterdir, 'data_prek_xy_train_trainraw_test.npz'), x_rest=x_rest, y_rest=y_rest, x_rest_raw=x_rest_raw, y_rest_raw=y_rest_raw, x_test=x_test, y_test=y_test)