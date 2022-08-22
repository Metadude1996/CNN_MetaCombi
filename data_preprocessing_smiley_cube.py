"Ryan van Mastrigt, 29.07.2022"
"This script preprocesses the smiley cube data for neural network training"

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
stride = (1, 1)
if k < 6:
    datastring = r'C:\\Users\\ryanv\\PycharmProjects\\MetaCombi\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz'.\
        format(k, k)
else:
    datastring = r'C:\Users\ryanv\PycharmProjects\MetaCombi\\smiley cube\\smiley_cube_uniform_sample_x_y_{:d}x{:d}.npz'.\
        format(k, k)

dataz = np.load(datastring)

if k < 6:
    x_total = dataz['configs']
    y_total = dataz['compatible']
else:
    x_total = dataz['x']
    y_total = dataz['y']

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

"make 5050 distribution rest set"
x_rest, rest_y = unison_shuffled_copies(rest_set, rest_y)
ind_U = np.argwhere(rest_y == 0)
ind_O = np.argwhere(rest_y == 1)
ind_U_short = ind_U[:int(ind_O.shape[0]), 0]
ind_O_short = ind_O[:int(ind_U.shape[0]), 0]
indices = np.append(ind_U_short, ind_O_short)
x_rest, y_rest = x_rest[indices], rest_y[indices]

"add channel axis"
x_rest = x_rest[..., np.newaxis]
rest_set = rest_set[..., np.newaxis]
x_test = test_set[..., np.newaxis]

"save the data"
np.savez('data_smiley_cube_train_trainraw_test_{:d}x{:d}.npz'.format(k, k),
         x_rest=x_rest, y_rest=y_rest, x_rest_raw=rest_set, y_rest_raw=rest_y, x_test=x_test, y_test=test_y)