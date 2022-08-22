import tensorflow as tf
import numpy as np
import time

"over range of k, see how long nn takes to classify"

"size of nn"
nf = 20
nh = 100
lr = 0.0030
kfold = 0

models = []
configs = []
for k in range(3, 9):
    if k < 5:
        filepath = u"D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050_2\\".format(k, k) + \
                   u"saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
    else:
        filepath = u"D:\\data\\cnniter_HP_GS_SKF_{:d}x{:d}_AB_5050\\".format(k, k) + \
                   u"saves\\nf={:d}_nh={:d}_lr={:.4f}\\kfold={:d}\\1\\".format(nf, nh, lr, kfold)
        # dataprek = np.load(u'.\\cnniter_HP_GS_SKF_5x5_AB_5050\\data_prek_xy_train_trainraw_test_5x5.npz')
    model = tf.keras.models.load_model(
        filepath, custom_objects=None, compile=False
    )
    models.append(model)
    test_sample = np.random.randint(0, 2, (1000, 2*k+2, 2*k+2)).astype(float)
    configs.append(test_sample[:, :, :, np.newaxis])

times= []
for i in range(len(configs)):
    start_time = time.time()
    models[i].predict(configs[i])
    times.append(time.time() - start_time)

print(times)



