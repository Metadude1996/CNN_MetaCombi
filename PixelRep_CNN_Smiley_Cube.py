import os
# Get number of threads from Slurm
numThreads = os.cpu_count()

# Set number of threads for inter-operator parallelism,
# start with a single thread
numInterOpThreads = int(numThreads/4)
# numInterOpThreads = int(1)
# The total number of threads must be an integer multiple
# of numInterOpThreads to make sure that all cores are used
assert numThreads % numInterOpThreads == 0

# Compute the number of intra-operator threads; the number
# of OpenMP threads for low-level libraries must be set to
# the same value for optimal performance
# numIntraOpThreads = numThreads // numInterOpThreads
numIntraOpThreads = (numThreads - 4*numInterOpThreads) // 4
os.environ['OMP_NUM_THREADS'] = str(numIntraOpThreads)
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)

os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'

# Import TensorFlow after setting OMP_NUM_THREADS to make sure
# that low-level libraries are initialised correctly
import tensorflow as tf
import numpy as np
import sys
from sklearn.model_selection import StratifiedKFold
# tf.config.optimizer.set_jit(True) # Enable XLA
# Configure TensorFlow
# config = tf.ConfigProto()
# tf.debugging.set_log_device_placement(True)
tf.config.threading.set_inter_op_parallelism_threads(
    numInterOpThreads
)
tf.config.threading.set_intra_op_parallelism_threads(
    numIntraOpThreads
)
np.random.seed(0)

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Model, load_model


# WorkingDirectory = sys.argv[1]
# InputDataDirectory = sys.argv[2]
# k = int(sys.argv[3])
# nhminarg = int(sys.argv[4])
# nhmaxarg = int(sys.argv[5])
# gpu_number = int(sys.argv[6])
kx = 5
ky = 5
WorkingDirectory = '.\\cnn_Smiley_Cube_kx_ky'
InputDataDirectory = '.'
gpu_number = 0
if not os.path.exists(WorkingDirectory):
    os.makedirs(WorkingDirectory)

gpu_list = ["/gpu:0", "/gpu:1", "/gpu:2", "/gpu:3"]
# tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Create 2 virtual GPUs with 3GB memory each
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3),
#          tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024*3)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPU,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)
strategy = tf.distribute.OneDeviceStrategy(device=gpu_list[gpu_number])
# strategy = tf.distribute.MirroredStrategy()


# with tf.device('/device:GPU:0'):
EARLY_STOP = 20


# k=3
filter_size = (2, 2)
stride = (1, 1)
# tf.config.threading.set_intra_op_parallelism_threads(8)
# tf.config.threading.set_inter_op_parallelism_threads(8)
# with tf.device('/CPU:0'):
#learning_rate = 0.005
n_classes=2
#n_filters =10
#n_hidden = 30
# EPOCHS = 15
EPOCHS = 100
save_cm = True
chk_restore = False
AB = True
fiftyfifty= True
uniq = False
periodic_padding = True
run_name = r'Smiley_Cube_{:d}x{:d}'.format(kx, ky)
#set directories
masterdir = os.path.join(WorkingDirectory, r'cnn_'+run_name)
if not os.path.exists(masterdir):
    os.makedirs(masterdir)

figdir = os.path.join(masterdir, r'figs', r'')
logdir = os.path.join(masterdir, r'logs', r'')
checkpoint_path = os.path.join(masterdir, r'ckpts', '')
save_path = os.path.join(masterdir, r'saves', r'')
save_dir = os.path.dirname(save_path)
# make sure input data is in the working directory
datapath = WorkingDirectory
respath = WorkingDirectory

if not os.path.exists(figdir):
    os.makedirs(figdir)
if not os.path.exists(logdir):
    os.makedirs(logdir)
if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
if not os.path.exists(save_path):
    os.makedirs(save_path)
#set fractions of dataset to be used for validation, testing and training

test_frac = 0.15
train_frac = 1-test_frac


#Set the seeds for reproducable results
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.backend.set_floatx('float32')

"Helper Functions"

@tf.function
def reset_weights(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model): #if you're using a model as a layer
            reset_weights(layer) #apply function recursively
            continue

        #where are the initializers?
        if hasattr(layer, 'cell'):
            init_container = layer.cell
        else:
            init_container = layer

        for key, initializer in init_container.__dict__.items():
            if "initializer" not in key: #is this item an initializer?
                  continue #if no, skip it

            # find the corresponding variable, like the kernel or the bias
            if key == 'recurrent_initializer': #special case check
                var = getattr(init_container, 'recurrent_kernel')
            else:
                var = getattr(init_container, key.replace("_initializer", ""))

            var.assign(initializer(var.shape, var.dtype))
            #use the initializer

@tf.function
def f1_score(precision, recall):
    return 2.*(precision*recall)/(precision+recall)

@tf.function
def calc_accuracy(y_true, y_pred):
    return tf.math.divide_no_nan(tf.cast(tf.math.count_nonzero(tf.math.equal(y_true, y_pred)), tf.float32), tf.cast(tf.shape(y_true)[0], tf.float32))



# aim_batchsize = 512
# BATCH_SIZE_PER_REPLICA = 128
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync

"old training data"
# rng = np.random.default_rng(0)
# npz = np.load(os.path.join(InputDataDirectory, r'data_smiley_cube_train_trainraw_test_{:d}x{:d}.npz'.format(k, k)))
# x_rest = npz['x_rest'].astype(np.float32)
# y_rest = npz['y_rest'].astype(np.int32)
# x_restraw = npz['x_rest_raw'].astype(np.float32)
# y_restraw = npz['y_rest_raw']
# int_0 = np.argwhere(y_restraw == 0)
# samples = rng.choice(int_0[:, 0], 1000, replace=False)
# x_rest = np.append(x_rest, x_restraw[samples], axis=0)
# y_rest = np.append(y_rest, y_restraw[samples], axis=0)
#
# ints = np.arange(np.shape(x_rest)[0])
# np.random.shuffle(ints)
# x_rest = x_rest[ints]
# y_rest = y_rest[ints]
# del npz

"new training data"
data = np.load("..\\MetaCombi\\smiley cube\\smiley_cube_x_y_{:d}x{:d}.npz".format(kx, ky))
x_all = data['configs']
y_all = data['compatible']
inds = np.arange(np.shape(y_all)[0])
np.random.shuffle(inds)
x_all = x_all[inds]
y_all = y_all[inds]
ind_0 = np.argwhere(y_all == 0)
ind_1 = np.argwhere(y_all == 1)
ind_0t = ind_0[:int(test_frac*np.shape(ind_0)[0]), 0]
ind_1t = ind_1[:int(test_frac*np.shape(ind_1)[0]), 0]
ind_test = np.append(ind_0t, ind_1t)
np.random.shuffle(ind_test)
x_test = x_all[ind_test][:, :, :, np.newaxis]
y_test = y_all[ind_test]
x_rest_raw = np.delete(x_all, ind_test, axis=0)[:, :, :, np.newaxis]
y_rest_raw = np.delete(y_all, ind_test, axis=0)
"do additional stuff to the training set"
"max dataimbalance: 2/BATCH_SIZE"
ind_0r = np.argwhere(y_rest_raw==0)[:, 0]
ind_1r = np.argwhere(y_rest_raw==1)[:, 0]
ind_0r = ind_0r[:int(BATCH_SIZE_PER_REPLICA * np.shape(ind_1r)[0] / 2.)]
inds_rest = np.append(ind_0r, ind_1r)
np.random.shuffle(inds_rest)
x_rest = x_rest_raw[inds_rest]
y_rest = y_rest_raw[inds_rest]
np.savez('.\\data_smiley_cube_train_trainraw_test_{:d}x{:d}.npz'.format(kx, ky), x_rest=x_rest, y_rest=y_rest,
         x_test=x_test, y_test=y_test, x_rest_raw=x_rest_raw, y_rest_raw=y_rest_raw)

pos_weight = np.sum(y_rest == 0) / np.sum(y_rest == 1)
print('weight:')
print(pos_weight)
del data
# y_rest = tf.one_hot(y_rest, 2)

BUFFER_SIZE = len(x_rest)

# HP_NUM_HIDDEN = hp.HParam('num_hidden_neurons', hp.Discrete([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
# HP_NUM_FILTERS = hp.HParam('num_conv_filters', hp.Discrete([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
# HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005]))

"Grid search HP"
# HP_NUM_HIDDEN_all = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# HP_NUM_HIDDEN = HP_NUM_HIDDEN_all[nhminarg:nhmaxarg]
# HP_NUM_FILTERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
# HP_LEARNING_RATE = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005]
# HP_NUM_HIDDEN = [100]
# HP_NUM_FILTERS = [20]
# HP_LEARNING_RATE = [0.002]
n_kfolds = 10

class CNN(Model):
    def __init__(self, n_f, n_h):
        super(CNN, self).__init__()
        self.conv1 = Conv2D(n_f, filter_size, activation='relu', padding='valid', strides=stride)
        self.flatten = Flatten()
        self.d1 = Dense(n_h, activation='relu')
        self.d2 = Dense(n_classes)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)

run_number = 0
"cnniter"
# for n_filters in HP_NUM_FILTERS:
#     for n_hidden in HP_NUM_HIDDEN:
#         with strategy.scope():
#             if run_number > 0:
#                 tf.keras.backend.clear_session()
#             model = CNN(n_filters, n_hidden)
#         tf.print('cleared session and loaded new model architecture')
#
#         for learning_rate in HP_LEARNING_RATE:
#             # start = time.time()
#             skf = StratifiedKFold(n_splits=n_kfolds)
#             kfold = 0
#             val_acc_avg = 0
#             val_loss_avg = 0
#             val_acc_list = np.zeros(n_kfolds)
#             val_loss_list = np.zeros(n_kfolds)
#             val_prec_list = np.zeros(n_kfolds)
#             val_rec_list = np.zeros(n_kfolds)
#             val_f1_list = np.zeros(n_kfolds)
#             for train_index, val_index in skf.split(x_rest, y_rest):
#
#                 train_ds = tf.data.Dataset.from_tensor_slices((x_rest[train_index], y_rest[train_index])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
#
#                 val_ds = tf.data.Dataset.from_tensor_slices((x_rest[val_index], y_rest[val_index])).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)
#
#                 current_run = os.path.join(r'nf={:d}_nh={:d}_lr={:.4f}'.format(n_filters, n_hidden, learning_rate),
#                                            r'kfold={:d}'.format(kfold))
#
#                 with strategy.scope():
#                     # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO, name='loss')
#
#
#                     def weighted_loss(weight):
#                         def weighted_cross_entropy_with_logits(labels, logits):
#                             loss = tf.nn.weighted_cross_entropy_with_logits(
#                                 tf.one_hot(tf.cast(labels, tf.int32), n_classes, axis=1)[:, :, 0], logits, weight
#                             )
#                             return loss
#
#                         return weighted_cross_entropy_with_logits
#                     # loss_object = tf.keras.losses.weighted_cross_entropy_with_logits
#                     # loss_object_nored = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
#                     #                                                             reduction=tf.keras.losses.Reduction.NONE,
#                     #                                                             name='loss_nored')
#
#                 # define metrics
#                 with strategy.scope():
#                     accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
#                     # accuracy_nored = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_nored')
#                 true_pos = tf.keras.metrics.TruePositives(name='True_pos')
#                 true_neg = tf.keras.metrics.TrueNegatives(name='True_neg')
#                 false_pos = tf.keras.metrics.FalsePositives(name='False_pos')
#                 false_neg = tf.keras.metrics.FalseNegatives(name='False_neg')
#                 Recall = tf.keras.metrics.Recall(name='Recall')
#                 Precision = tf.keras.metrics.Precision(name='Precision')
#                 new_metrics = [Recall, Precision, true_pos, false_pos, true_neg, false_neg]
#
#                 # tf.print('weight reset')
#                 with strategy.scope():
#                     optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
#
#                 train_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'train', r'')
#                 val_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'val', r'')
#                 cm_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'cm', r'')
#                 tb_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'')
#
#                 if not os.path.exists(os.path.join(save_path, current_run)):
#                     os.makedirs(os.path.join(save_path, current_run))
#                 np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index, val_index=val_index)
#
#                 callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP, restore_best_weights=True)
#                 with strategy.scope():
#                     model.compile(optimizer=optimizer, loss=weighted_loss(pos_weight),
#                                   metrics=[accuracy])
#                 # tf.profiler.experimental.ProfilerOptions(
#                 #     host_tracer_level=2, python_tracer_level=1, device_tracer_level=1
#                 # )
#                 # tf.profiler.experimental.start(tb_log_dir)
#                 # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir,
#                 #                                             profile_batch='10, 15')
#                 history = model.fit(train_ds, validation_data=val_ds, validation_steps=None, epochs=EPOCHS, callbacks=[callback], verbose=1)
#
#                 resultdir = os.path.join(logdir, current_run, '')
#                 if not os.path.exists(resultdir):
#                     os.makedirs(resultdir)
#                 np.save(os.path.join(resultdir, r'training_epoch_tloss_tacc_vloss_vacc.npy'),
#                          history.history)
#                 # read with history=np.load('training_epoch_tloss_tacc_vloss_vacc.npy',allow_pickle='TRUE').item()
#                 # with strategy.scope():
#                 #     model.compile(optimizer=model.optimizer,
#                 #                   loss=model.loss,
#                 #                   metrics=model.metrics + new_metrics)
#                 true_pos.reset_states()
#                 true_neg.reset_states()
#                 false_neg.reset_states()
#                 false_pos.reset_states()
#                 # accuracy_nored.reset_states()
#                 Recall.reset_states()
#                 Precision.reset_states()
#
#                 y_val = y_rest[val_index]
#                 # val_loss, val_acc, val_rec, val_prec, val_tp, val_fp, val_tn, val_fn = model.evaluate(val_ds)
#                 y_valpred = model.predict(val_ds)
#                 y_valpredam = tf.argmax(y_valpred, axis=1)
#                 #
#                 # val_loss = loss_object(y_val, y_valpred)
#                 val_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(
#                                 tf.one_hot(tf.cast(y_val, tf.int32), n_classes, axis=1), y_valpred, pos_weight
#                             ), axis=1))
#                 val_acc = calc_accuracy(tf.cast(y_val, tf.int64), y_valpredam)
#                 Precision.update_state(y_val, y_valpredam)
#                 Recall.update_state(y_val, y_valpredam)
#                 true_pos.update_state(y_val, y_valpredam)
#                 true_neg.update_state(y_val, y_valpredam)
#                 false_pos.update_state(y_val, y_valpredam)
#                 false_neg.update_state(y_val, y_valpredam)
#
#                 val_prec = Precision.result()
#                 val_rec = Recall.result()
#                 val_f1 = f1_score(val_prec, val_rec)
#
#                 # Export the model to a SavedModel
#                 savesave = os.path.join(save_path, current_run, r'1', r'')
#                 if not os.path.exists(savesave):
#                     os.makedirs(savesave)
#                 model.save(savesave, save_format='tf')
#                 # tf.saved_model.save(model, savesave)
#
#                 f = open(resultdir+r'results.txt', 'w')
#                 f.write('{} \t {} \t {} \t {} \t {}'.format(val_loss, val_acc, val_prec, val_rec, val_f1))
#                 f.close()
#                 val_acc_list[kfold] = val_acc
#                 val_loss_list[kfold] = val_loss.numpy()
#                 val_prec_list[kfold] = val_prec.numpy()
#                 val_rec_list[kfold] = val_rec.numpy()
#                 val_f1_list[kfold] = val_f1.numpy()
#
#                 ftxt = open(os.path.join(resultdir, r'valset_TP_FP_TN_FN.txt'), 'ab')
#                 np.savetxt(ftxt, np.array([[n_filters, n_hidden, learning_rate, kfold, true_pos.result(), false_pos.result(), true_neg.result(), false_neg.result()]]), delimiter=',')
#                 # np.savetxt(ftxt, np.array([[n_filters, n_hidden, learning_rate, kfold, val_tp,
#                 #                             val_fp, val_tn, val_fn]]), delimiter=',')
#                 ftxt.close()
#
#                 kfold += 1
#                 tf.print('reset weights')
#                 with strategy.scope():
#                     reset_weights(model)
#
#             # f = open(masterdir + '\\nf10_nh_10_10kfold_val_acc_loss.txt')
#             # f.write('{} \t {} \t {} \t {}'.format())
#             np.savez(os.path.join(masterdir, r'nf{:d}_nh{:d}_lr{:.4f}_kfold_val_acc_loss_prec_rec_f1.npz'.format(n_filters, n_hidden, learning_rate)),
#                      val_acc_list=val_acc_list, val_loss_list=val_loss_list, val_prec_list=val_prec_list,
#                      val_rec_list=val_rec_list, val_f1_list=val_f1_list)
#             val_acc_avg = np.mean(val_acc_list)
#             val_acc_var = np.var(val_acc_list)
#             val_loss_avg = np.mean(val_loss_list)
#             val_loss_var = np.var(val_loss_list)
#             val_prec_avg = np.mean(val_prec_list)
#             val_prec_var = np.var(val_prec_list)
#             val_rec_avg = np.mean(val_rec_list)
#             val_rec_var = np.var(val_rec_list)
#             val_f1_avg = np.mean(val_f1_list)
#             val_f1_var = np.var(val_f1_list)
#             # val_acc_avg = val_acc_avg/n_kfolds
#             # val_loss_avg = val_loss_avg/n_kfolds
#             f = open(os.path.join(masterdir, r'kfold_avg_val_results.txt'), 'a')
#             f.write('{:d} \t {:d} \t {:.5f} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n'.format(
#                 n_filters, n_hidden, learning_rate,
#                                                                                      val_acc_avg, val_acc_var,
#                                                                                      val_loss_avg, val_loss_var,
#                                                                                      val_prec_avg, val_prec_var,
#                                                                                      val_rec_avg, val_rec_var,
#                                                                                      val_f1_avg, val_f1_var))
#             f.close()
#
#             run_number += 1
#                     # end = time.time()
#         # tf.print('time single run:')
#         # tf.print(end-start)
#         f.close()  # you can omit in most cases as the destructor will call it

"one-shot cnn"
n_filters = 20
n_hidden = 100
learning_rate = 0.005
with strategy.scope():
    if run_number > 0:
        tf.keras.backend.clear_session()
    model = CNN(n_filters, n_hidden)
tf.print('cleared session and loaded new model architecture')


skf = StratifiedKFold(n_splits=n_kfolds)
kfold = 0
val_acc_avg = 0
val_loss_avg = 0
val_acc_list = np.zeros(n_kfolds)
val_loss_list = np.zeros(n_kfolds)
val_prec_list = np.zeros(n_kfolds)
val_rec_list = np.zeros(n_kfolds)
val_f1_list = np.zeros(n_kfolds)

for train_index, val_index in skf.split(x_rest, y_rest):

    train_ds = tf.data.Dataset.from_tensor_slices((x_rest[train_index].astype(np.float32),
                                                   y_rest[train_index].astype(np.float32))).shuffle(BUFFER_SIZE)\
        .batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

    val_ds = tf.data.Dataset.from_tensor_slices((x_rest[val_index].astype(np.float32),
                                                 y_rest[val_index].astype(np.float32))).batch(GLOBAL_BATCH_SIZE)\
        .prefetch(tf.data.experimental.AUTOTUNE)

    current_run = os.path.join(r'nf={:d}_nh={:d}_lr={:.4f}'.format(n_filters, n_hidden, learning_rate),
                               r'kfold={:d}'.format(kfold))

    with strategy.scope():
        # loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.AUTO, name='loss')


        def weighted_loss(weight):
            def weighted_cross_entropy_with_logits(labels, logits):
                loss = tf.nn.weighted_cross_entropy_with_logits(
                    tf.one_hot(tf.cast(labels, tf.int32), n_classes, axis=1)[:, :, 0], logits, weight
                )
                return loss

            return weighted_cross_entropy_with_logits
        # loss_object = tf.keras.losses.weighted_cross_entropy_with_logits
        # loss_object_nored = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True,
        #                                                             reduction=tf.keras.losses.Reduction.NONE,
        #                                                             name='loss_nored')

    # define metrics
    with strategy.scope():
        accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy')
        # accuracy_nored = tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy_nored')
    true_pos = tf.keras.metrics.TruePositives(name='True_pos')
    true_neg = tf.keras.metrics.TrueNegatives(name='True_neg')
    false_pos = tf.keras.metrics.FalsePositives(name='False_pos')
    false_neg = tf.keras.metrics.FalseNegatives(name='False_neg')
    Recall = tf.keras.metrics.Recall(name='Recall')
    Precision = tf.keras.metrics.Precision(name='Precision')
    new_metrics = [Recall, Precision, true_pos, false_pos, true_neg, false_neg]

    # tf.print('weight reset')
    with strategy.scope():
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    train_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'train', r'')
    val_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'val', r'')
    cm_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'cm', r'')
    tb_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'')

    if not os.path.exists(os.path.join(save_path, current_run)):
        os.makedirs(os.path.join(save_path, current_run))
    np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index, val_index=val_index)

    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=EARLY_STOP, restore_best_weights=True)
    with strategy.scope():
        model.compile(optimizer=optimizer, loss=weighted_loss(pos_weight),
                      metrics=[accuracy])
    # tf.profiler.experimental.ProfilerOptions(
    #     host_tracer_level=2, python_tracer_level=1, device_tracer_level=1
    # )
    # tf.profiler.experimental.start(tb_log_dir)
    # tb_callback = tf.keras.callbacks.TensorBoard(log_dir=tb_log_dir,
    #                                             profile_batch='10, 15')
    history = model.fit(train_ds, validation_data=val_ds, validation_steps=None, epochs=EPOCHS, callbacks=[callback], verbose=1)

    resultdir = os.path.join(logdir, current_run, '')
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)
    np.save(os.path.join(resultdir, r'training_epoch_tloss_tacc_vloss_vacc.npy'),
             history.history)
    # read with history=np.load('training_epoch_tloss_tacc_vloss_vacc.npy',allow_pickle='TRUE').item()
    # with strategy.scope():
    #     model.compile(optimizer=model.optimizer,
    #                   loss=model.loss,
    #                   metrics=model.metrics + new_metrics)
    true_pos.reset_states()
    true_neg.reset_states()
    false_neg.reset_states()
    false_pos.reset_states()
    # accuracy_nored.reset_states()
    Recall.reset_states()
    Precision.reset_states()

    y_val = y_rest[val_index]
    # val_loss, val_acc, val_rec, val_prec, val_tp, val_fp, val_tn, val_fn = model.evaluate(val_ds)
    y_valpred = model.predict(val_ds)
    y_valpredam = tf.argmax(y_valpred, axis=1)
    #
    # val_loss = loss_object(y_val, y_valpred)
    val_loss = tf.math.reduce_mean(tf.math.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(
                    tf.one_hot(tf.cast(y_val, tf.int32), n_classes, axis=1), y_valpred, pos_weight
                ), axis=1))
    val_acc = calc_accuracy(tf.cast(y_val, tf.int64), y_valpredam)
    Precision.update_state(y_val, y_valpredam)
    Recall.update_state(y_val, y_valpredam)
    true_pos.update_state(y_val, y_valpredam)
    true_neg.update_state(y_val, y_valpredam)
    false_pos.update_state(y_val, y_valpredam)
    false_neg.update_state(y_val, y_valpredam)

    val_prec = Precision.result()
    val_rec = Recall.result()
    val_f1 = f1_score(val_prec, val_rec)

    # Export the model to a SavedModel
    savesave = os.path.join(save_path, current_run, r'1', r'')
    if not os.path.exists(savesave):
        os.makedirs(savesave)
    model.save(savesave, save_format='tf')
    # tf.saved_model.save(model, savesave)

    f = open(resultdir+r'results.txt', 'w')
    f.write('{} \t {} \t {} \t {} \t {}'.format(val_loss, val_acc, val_prec, val_rec, val_f1))
    f.close()
    val_acc_list[kfold] = val_acc
    val_loss_list[kfold] = val_loss.numpy()
    val_prec_list[kfold] = val_prec.numpy()
    val_rec_list[kfold] = val_rec.numpy()
    val_f1_list[kfold] = val_f1.numpy()

    ftxt = open(os.path.join(resultdir, r'valset_TP_FP_TN_FN.txt'), 'ab')
    np.savetxt(ftxt, np.array([[n_filters, n_hidden, learning_rate, kfold, true_pos.result(), false_pos.result(), true_neg.result(), false_neg.result()]]), delimiter=',')
    # np.savetxt(ftxt, np.array([[n_filters, n_hidden, learning_rate, kfold, val_tp,
    #                             val_fp, val_tn, val_fn]]), delimiter=',')
    ftxt.close()

    kfold += 1
    tf.print('reset weights')
    with strategy.scope():
        reset_weights(model)

# f = open(masterdir + '\\nf10_nh_10_10kfold_val_acc_loss.txt')
# f.write('{} \t {} \t {} \t {}'.format())
np.savez(os.path.join(masterdir, r'nf{:d}_nh{:d}_lr{:.4f}_kfold_val_acc_loss_prec_rec_f1.npz'.format(n_filters, n_hidden, learning_rate)),
         val_acc_list=val_acc_list, val_loss_list=val_loss_list, val_prec_list=val_prec_list,
         val_rec_list=val_rec_list, val_f1_list=val_f1_list)
val_acc_avg = np.mean(val_acc_list)
val_acc_var = np.var(val_acc_list)
val_loss_avg = np.mean(val_loss_list)
val_loss_var = np.var(val_loss_list)
val_prec_avg = np.mean(val_prec_list)
val_prec_var = np.var(val_prec_list)
val_rec_avg = np.mean(val_rec_list)
val_rec_var = np.var(val_rec_list)
val_f1_avg = np.mean(val_f1_list)
val_f1_var = np.var(val_f1_list)
# val_acc_avg = val_acc_avg/n_kfolds
# val_loss_avg = val_loss_avg/n_kfolds
f = open(os.path.join(masterdir, r'kfold_avg_val_results.txt'), 'a')
f.write('{:d} \t {:d} \t {:.5f} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \t {} \n'.format(
    n_filters, n_hidden, learning_rate,
                                                                         val_acc_avg, val_acc_var,
                                                                         val_loss_avg, val_loss_var,
                                                                         val_prec_avg, val_prec_var,
                                                                         val_rec_avg, val_rec_var,
                                                                         val_f1_avg, val_f1_var))
f.close()

run_number += 1
        # end = time.time()
# tf.print('time single run:')
# tf.print(end-start)
f.close()  # you can omit in most cases as the destructor will call it
