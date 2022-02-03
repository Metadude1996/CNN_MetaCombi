"Ryan van Mastrigt, 31.01.2022"
"train convolutional neural networks to classify 5x5 unit cells into class C or I"


import os
# Get number of threads from Slurm
numThreads = os.cpu_count()

# Set number of threads for inter-operator parallelism,
# start with a single thread
numInterOpThreads = int(numThreads/2)
# numInterOpThreads = 2

# The total number of threads must be an integer multiple
# of numInterOpThreads to make sure that all cores are used
assert numThreads % numInterOpThreads == 0

# Compute the number of intra-operator threads; the number
# of OpenMP threads for low-level libraries must be set to
# the same value for optimal performance
numIntraOpThreads = numThreads // numInterOpThreads
os.environ['OMP_NUM_THREADS'] = str(numIntraOpThreads)
SEED = 0
os.environ['PYTHONHASHSEED'] = str(SEED)

# Import TensorFlow after setting OMP_NUM_THREADS to make sure
# that low-level libraries are initialised correctly
import tensorflow as tf
import numpy as np
import sys
import io
import matplotlib.pyplot as plt
import sklearn.metrics
import itertools
from sklearn.model_selection import StratifiedKFold
import time

# Configure TensorFlow
# config = tf.ConfigProto()
tf.config.threading.set_inter_op_parallelism_threads(
    numInterOpThreads
)
tf.config.threading.set_intra_op_parallelism_threads(
    numIntraOpThreads
)


from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.python.keras.models import Model, load_model

tf.debugging.set_log_device_placement(False)
gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#   # Create 2 virtual GPUs with 1GB memory each
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
strategy = tf.distribute.MirroredStrategy()

"input: work directory, directory of the input data, size of the unit cell, minimum index nh_list, "
"maximum index nh_list"
WorkingDirectory = sys.argv[1]
InputDataDirectory = sys.argv[2]
k = int(sys.argv[3])
nhminarg = int(sys.argv[4])
nhmaxarg = int(sys.argv[5])

# with tf.device('/device:GPU:0'):
EARLY_STOP = 3


# k=3
filter_size = (2, 2)
stride = (2, 2)
# tf.config.threading.set_intra_op_parallelism_threads(8)
# tf.config.threading.set_inter_op_parallelism_threads(8)
# with tf.device('/CPU:0'):
#learning_rate = 0.005
n_classes=2
#n_filters =10
#n_hidden = 30
EPOCHS = 20
save_cm = True
chk_restore = False
AB = True
fiftyfifty= True
uniq = False
periodic_padding = True
run_name = r'HP_GS_SKF_{:d}x{:d}_AB_5050_nh_{:d}to{:d}'.format(k, k, nhminarg, nhmaxarg)
#set directories
masterdir = os.path.join(WorkingDirectory, r'cnniter_'+run_name)
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

#load input data (features and labels)
# raw_data = np.loadtxt(os.path.join(InputDataDirectory, 'PixelRep_{:d}x{:d}.txt'.format(k, k)), delimiter=',', dtype=np.float32)
# raw_results = np.loadtxt(os.path.join(InputDataDirectory, 'results_analysis_new_rrQR_i_Scen_slope_offset_M1k_{:d}x{:d}.txt'.format(k, k)), delimiter=',', dtype=np.float32)

"Helper Functions"
def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]

def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image

def plot_confusion_matrix(cm, class_names, dir):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(10, 10))
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float32') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues, vmin=0, vmax=1)
    plt.title("Confusion matrix")
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names)
    plt.yticks(tick_marks, class_names)


    # Use white text if squares are dark; otherwise black.
    threshold = 0.5
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    #plt.tight_layout()
    plt.colorbar()
    plt.xlim(-0.5, cm.shape[1]-0.5)
    plt.ylim(-0.5, cm.shape[0]-0.5)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if save_cm:
        plt.savefig(dir+'\\cm.pdf', transparent=True)
        plt.savefig(dir+'\\cm.png', transparent=True)
    return figure

def plot_weight_matrix(wm, layer_name, dir):
    """
    Returns a matplotlib figure containing the plotted weight matrix.

    Args:
    wm (array, shape = [n, n]): a weight matrix belong to a layer
    layer_name (string): String name of the layer the weight matrix belongs to
    """
    figure = plt.figure(figsize=(10, 10))
    plt.imshow(wm.T, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Weight matrix "+layer_name+' layer')
    tick_marks_x = np.arange(wm.shape[0])
    tick_marks_y = np.arange(wm.shape[1])
    plt.xticks(tick_marks_x)
    plt.yticks(tick_marks_y)


    # Use white text if squares are dark; otherwise black.
    # wm = np.around(wm, decimals=2)
    # threshold = wm.min() + (wm.min()-wm.max())/2.
    # for i, j in itertools.product(range(wm.shape[0]), range(wm.shape[1])):
    #     color = "white" if wm[i, j] > threshold else "black"
    #     plt.text(i, j, wm[i, j], horizontalalignment="center", color=color)

    #plt.tight_layout()
    plt.colorbar()
    plt.xlim(-0.5, wm.shape[0]-0.5)
    plt.ylim(-0.5, wm.shape[1]-0.5)
    plt.ylabel('i')
    plt.xlabel('j')
    plt.savefig(dir+'\\weight_matrix_'+layer_name+'_layer.pdf')
    plt.savefig(dir+'\\weight_matrix_' + layer_name + '_layer.png')
    plt.show()
    plt.close()
    return figure

def plot_bias_vector(bv, layer_name, dir):
    """
    Returns a matplotlib figure containing the plotted weight matrix.

    Args:
    bv (array, shape = [n]): a weight matrix belong to a layer
    layer_name (string): String name of the layer the bias vector belongs to
    """
    bv = np.reshape(bv, (1, -1))
    figure = plt.figure()
    plt.imshow(bv.T, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Bias vector "+layer_name+' layer')
    tick_marks_x = np.arange(bv.shape[0])
    tick_marks_y = np.arange(bv.shape[1])
    plt.xticks(tick_marks_x)
    plt.yticks(tick_marks_y)


    # Use white text if squares are dark; otherwise black.
    bv = np.around(bv, decimals=2)
    threshold = bv.min() + (bv.max()-bv.min())/2.
    for i, j in itertools.product(range(bv.shape[0]), range(bv.shape[1])):
        color = "white" if bv[i, j] > threshold else "black"
        plt.text(i, j, bv[i, j], horizontalalignment="center", color=color)

    #plt.tight_layout()
    plt.colorbar()
    plt.xlim(-0.5, bv.shape[0]-0.5)
    plt.ylim(-0.5, bv.shape[1]-0.5)
    plt.ylabel('i')
    plt.xlabel('j')
    plt.tight_layout(pad=0.4)
    plt.savefig(dir+'\\bias_vector_'+layer_name+'_layer.pdf')
    plt.savefig(dir+'\\bias_vector_' + layer_name + '_layer.png')
    #plt.show()
    plt.close()
    return figure

def plot_filters(filts, layer_name, dir):
    """
      Returns a matplotlib figure containing the plotted filters.

      Args:
      filter (array, shape = [f_x, f_y, channels, n_filters]): list of filters belonging to a convolution layer
      layer_name (string): String name of the layer the filters belong to
      """
    SMALL_SIZE = 8
    SMALLER_SIZE = 6
    n_plots = filts.shape[3]
    n_plots_x = 4
    n_plots_y = np.ceil(n_plots/n_plots_x).astype(int)
    figure, axs = plt.subplots(n_plots_y, n_plots_x, sharex=True, sharey=True)
    for plot in range(n_plots):
        ax_x = plot % n_plots_x
        ax_y = int(plot/n_plots_x)
        if n_plots_y>1:
            ax = axs[ax_y, ax_x]
        else:
            ax = axs[ax_x]
        filt = filts[:, :, 0, plot]
        im = ax.imshow(filt.T, interpolation='nearest', cmap=plt.cm.Blues)
        # ax.set_title('{:d}'.format(plot))
        ax.set_xticks(np.arange(filt.shape[0]))
        ax.set_yticks(np.arange(filt.shape[1]))
        filt = np.around(filt, decimals=2)
        threshold = filt.min() + (filt.max()-filt.min())/2.
        for i, j in itertools.product(range(filt.shape[0]), range(filt.shape[1])):
            color = "white" if filt[i, j] > threshold else "black"
            ax.text(i, j, filt[i, j], horizontalalignment="center", color=color, fontsize=SMALLER_SIZE)
        #plt.colorbar(im, cax=ax)
        ax.set_xlim(-0.5, filt.shape[0] - 0.5)
        ax.set_ylim(-0.5, filt.shape[1] - 0.5)
        #ax.set_xlabel('i')
        #ax.set_ylabel('j')
    #plt.xticks(np.arange(filt.shape[0]))
    #plt.yticks(np.arange(filt.shape[1]))
    #plt.rc('axes', titlesize=SMALLER_SIZE)    # fontsize of the subplot tiles
    plt.rc('axes', labelsize=SMALLER_SIZE)   # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALLER_SIZE)   # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALLER_SIZE)   # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALLER_SIZE)
    plt.tight_layout()
    plt.savefig(dir + '\\filters_' + layer_name + '_layer.pdf')
    plt.savefig(dir + '\\filters_' + layer_name + '_layer.png')
    #plt.show()
    plt.close()
    return figure

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
def determine_precision(classifier_pred, true_pred):
    "calculate precision=number of correct pos classified/total pos classified"
    n_true_pos_class = tf.cast(classifier_pred, tf.int32) + tf.cast(true_pred, tf.int32) ==2
    n_true_pos_class = tf.math.reduce_sum(n_true_pos_class)
    n_pos_class = tf.math.reduce_sum(classifier_pred)
    return n_true_pos_class/n_pos_class

@tf.function
def determine_recall(classifier_pred, true_pred):
    "calculate recall = number of correct pos classified/ number of true pos"
    n_true_pos_class = tf.cast(classifier_pred, tf.int32) + tf.cast(true_pred, tf.int32) ==2
    n_true_pos_class = tf.math.reduce_sum(n_true_pos_class)
    n_true_pos = tf.math.reduce_sum(true_pred)
    return n_true_pos_class/n_true_pos

@tf.function
def f1_score(precision, recall):
    return 2.*(precision*recall)/(precision+recall)
#
#
#
# # if not one-hot:
# y_total = raw_results[:, 1].astype(int)
# # data should be just 0s and 1s and an 2*k x 2*k image
# x_total = raw_data.reshape(-1, 2*k, 2*k)
# # shuffle the data
# x_total, y_total = unison_shuffled_copies(x_total, y_total)
# x_total_raw, y_total_raw = x_total, y_total
#
# if periodic_padding:
#     # create periodic padding (make sure to set conv2d padding to valid!)
#     x_total = np.tile(x_total, (1, 3, 3))
#     # check how far the filter reaches with given stride and filter size over standard increased image of original
#     # size + filter_size. Add this length to the image if the extra length is not a multiple of k
#     x_over = ((filter_size[0]+np.ceil((2*k)/stride[0])*stride[0]) % (2*k+filter_size[0])) % k
#     y_over = ((filter_size[1]+np.ceil((2*k)/stride[1])*stride[1]) % (2*k+filter_size[1])) % k
#     x_left = np.floor(2*k-0.5*filter_size[0]).astype(int)
#     x_right = np.ceil(4*k+0.5*filter_size[0]+x_over).astype(int)
#     y_down = np.floor(2*k-0.5*filter_size[0]).astype(int)
#     y_up = np.ceil(4*k + 0.5*filter_size[0]+y_over).astype(int)
#     x_total = x_total[:, x_left:x_right, y_down:y_up]
#
# # divide into different sets
# ind_test = int(test_frac * x_total.shape[0])
# # ind_val = ind_train + int(val_frac*x_total.shape[0])
# x_test, y_test = x_total[0:ind_test], y_total[0:ind_test]
# # x_train, y_train = x_total, y_total
# x_rest, y_rest = x_total[ind_test:], y_total[ind_test:]
# # x_test, y_test = x_total[ind_val:-1], y_total[ind_val:-1]
# # create even distribution of scenarios A & B for training data
# if AB:
#     ind_A = np.argwhere(y_rest == 0)
#     ind_B = np.argwhere(y_rest == 1)
#     #ind_C = np.argwhere(y_total == 2)
#     if fiftyfifty:
#         ind_A_short = ind_A[:int(ind_B.shape[0])]
#         ind_B_short = ind_B[:int(ind_A.shape[0])]
#         indices = np.append(ind_A_short, ind_B_short)
#     else:
#         indices = np.append(ind_A, ind_B)
#     indices_raw = np.append(ind_A, ind_B)
#     x_rest_raw, y_rest_raw = x_rest[indices_raw], y_rest[indices_raw]
#     x_rest, y_rest = x_rest[indices], y_rest[indices]
#     x_rest, y_rest = unison_shuffled_copies(x_rest, y_rest)
#     x_rest_raw, y_rest_raw = unison_shuffled_copies(x_rest_raw, y_rest_raw)
#
# # #only unique datapoints:
# if uniq:
#     x_rest = x_rest.round(decimals=9)
#     count=0
#     for i in range(n_classes):
#         ind_lab = np.argwhere(y_rest == i)
#         ind_uniq_0 = np.unique(x_rest[ind_lab, 0], return_index=True, axis=0)[1]
#         for j, row1 in enumerate(ind_uniq_0):
#             ind_1 = np.argwhere(x_rest[ind_lab, 0] == x_rest[ind_lab[row1], 0])
#             ind_uniq_1 = np.unique(x_rest[ind_lab[ind_1], 1], return_index=True, axis=0)[1]
#             if count == 0:
#                 ind_uniq = ind_lab[ind_1[ind_uniq_1]]
#             else:
#                 ind_uniq = np.append(ind_uniq, ind_lab[ind_1[ind_uniq_1]])
#             count += 1
#     x_rest = x_rest[ind_uniq]
#     y_rest = y_rest[ind_uniq]
#     # shuffle the data again
#     x_rest, y_rest = unison_shuffled_copies(x_rest, y_rest)
#
# # add channel dimension
# x_rest = x_rest[..., tf.newaxis]
# x_rest_raw = x_rest_raw[..., tf.newaxis]
# x_test = x_test[..., tf.newaxis]

# np.savez(os.path.join(masterdir, 'data_prek_xy_train_trainraw_test.npz'), x_rest=x_rest, y_rest=y_rest, x_rest_raw=x_rest_raw, y_rest_raw=y_rest_raw, x_test=x_test, y_test=y_test)

npz = np.load(os.path.join(InputDataDirectory, r'data_prek_xy_train_trainraw_test_{:d}x{:d}.npz'.format(k, k)))
x_rest = npz['x_rest']
y_rest = npz['y_rest']
x_rest_raw = npz['x_rest_raw']
y_rest_raw = npz['y_rest_raw']
x_test = npz['x_test']
y_test = npz['y_test']
BUFFER_SIZE = len(x_rest)

# aim_batchsize = 512
BATCH_SIZE_PER_REPLICA = 256
GLOBAL_BATCH_SIZE = BATCH_SIZE_PER_REPLICA * strategy.num_replicas_in_sync



# Names of the integer classes, i.e., 0 -> T-short/top, 1 -> Trouser, etc.
class_names = ['A', 'B']

# HP_NUM_HIDDEN = hp.HParam('num_hidden_neurons', hp.Discrete([2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]))
# HP_NUM_FILTERS = hp.HParam('num_conv_filters', hp.Discrete([2, 4, 6, 8, 10, 12, 14, 16, 18, 20]))
# HP_LEARNING_RATE = hp.HParam('learning_rate', hp.Discrete([0.0001, 0.001, 0.002, 0.003, 0.004, 0.005]))

"Grid search HP"
# HP_NUM_HIDDEN_all = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# HP_NUM_HIDDEN = HP_NUM_HIDDEN_all[nhminarg:nhmaxarg]
HP_NUM_FILTERS = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
HP_LEARNING_RATE = [0.0001, 0.001, 0.002, 0.003, 0.004, 0.005]
HP_NUM_HIDDEN = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# HP_NUM_FILTERS = [18]
# HP_LEARNING_RATE = [0.002]
n_kfolds = 10



def log_confusion_matrix(epoch, logs, dir):
    # Use the model to predict the values from the validation dataset.
    test_pred_raw = model.predict(x_rest_raw)
    test_pred = np.argmax(test_pred_raw, axis=1)

    # Calculate the confusion matrix.
    cm = sklearn.metrics.confusion_matrix(y_rest_raw, test_pred)
    # Log the confusion matrix as an image summary.
    figure = plot_confusion_matrix(cm, class_names=class_names, dir=dir)
    cm_image = plot_to_image(figure)

    # Log the confusion matrix as an image summary.
    with file_writer_cm.as_default():
        tf.summary.image("Confusion Matrix", cm_image, step=epoch)


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


#hp run

run_number = 0
val_loss_old = 100.
stop_count = 0
for n_filters in HP_NUM_FILTERS:
    for n_hidden in HP_NUM_HIDDEN:
        if run_number>0:
            tf.keras.backend.clear_session()
        with strategy.scope():
            # tf.keras.backend.clear_session()
            model = CNN(n_filters, n_hidden)
        tf.print('cleared session and loaded new model architecture')
        for learning_rate in HP_LEARNING_RATE:
            # start = time.time()
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
                val_loss_old = 100.
                training_data = np.zeros((EPOCHS, 5))
                train_ds = tf.data.Dataset.from_tensor_slices((x_rest[train_index], y_rest[train_index])).shuffle(BUFFER_SIZE).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

                val_ds = tf.data.Dataset.from_tensor_slices((x_rest[val_index], y_rest[val_index])).batch(GLOBAL_BATCH_SIZE).prefetch(tf.data.experimental.AUTOTUNE)

                train_dist_dataset = strategy.experimental_distribute_dataset(train_ds)
                val_dist_dataset = strategy.experimental_distribute_dataset(val_ds)

                current_run = os.path.join(r'nf={:d}_nh={:d}_lr={:.4f}'.format(n_filters, n_hidden, learning_rate),
                                           r'kfold={:d}'.format(kfold))


                # print('run {:d}; nf={:d}; nh={:d}; lr={:.4f}; kfold={:d}\n'.format(run_number, n_filters, n_hidden, learning_rate, kfold))
                with strategy.scope():
                    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction=tf.keras.losses.Reduction.NONE)
                    def compute_loss(labels, predictions):
                        per_example_loss = loss_object(labels, predictions)
                        return tf.nn.compute_average_loss(per_example_loss, global_batch_size=GLOBAL_BATCH_SIZE)

                # define metrics
                with strategy.scope():
                    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
                    test_loss = tf.keras.metrics.Mean(name='test_loss')
                    test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
                    val_loss = tf.keras.metrics.Mean(name='val_loss')
                    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')
                    true_pos = tf.keras.metrics.TruePositives(name='True_pos')
                    true_neg = tf.keras.metrics.TrueNegatives(name='True_neg')
                    false_pos = tf.keras.metrics.FalsePositives(name='False_pos')
                    false_neg = tf.keras.metrics.FalseNegatives(name='False_neg')

                # Set up Checkpoints
                checkpoint_dir = os.path.join(checkpoint_path, current_run, '')
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)

                # tf.print('weight reset')
                with strategy.scope():
                    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                    ckpt = tf.train.Checkpoint(step=tf.Variable(0), optimizer=optimizer, model=model)
                    manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=EARLY_STOP)

                def train_step(images, labels):
                    with tf.GradientTape() as tape:
                        # training=True is only needed if there are layers with different
                        # behaviour during training versus inference (e.g. Dropout)
                        predictions = model(images, training=True)
                        loss = compute_loss(labels, predictions)
                    gradients = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

                    # train_loss.update_state(loss)
                    train_accuracy.update_state(labels, predictions)
                    return loss


                def val_step(images, labels):
                    predictions = model(images, training=False)
                    v_loss = compute_loss(labels, predictions)

                    val_loss.update_state(v_loss)
                    val_accuracy.update_state(labels, predictions)


                def val_predict(images, labels):
                    predictions = model(images, training=False)
                    # val_pred_argmax = tf.math.argmax(predictions, axis=1)
                    v_loss = compute_loss(labels, predictions)
                    #
                    val_loss.update_state(v_loss)
                    pred_red = tf.math.argmax(predictions, axis=1, output_type=tf.int32)
                    val_accuracy.update_state(labels, predictions)
                    true_pos.update_state(labels, pred_red)
                    true_neg.update_state(labels, pred_red)
                    false_pos.update_state(labels, pred_red)
                    false_neg.update_state(labels, pred_red)


                @tf.function
                def distributed_train_step(images, labels):
                    per_replica_losses = strategy.run(train_step, args=(images, labels,))
                    return strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

                @tf.function
                def distributed_val_step(images, labels):
                    return strategy.run(val_step, args=(images, labels,))

                @tf.function
                def distributed_val_predict(images, labels):
                    return strategy.run(val_predict, args=(images, labels,))

                # set up log directory
                #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

                train_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'train', r'')
                val_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'val', r'')
                cm_log_dir = os.path.join(logdir, r'gradient_tape', current_run, r'cm', r'')
                with strategy.scope():
                    train_summary_writer = tf.summary.create_file_writer(train_log_dir)
                    val_summary_writer = tf.summary.create_file_writer(val_log_dir)
                    file_writer_cm = tf.summary.create_file_writer(cm_log_dir)
                if not os.path.exists(os.path.join(save_path, current_run)):
                    os.makedirs(os.path.join(save_path, current_run))
                np.savez(os.path.join(save_path, current_run, r'kfold_data_indices.npz'), train_index=train_index, val_index=val_index)

                if chk_restore:
                    ckpt.restore(manager.latest_checkpoint)
                    if manager.latest_checkpoint:
                        print("Restored from {}".format(manager.latest_checkpoint))
                    else:
                        print("Initializing from scratch")

                epoch = tf.cast(ckpt.step, tf.int32)
                for epoch in range(tf.cast(ckpt.step, tf.int32), EPOCHS):
                    # Reset the metrics at the start of the next epoch
                    # train_loss.reset_states()

                    total_loss = 0.0
                    num_batches = 0

                    for images, labels in train_dist_dataset:
                        total_loss += distributed_train_step(images, labels)
                        num_batches += 1
                    train_loss = total_loss / num_batches

                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss, step=epoch)
                        tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)


                    ckpt.step.assign_add(1)
                    save_path_ckpt = manager.save()
                    # print('Saved checkpoint for step {}: {}'.format(int(ckpt.step), save_path_ckpt))
                    for val_images, val_labels in val_dist_dataset:
                        distributed_val_step(val_images, val_labels)
                    with val_summary_writer.as_default():
                        tf.summary.scalar('loss', val_loss.result(), step=epoch)
                        tf.summary.scalar('accuracy', val_accuracy.result(), step=epoch)


                    #early stopping with test loss
                    val_loss_new = val_loss.result()
                    val_loss_min = tf.math.minimum(val_loss_new, val_loss_old)
                    if val_loss_new > val_loss_min:
                        stop_count += 1
                    else:
                        ckpt_best = ckpt.step
                        stop_count = 0
                    val_loss_old = val_loss_min

                    template = 'run {}, kfold {}, Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
                    tf.print(template.format(run_number, kfold,
                                          epoch+1,
                                          train_loss,
                                          train_accuracy.result()*100,
                                          val_loss.result(),
                                          val_accuracy.result()*100))
                    training_data[epoch] = np.array([epoch+1,
                                          train_loss,
                                          train_accuracy.result()*100,
                                          val_loss.result(),
                                          val_accuracy.result()*100])


                    train_accuracy.reset_states()
                    val_loss.reset_states()
                    val_accuracy.reset_states()
                    if stop_count == EARLY_STOP:
                        #early stopping, restore best model
                        ckpt.restore(manager.checkpoints[0])
                        tf.print('restored from {}'.format(manager.checkpoints[0]))
                        epoch = epoch - EARLY_STOP
                        break


                # predictions = model.predict(x_rest[val_index])
                # val_pred_argmax = tf.math.argmax(predictions, axis=1)
                # v_loss = compute_loss(y_rest[val_index], predictions)
                #
                # val_loss.update_state(v_loss)
                # val_accuracy.update_state(y_rest[val_index], predictions)
                true_pos.reset_states()
                true_neg.reset_states()
                false_neg.reset_states()
                false_pos.reset_states()
                val_loss.reset_states()
                val_accuracy.reset_states()

                distributed_val_predict(x_rest[val_index], y_rest[val_index])

                val_prec = true_pos.result() / (tf.math.add(true_pos.result(), false_pos.result()))
                val_rec = true_pos.result() / (tf.math.add(true_pos.result(), false_neg.result()))
                val_f1 = f1_score(val_prec, val_rec)

                # print('Validation Loss: {}, Accuracy: {}, Precision: {}, Recall: {}, F1-score: {} \n'.format(val_loss.result(), val_accuracy.result()*100, val_prec, val_rec, val_f1))
                # figsave = os.path.join(figdir, current_run, r'')
                # if not os.path.exists(figsave):
                #     os.makedirs(figsave)
                # log_confusion_matrix(EPOCHS, cm_log_dir, figsave)
                # weights = model.get_weights()
                # plot_filters(weights[0], 'conv', figsave)
                # plot_bias_vector(weights[1], 'conv', figsave)
                #plot_weight_matrix(weights[2], 'hidden')
                #plot_weight_matrix(weights[4], 'output')
                #plot_bias_vector(weights[3], 'hidden')
                #plot_bias_vector(weights[5], 'output')


                # Export the model to a SavedModel
                savesave = os.path.join(save_path, current_run, r'1', r'')
                if not os.path.exists(savesave):
                    os.makedirs(savesave)
                model.save(savesave, save_format='tf')
                # tf.saved_model.save(model, savesave)


                resultdir = os.path.join(logdir, current_run, '')
                if not os.path.exists(resultdir):
                    os.makedirs(resultdir)
                np.savez(os.path.join(resultdir, r'training_epoch_tloss_tacc_vloss_vacc.npy'),
                         training_data)
                f = open(resultdir+r'results.txt', 'w')
                f.write('{} \t {} \t {} \t {} \t {}'.format(val_loss.result(), val_accuracy.result(), val_prec, val_rec, val_f1))
                f.close()
                val_acc_list[kfold] = val_accuracy.result()
                val_loss_list[kfold] = val_loss.result()
                val_prec_list[kfold] = val_prec
                val_rec_list[kfold] = val_rec
                val_f1_list[kfold] = val_f1

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