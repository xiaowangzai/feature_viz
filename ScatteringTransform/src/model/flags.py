
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def define_flags():

    ############
    # Run mode
    ############
    tf.app.flags.DEFINE_string('run', None, "Which operation to run. [train|inference]")

    ##########################
    # Training parameters
    ###########################
    tf.app.flags.DEFINE_integer('nb_epoch', 10, "Number of epochs")
    tf.app.flags.DEFINE_integer('batch_size', 128, "Number of samples per batch.")
    tf.app.flags.DEFINE_integer('nb_batch_per_epoch', 30, "Number of batches per epoch")
    tf.app.flags.DEFINE_float('learning_rate', 1E-5, "Learning rate used for AdamOptimizer")
    tf.app.flags.DEFINE_integer('noise_dim', 100, "Noise dimension for GAN generation")
    tf.app.flags.DEFINE_integer('random_seed', 0, "Seed used to initialize rng.")

    ############################################
    # General tensorflow parameters parameters
    #############################################
    tf.app.flags.DEFINE_bool('use_XLA', False, "Whether to use XLA compiler.")
    tf.app.flags.DEFINE_integer('num_threads', 2, "Number of threads to fetch the data")
    tf.app.flags.DEFINE_float('capacity_factor', 32, "Nuumber of batches to store in queue")

    ##########
    # Datasets
    ##########
    tf.app.flags.DEFINE_string('data_format', "NCHW", "Tensorflow image data format.")
    tf.app.flags.DEFINE_integer('channels', 1, "Number of channels")
    tf.app.flags.DEFINE_float('central_fraction', 0.8, "Central crop as a fraction of total image")
    tf.app.flags.DEFINE_integer('img_size', 28, "Image size")

    ##############
    # Directories
    ##############
    tf.app.flags.DEFINE_string('model_dir', '/home/ubuntu/feature_viz/ScatteringTransform/src/models', "Output folder where checkpoints are dumped.")
    tf.app.flags.DEFINE_string('log_dir', '/home/ubuntu/feature_viz/ScatteringTransform/src/logs', "Logs for tensorboard.")
    tf.app.flags.DEFINE_string('fig_dir', '/home/ubuntu/feature_viz/ScatteringTransform/src/figures', "Where to save figures.")
    tf.app.flags.DEFINE_string('raw_dir', '/home/ubuntu/feature_viz/ScatteringTransform/src/data/raw', "Where raw data is saved")
    tf.app.flags.DEFINE_string('data_dir', '/home/ubuntu/feature_viz/ScatteringTransform/src/data/processed', "Where processed data is saved")
