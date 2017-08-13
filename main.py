import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
import glob

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # NOTE: JUST LOAD PRETRAINED VGG16 MODEL -> The encoder for FCN-8 is the VGG16 model pretrained on ImageNet for classification

    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    # Load model
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    # Get graph
    vgg_graph = tf.get_default_graph()
    # Tensors from VGG
    image_input = vgg_graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = vgg_graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = vgg_graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = vgg_graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = vgg_graph.get_tensor_by_name(vgg_layer7_out_tensor_name)

    return image_input, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)


def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer7_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer3_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # NOTE: Create the layers for a FCN (DECODER)(1x1, TRANSPOSED AND SKIP CONNECTIONS) -> LESSON 2
    # To build the decoder portion of FCN-8, we’ll upsample the input to the original image size. The shape of the tensor after the final convolutional transpose layer will be 4-dimensional.

    # Transpose Pool VGG Layer 7
    pool_7 = tf.layers.conv2d(vgg_layer7_out, num_classes, kernel_size=(1,1), strides=(1,1))
    transpose_pool_7 = tf.layers.conv2d_transpose(pool_7, num_classes, kernel_size=4, strides=(2,2), padding='same')

    # Pool VGG Layer 4
    pool_4 = tf.layers.conv2d(vgg_layer4_out, num_classes, kernel_size=(1,1), strides=(1,1))

    # Skip connections
    skip_1 = tf.add(transpose_pool_7, pool_4)

    # Transpose Skip
    transpose_skip_1 = tf.layers.conv2d_transpose(skip_1, num_classes, kernel_size=(4,4), strides=(2,2), padding='same')

    # Pool VGG Layer 3
    pool_3 = tf.layers.conv2d(vgg_layer3_out, num_classes, kernel_size=(1,1), strides=(1,1))

    # Skip
    skip_2 = tf.add(transpose_skip_1, pool_3)

    # Transpose
    transpose_skip_2 = tf.layers.conv2d_transpose(skip_2, num_classes, kernel_size=(16,16), strides=(8,8), padding='same')
    nn_last_layer = transpose_skip_2

    return nn_last_layer
tests.test_layers(layers)


def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # NOTE: Optimize for training
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    labels = tf.reshape(correct_label, (-1, num_classes))
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)


def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # Implement function (just a standard training phase using ADAM OPTIMIZER)

    # hyperparams
    l_rate = 0.0005
    dropout = 0.2
    # display options
    display_training_step = 50

    for epoch in range(epochs):
        training_step = 0
        for images, labels in get_batches_fn(batch_size):

            # RUN train operation
            sess.run(train_op, feed_dict={input_image: images, correct_label: labels, keep_prob: 1-dropout, learning_rate: l_rate})
            # The loss of the network should be printed while the network is training.
            if training_step % display_training_step == 0:
                acc_loss = sess.run(cross_entropy_loss, feed_dict={input_image: images, correct_label: labels, keep_prob: 1.0})
                print("Epoch: " + str(epoch)+ " Step: " + str(training_step) + " Loss: {:.6f}".format(acc_loss))
            training_step += 1
tests.test_train_nn(train_nn)


def run():
    num_classes = 2 # Road, not road
    image_shape = (160, 576)
    data_dir = './data'
    runs_dir = './runs'

    # hyperparams
    epochs = 20
    batch_size = 8

    tests.test_for_kitti_dataset(data_dir)
    # LOGS to DEBUG LEVEL
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    correct_label = tf.placeholder(tf.float32, [None, image_shape[0], image_shape[1], num_classes])
    learning_rate = tf.placeholder(tf.float32)
    keep_prob = tf.placeholder(tf.float32)

    # Create a TensorFlow configuration object. This will be
    # passed as an argument to the session.
    # config = tf.Config()
    # JIT level, this can be set to ON_1 or ON_2
    # jit_level = tf.OptimizerOptions.ON_1
    # config.graph_options.optimizer_options.global_jit_level = jit_level

    with tf.Session() as sess:

        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg')

        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)
        num_images = len(glob.glob(os.path.join(data_dir, 'data_road/training/calib/*.*')))

        print("Tranining set: " + str(num_images))
        print("Batch size: " + str(batch_size))

        # Build NN using load_vgg, layers, and optimize function
        # load vgg
        image_input, keep_prob, layer3_out, layer4_out, layer7_out = load_vgg(sess, vgg_path)
        # decoder
        nn_last_layer = layers(layer3_out, layer4_out, layer7_out, num_classes)
        # optimize
        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # Initialize and Train NN using the train_nn function
        sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, image_input,
             correct_label, keep_prob, learning_rate)

        # Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, image_input)

if __name__ == '__main__':
    run()
