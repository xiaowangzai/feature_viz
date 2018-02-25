from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1
import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import numpy as np
from nets import nets_factory

res_scope = nets_factory.arg_scopes_map['resnet_v1_50']
N_CLASSES=10
class MLP:
    def __init__(self, res_checkpoint_dir):
        self.imgs = tf.placeholder(tf.float32, [None, None, None, 3])
        self.feature_extractor()
        self.build()

        self.labels = tf.placeholder(tf.int64, [None])
        y = tf.one_hot(self.labels, depth=N_CLASSES, dtype=tf.int64)

        loss = tf.losses.mean_squared_error(y, self.y_hat)
        opt = tf.train.AdamOptimizer()
        self.grads = opt.compute_gradients(loss)
        self.grads = [(g, v) for (g, v) in self.grads if self.grad_filter(block_k=0, v=v)] # if 0 update everything but logits
        self.update = opt.apply_gradients(self.grads)
        n_equals = tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(self.y_hat, axis=1)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(n_equals)

        # pretrained resnet
        restore_fn = slim.assign_from_checkpoint_fn(res_checkpoint_dir, self.res_variables_to_restore)
        self.saver = tf.train.Saver(tf.trainable_variables())
        self.sess = tf.Session()
        restore_fn(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def grad_filter(self, block_k, v):
        if block_k == 0:
            if 'logits' not in v.name:
                return True
        if 'classifier' in v.name:
            return True
        for k in range(block_k, 5, 1): # max block number is 4 
            if 'block{}'.format(k) in v.name:
                return True
        return False

    def feature_extractor(self):
        network_fn = nets_factory.get_network_fn('resnet_v1_50', 
            num_classes=1000, is_training=False) # 1k b/c of pretrained weight shape
        _, endpoints = network_fn(self.imgs)
        self.features = tf.reshape(endpoints['resnet_v1_50/block4'], [tf.shape(self.imgs)[0], 8192])
        self.res_variables_to_restore = slim.get_variables_to_restore()
         
    def build(self):
        with tf.variable_scope('classifier'):
            l1 = slim.fully_connected(self.features, 200, activation_fn=tf.nn.relu)
            l2 = slim.fully_connected(l1, 200, activation_fn=tf.nn.relu)
            self.y_hat = tf.layers.dense(l2, N_CLASSES, activation=tf.nn.softmax)

    def preprocess_imgs(self, imgs): 
        transformed_imgs = []
        for i in range(len(imgs)):
            new_img = np.reshape(imgs[i], [28, 28, 1])  # [batch, height, width, channels]
            new_img = np.repeat(new_img, 3, axis=2) # copy channel b/c mnist is grayscale
            new_img = cv2.resize(new_img, (33, 33))  # gotta be atleast 33x33 b/c resnet feature extractor is a joke
            new_img = 255 * new_img # mnist was represented between 0-1 but pretrained imgs 0-255
            transformed_imgs.append(new_img)
        transformed_imgs = np.reshape(transformed_imgs, [-1, 33, 33, 3])
        return transformed_imgs

    def test(self, labels, imgs, i):
        imgs = self.preprocess_imgs(imgs)
        y_hat, accuracy = self.sess.run([self.y_hat, self.accuracy], feed_dict={self.imgs: imgs, self.labels:labels})
        print('iter {} test accuracy: {}'.format(i, accuracy))

    def train(self, labels, imgs, i):
        imgs = self.preprocess_imgs(imgs)
        g, _, acc = self.sess.run([self.grads, self.update, self.accuracy], feed_dict={self.imgs: imgs, self.labels: labels}) 
        print('iter {} train accuracy: {}'.format(i, acc))

    def save(self, i):
        self.saver.save(self.sess, 'mnist_ckpt/mnist', global_step=i)

    def load(self):
        ckpt_path = tf.train.latest_checkpoint('mnist_ckpt')
        if ckpt_path is not None:
            self.saver.restore(self.sess, ckpt_path) 
        return 0 if ckpt_path is None else int(ckpt_path.split('-')[-1])

def vizualize_feature():
    return

mnist = tf.contrib.learn.datasets.load_dataset("mnist")

batch_x, batch_y = mnist.train.next_batch(1)
mlp = MLP(res_checkpoint_dir='/home/ubuntu/feature_viz/resnet_v1_50.ckpt')
i = mlp.load()
batch_size = 128
while(mnist.train._epochs_completed < 9):
    batch_x, batch_y = mnist.train.next_batch(batch_size)
    mlp.train(batch_y, batch_x, i)
    i += 1
    if i % 100 == 0:
        mlp.save(i)
        print('EPOCH: ', mnist.train._epochs_completed)
        print('saving after {} iterations'.format(i))
