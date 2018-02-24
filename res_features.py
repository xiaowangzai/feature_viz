from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1
import tensorflow as tf
from tensorflow.contrib import slim
import cv2
import numpy as np
from nets import nets_factory

res_scope = nets_factory.arg_scopes_map['resnet_v1_50']
N_CLASSES=10
class MLP:
    def __init__(self, feature_shape):
        self.extracted_features = tf.placeholder(tf.float32, [None, feature_shape])
        self.build(self.extracted_features)

        self.labels = tf.placeholder(tf.int64, [None])
        y = tf.one_hot(self.labels, depth=N_CLASSES, dtype=tf.int64)
        
        loss = tf.losses.mean_squared_error(y, self.y_hat)
        opt = tf.train.AdamOptimizer()
        self.update = opt.minimize(loss)
        n_equals = tf.cast(tf.equal(tf.argmax(y, axis=1), tf.argmax(self.y_hat, axis=1)), dtype=tf.float32)
        self.accuracy = tf.reduce_mean(n_equals)
        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

    def build(self, features):
        l1 = tf.layers.dense(features, 200, activation=tf.nn.relu)
        l2 = tf.layers.dense(l1, 200, activation=tf.nn.relu)
        self.y_hat = tf.layers.dense(l2, N_CLASSES, activation=tf.nn.softmax)
    
    def test(self, labels, features, i):
        accuracy = self.sess.run(self.accuracy, feed_dict={self.extracted_features: features, self.labels:labels})
        print('iter {} test accuracy: {}'.format(i, accuracy))

    def train(self, labels, features, i):
        _, acc = self.sess.run([self.update, self.accuracy], feed_dict={self.extracted_features: features, self.labels: labels}) 
        print('iter {} train accuracy: {}'.format(i, acc))

class ResFeatureMap:
    def __init__(self, checkpoint_dir):
        self.imgs = tf.placeholder(dtype='float32', shape=(None, None, None, None))
        
        FeatureExtractor = faster_rcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor(
            is_training=False,
            first_stage_features_stride=16,
            reuse_weights=None,
            weight_decay=0.0)
        
        preprocessed_input = FeatureExtractor.preprocess(self.imgs)
        self.rpn_feature_map = FeatureExtractor.extract_proposal_features(preprocessed_input, scope='test_scope')
        #self._network_fn = nets_factory.get_network_fn('resnet_v1_50', num_classes=num_classes, is_training=False)
        #variables_to_restore = slim.get_variables_to_restore()
        #checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        #restore_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

        self.sess = tf.Session()
        #restore_fn(self.sess)
        self.sess.run(tf.global_variables_initializer())

    def generate(self, imgs):
        transformed_imgs = []
        for i in range(len(imgs)):
            new_img = np.reshape(imgs[i], [28, 28, 1])# [batch, height, width, channels]
            new_img = cv2.resize(new_img, (33, 33))# gotta be atleast 33x33 b/c resnet feature extractor is a joke

            transformed_imgs.append(new_img)
        transformed_imgs = np.reshape(transformed_imgs, [-1, 33, 33, 1])
        learned_embeddings = self.sess.run(self.rpn_feature_map, feed_dict={self.imgs: transformed_imgs})
        learned_embeddings = np.reshape(learned_embeddings, [len(imgs), -1])
        return learned_embeddings 

def vizualize_feature():
    return

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
train_labels = np.asarray(mnist.train.labels, dtype=np.int32)

res_feature_map = ResFeatureMap(checkpoint_dir='/home/ubuntu/feature_viz/res50')
features = np.reshape(res_feature_map.generate([train_data[0]]), [1, -1])
mlp = MLP(features.shape[-1])
batch_size = 128
for i in range(0, train_data.shape[0], batch_size):
    x = train_data[i:i+batch_size]
    y = train_labels[i:i+batch_size] 
    features = res_feature_map.generate(x)

    mlp.train(y, features, i)

