from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1
import tensorflow as tf
import cv2
import numpy as np

def res_feature_maps(imgs):
    img = tf.placeholder(dtype='float32', shape=(None, None, None, None))
    FeatureExtractor = faster_rcnn_resnet_v1.FasterRCNNResnet50FeatureExtractor(
        is_training=False,
        first_stage_features_stride=16,
        reuse_weights=None,
        weight_decay=0.0)
    preprocessed_input = FeatureExtractor.preprocess(img)
    rpn_feature_map = FeatureExtractor.extract_proposal_features(preprocessed_input,
                                                                 scope='TestScope')

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        for i, original_img in enumerate(imgs):
            print('feature embed iter {}/{}'.format(i, len(imgs)))
            original_img = np.reshape(original_img, [28, 28, 1]) # [batch, height, width, channels]
            original_img = cv2.resize(original_img, (33, 33)) # gotta be atleast 33x33 b/c resnet feature extractor is a joke
            original_img = np.reshape(original_img, [-1, 33, 33, 1])
            import pdb; pdb.set_trace()
            learned_embedding = sess.run(rpn_feature_map, feed_dict={img: original_img}).flatten()
            return learned_embedding

def vizualize_feature():
    return

mnist = tf.contrib.learn.datasets.load_dataset("mnist")
train_data = mnist.train.images
features = res_feature_maps([train_data[0]])
import pdb; pdb.set_trace()

