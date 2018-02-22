from object_detection.models import faster_rcnn_resnet_v1_feature_extractor as faster_rcnn_resnet_v1
import tensorflow as tf
import cv2
import numpy as np

def feature_maps(img_locs):
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
        for i, loc_i in enumerate(img_locs):
            print('feature embed iter {}/{}'.format(i, len(img_locs)))
            original_img = cv2.imread(loc_i)
            #original_img = cv2.resize(original_img, dsize=(320, 240))
            original_img = np.expand_dims(
                original_img, axis=0).astype(float)
            learned_embedding = sess.run(rpn_feature_map, feed_dict={
                                         img: original_img}).flatten()
            return learned_embedding

features = feature_maps(['p.jpg'])
import pdb; pdb.set_trace()

