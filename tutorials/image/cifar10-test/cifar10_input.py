import tensorflow as tf
import os
import sys

def distorted_inputs(data_dir, batch_size):
    filenames = [os.path.join(data_dir, 'data_batch_%d.bin' % i)
                 for i in xrange(1, 6)]
    
    for file in filenames:
        if not tf.gfile.Exists(file):
            raise ValueError("文件不存在：" + file)
    
    with tf.name_scope('data_augmentation'):
        
    