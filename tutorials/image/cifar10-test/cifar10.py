import tensorflow as tf
import os
import sys
import tarfile

from six.moves import urllib

import cifar10_input


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('data_dir', '/tmp2/cifar10_data', 
                           """数据集目录""")
tf.app.flags.DEFINE_integer('batch_size', 128, 
                           """批处理量""")

# 数据集下载地址
DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

def distorted_inputs():
    """获取输入"""
    if not FLAGS.data_dir:
        raise ValueError('请提供原始数据目录 data_dir')
    data_dir = os.path.join(FLAGS.data_dir, 'cifar-10-batches-bin')
    iamges, lables = cifar10_input.distorted_inputs(data_dir=data_dir, 
                                                    bath_size=FLAGS.batch_size)
    return iamges, lables

def maybe_download_and_extract():
    """下载并且解压数据"""
    dest_dir = FLAGS.data_dir
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    
    # 文件不存在即下载
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, 
                float(count * block_size) / float(total_size) * 100.0))
            sys.stdout.flush()
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        print()
        statinfo = os.stat(filepath)
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    
    # 解压数据
    extract_dir_path = os.path.join(dest_dir, 'cifar-10-batches-bin')
    if not os.path.exists(extract_dir_path):
        tarfile.open(filename, 'r:gz').extractall(dest_dir)
        
    
