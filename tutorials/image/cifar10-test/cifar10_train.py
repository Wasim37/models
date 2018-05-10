import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp2/cifar10_train',
                           """存放event日志和checkpoint的目录""")

def train():
    with tf.Graph().as_default():
        cifar10.distorted_inputs()
    pass

def main(argv=None):
    cifar10.maybe_download_and_extract()
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()

if __name__ == '__main__':
    tf.app.run()