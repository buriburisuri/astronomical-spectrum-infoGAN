import sugartensor as tf
import numpy as np
import matplotlib.pyplot as plt
from model import *


__author__ = 'namju.kim@kakaobrain.com'


# set log level to debug
tf.sg_verbosity(10)


#
# hyper parameters
#

batch_size = 100  # batch size


#
# inputs
#

# target continuous variable
target_cval = []
for _ in range(con_dim):
    target_cval.append(tf.placeholder(dtype=tf.sg_floatx, shape=batch_size))

# continuous variables
z = target_cval[0].sg_expand_dims()
for i in range(1, con_dim):
    z = z.sg_concat(target=target_cval[i].sg_expand_dims())

# random seed = continuous variable + random normal
z = z.sg_concat(target=tf.random_normal((batch_size, rand_dim)))

# generator
gen = generator(z).sg_squeeze(axis=(2, 3))


#
# run generator
#
def run_generator(fig_name, cval):

    with tf.Session() as sess:

        # init session
        tf.sg_init(sess)

        # restore parameters
        saver = tf.train.Saver()
        saver.restore(sess, tf.train.latest_checkpoint('asset/train'))

        feed_dic = {}
        for t, c in zip(target_cval, cval):
            feed_dic[t] = c

        # run generator
        imgs = sess.run(gen, feed_dic)

        # plot result
        _, ax = plt.subplots(10, 10, sharex=True, sharey=True)
        for i in range(10):
            for j in range(10):
                ax[i][j].plot(imgs[i * 10 + j])
                ax[i][j].set_axis_off()
        plt.savefig('asset/train/' + fig_name, dpi=600)
        tf.sg_info('Sample image saved to "asset/train/%s"' % fig_name)
        plt.close()


#
# draw sample by categorical division
#

# fake image
run_generator('fake.png',
              [np.random.rand(batch_size),
               np.random.rand(batch_size),
               np.random.rand(batch_size)])

#
# draw sample by continuous division
#

# continuous factor 0 to 1
run_generator('sample_0_1.png',
              [np.linspace(0, 1, 10).repeat(10),
               np.expand_dims(np.linspace(0, 1, 10), axis=1).repeat(10, axis=1).T.flatten(),
               np.ones(batch_size) * 0.5])

# continuous factor 1 to 2
run_generator('sample_1_2.png',
              [np.ones(batch_size) * 0.5,
               np.linspace(0, 1, 10).repeat(10),
               np.expand_dims(np.linspace(0, 1, 10), axis=1).repeat(10, axis=1).T.flatten()])

# continuous factor 0 to 1
run_generator('sample_0_2.png',
              [np.linspace(0, 1, 10).repeat(10),
               np.ones(batch_size) * 0.5,
               np.expand_dims(np.linspace(0, 1, 10), axis=1).repeat(10, axis=1).T.flatten()])
