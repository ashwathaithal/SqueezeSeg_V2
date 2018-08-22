# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Train"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import os.path
import sys
import time

import math
import numpy as np
from six.moves import xrange
import tensorflow as tf
import threading

from config import *
from imdb import kitti
from utils.util import *
from nets import *

import pdb

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('dataset', 'KITTI',
                           """Currently only support KITTI dataset.""")
tf.app.flags.DEFINE_string('data_path', '', """Root directory of data""")
tf.app.flags.DEFINE_string('image_set', 'train',
                           """ Can be train, trainval, val, or test""")
tf.app.flags.DEFINE_string('train_dir', '/tmp/bichen/logs/squeezeseg/train',
                            """Directory where to write event logs """
                            """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 1000000,
                            """Maximum number of batches to run.""")
tf.app.flags.DEFINE_string('net', 'squeezeSeg',
                           """Neural net architecture. """)
tf.app.flags.DEFINE_string('pretrained_model_path', '',
                           """Path to the pretrained model.""")
tf.app.flags.DEFINE_integer('summary_step', 50,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_integer('checkpoint_step', 500,
                            """Number of steps to save summary.""")
tf.app.flags.DEFINE_string('gpu', '0', """gpu id.""")

def train():
  #pdb.set_trace()
  """Train SqueezeSeg model"""
  assert FLAGS.dataset == 'KITTI', \
      'Currently only support KITTI dataset'

  os.environ['CUDA_VISIBLE_DEVICES'] = FLAGS.gpu

  with tf.Graph().as_default():

    assert FLAGS.net == 'squeezeSeg', \
        'Selected neural net architecture not supported: {}'.format(FLAGS.net)


    if FLAGS.net == 'squeezeSeg':
      mc = kitti_squeezeSeg_config()
      mc.PRETRAINED_MODEL_PATH = FLAGS.pretrained_model_path
      model = SqueezeSeg(mc)

    imdb = kitti(FLAGS.image_set, FLAGS.data_path, mc)

    # save model size, flops, activations by layers
    with open(os.path.join(FLAGS.train_dir, 'model_metrics.txt'), 'w') as f:
      f.write('Number of parameter by layer:\n')
      count = 0
      for c in model.model_size_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nActivation size by layer:\n')
      for c in model.activation_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))

      count = 0
      f.write('\nNumber of flops by layer:\n')
      for c in model.flop_counter:
        f.write('\t{}: {}\n'.format(c[0], c[1]))
        count += c[1]
      f.write('\ttotal: {}\n'.format(count))
    f.close()
    print ('Model statistics saved to {}.'.format(
      os.path.join(FLAGS.train_dir, 'model_metrics.txt')))

    def enqueue(sess, coord):
      with coord.stop_on_exception():
        while not coord.should_stop():
          # read batch input
          lidar_per_batch, lidar_mask_per_batch,\
              intensity_per_batch,\
              multiplier_per_batch, bin_per_batch, delta_per_batch = imdb.read_batch()

          feed_dict = {
              model.ph_keep_prob: mc.KEEP_PROB,
              model.ph_lidar_input: lidar_per_batch,
              model.ph_lidar_mask: lidar_mask_per_batch,
              model.ph_intensity: intensity_per_batch,
              model.ph_multiplier: multiplier_per_batch,
              model.ph_bin: bin_per_batch,
              model.ph_delta: delta_per_batch
          }

          sess.run(model.enqueue_op, feed_dict=feed_dict)
   
    saver = tf.train.Saver(tf.all_variables(), max_to_keep=10000)
    summary_op = tf.summary.merge_all()
    init = tf.initialize_all_variables()

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    sess.run(init)
    #saver.restore(sess, tf.train.latest_checkpoint('./log/trainV1/')) 
    
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    coord = tf.train.Coordinator()
    enq_threads = []
    for _ in range(mc.NUM_ENQUEUE_THREAD):
      eqth = threading.Thread(target=enqueue, args=[sess, coord])
      eqth.start()
      enq_threads.append(eqth)

    run_options = tf.RunOptions(timeout_in_ms=60000)
    try:
      for step in xrange(FLAGS.max_steps):
        start_time = time.time()

        if step % FLAGS.summary_step == 0 or step == FLAGS.max_steps-1:
          op_list = [
              model.lidar_input, model.lidar_mask, model.intensity,\
               model.multiplier, model.bin, model.delta, model.train_op,\
              model.loss, model.bin_loss, model.delta_loss, model.pred_R, model.pred_intensity, model.pred_bin,\
              model.pred_delta, summary_op
          ]

          lidar_per_batch, lidar_mask_per_batch, \
              intensity_per_batch, multiplier_per_batch,\
              bin_per_batch, delta_per_batch,\
              _, loss_value, bin_loss_value, delta_loss_value, pred_R, pred_intensity, pred_bin, \
              pred_delta, summary_str = sess.run(op_list)

          #ground_truth_intensity, pred_intensity, 
          label_image = intensity_per_batch[:6, :, :]
          label_image = label_image[:, :, :, np.newaxis]
          pred_image = pred_intensity[:6, :, :]
          pred_image = pred_image[:, :, :, np.newaxis]

          se, sam_num = evaluate_se(intensity_per_batch, pred_intensity \
              * np.squeeze(lidar_mask_per_batch))

          sam_num_mask = lidar_mask_per_batch.size

          feed_dict = {}
          feed_dict[model.mse_summary_placeholders[0]] = np.log(se/sam_num_mask)

          mse_summary_list = sess.run(model.mse_summary_ops[0], feed_dict)

          # Run visualization
          viz_op_list = [model.show_label, model.show_depth_img, model.show_pred]
          viz_summary_list = sess.run(
              viz_op_list, 
              feed_dict={
                  model.depth_image_to_show: lidar_per_batch[:6, :, :, [3]],
                  model.label_to_show: label_image,
                  model.pred_image_to_show: pred_image,
              }
          )

          # Add summaries
          summary_writer.add_summary(summary_str, step)

          summary_writer.add_summary(mse_summary_list, step)

          for viz_sum in viz_summary_list:
            summary_writer.add_summary(viz_sum, step)

          # force tensorflow to synchronise summaries
          summary_writer.flush()

        else:
          _, loss_value, bin_loss_value, delta_loss_value = sess.run(
              [model.train_op, model.loss, model.bin_loss, model.delta_loss])

        duration = time.time() - start_time

        assert not np.isnan(loss_value), \
            'Model diverged. Total loss: {}, conf_loss: {}, bbox_loss: {}, ' \
            'class_loss: {}'.format(loss_value, conf_loss, bbox_loss,
                                    class_loss)

        if step % 10 == 0:
          num_images_per_step = mc.BATCH_SIZE
          images_per_sec = num_images_per_step / duration
          sec_per_batch = float(duration)
          format_str = ('%s: step %d, loss = %.4f, bin_loss = %.4f, delta_loss = %.4f (%.1f images/sec; %.3f '
                        'sec/batch)')
          print (format_str % (datetime.now(), step, loss_value, bin_loss_value, delta_loss_value,
                               images_per_sec, sec_per_batch))
          sys.stdout.flush()

        # Save the model checkpoint periodically.
        if step % FLAGS.checkpoint_step == 0 or step == FLAGS.max_steps-1:
          checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
          saver.save(sess, checkpoint_path, global_step=step)
    except Exception, e:
      coord.request_stop(e)
    finally:
      coord.request_stop()
      sess.run(model.q.close(cancel_pending_enqueues=True))
      coord.join(enq_threads)


def main(argv=None):  # pylint: disable=unused-argument
  if tf.gfile.Exists(FLAGS.train_dir):
    tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()
