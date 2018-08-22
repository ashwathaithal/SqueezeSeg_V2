# Author: Bichen Wu (bichen@berkeley.edu) 08/25/2016

"""Model configuration for pascal dataset"""

import numpy as np

from config import base_model_config

def kitti_squeezeSeg_config():
  """Specify the parameters to tune below."""
  mc                    = base_model_config('KITTI')
  mc.NUM_BIN            = 10
  mc.BIN_INTERVALS       = np.array([100] * 10)
  mc.MID_VALUES         = np.array(range(50, 1000, 100))
  mc.BIN_LOSS_WEIGHT    =  np.array([5, 5, 4, 4, 3, 3, 2, 2, 1, 1])
  mc.BIN_LOSS_COEF      = 4
  mc.DELTA_LOSS_COEF    = 30e-4
  mc.SOFT_LABEL_SIGMA   = 0.5

  mc.BATCH_SIZE         = 48
  mc.AZIMUTH_LEVEL      = 512
  mc.ZENITH_LEVEL       = 64

  mc.LCN_HEIGHT         = 3
  mc.LCN_WIDTH          = 5
  mc.RCRF_ITER          = 3
  mc.BI_FILTER_COEF     = 0.1
  mc.ANG_FILTER_COEF    = 0.02
  mc.ECULIDEAN_LOSS_COEF = 15.0
  mc.CLS_LOSS_COEF      = 15.0
  mc.WEIGHT_DECAY       = 0.0001
  #mc.LEARNING_RATE      = 0.01
  mc.LEARNING_RATE      = 0.05
  mc.DECAY_STEPS        = 10000
  mc.MAX_GRAD_NORM      = 1.0
  mc.MOMENTUM           = 0.9
  mc.LR_DECAY_FACTOR    = 0.5

  mc.DATA_AUGMENTATION  = True
  mc.RANDOM_FLIPPING    = True

  # x, y, z, intensity, distance
  mc.INPUT_MEAN_KITTI = np.array([[[10.88, 0.23, -1.04, 0.21, 12.12]]])
  mc.INPUT_STD_KITTI = np.array([[[11.47, 6.91, 0.86, 0.16, 12.32]]])
  mc.INPUT_MEAN_GTAV = np.array([[[7.98, 0.22, -0.67, 8.91]]])
  mc.INPUT_STD_GTAV = np.array([[[9.82, 5.43, 0.73, 10.56]]])

  return mc
