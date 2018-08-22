# Author: Bichen Wu (bichen@berkeley.edu) 02/27/2017

"""The data base wrapper class"""

import os
import random
import shutil
import pdb

import numpy as np
import scipy
import scipy.ndimage
import py3d
from numpy import linalg as LA
from utils.util import *

class imdb(object):
  """Image database."""

  def __init__(self, name, mc):
    self._name = name
    self._image_set = []
    self._image_idx = []
    self._data_root_path = []
    self.mc = mc
    self._output_path = os.path.join('/rscratch18/schzhao/SqueezeSeg/data', 'gtav_predicted_R_final')
    # batch reader
    self._perm_idx = []
    self._cur_idx = 0

  @property
  def name(self):
    return self._name

  @property
  def image_idx(self):
    return self._image_idx

  @property
  def image_set(self):
    return self._image_set

  @property
  def data_root_path(self):
    return self._data_root_path

  def _shuffle_image_idx(self):
    self._perm_idx = [self._image_idx[i] for i in
        np.random.permutation(np.arange(len(self._image_idx)))]
    self._cur_idx = 0

  def calculate_normals(self, position_vectors):
    mc = self.mc
    point_cloud = py3d.PointCloud()
    position_vectors = np.reshape(position_vectors, (mc.ZENITH_LEVEL * mc.AZIMUTH_LEVEL, 3))
    point_cloud.points = py3d.Vector3dVector(position_vectors)#change numpy into point cloud object
    py3d.estimate_normals(point_cloud, \
                          search_param = py3d.KDTreeSearchParamHybrid(radius = 1.1, max_nn = 1500))#calculate the normals at all points in point cloud
    return np.array(point_cloud.normals)

  def calculate_RAndMulti(self, depths, normals, position_vectors, intensity):
    mc = self.mc
    position_vectors = np.reshape(position_vectors, (mc.ZENITH_LEVEL * mc.AZIMUTH_LEVEL, 3))
    dot_products = np.absolute((position_vectors * normals).sum(-1))
    intensity = np.reshape(intensity, (mc.ZENITH_LEVEL * mc.AZIMUTH_LEVEL))
    depths = np.reshape(depths,  (mc.ZENITH_LEVEL * mc.AZIMUTH_LEVEL))
    dp_mask = np.reshape((dot_products != 0), [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL])
    change_zero = lambda x: 1.0 if x == 0 else x 
    R = np.divide((intensity * np.power(depths, 3)), np.array([change_zero(d) for d in dot_products]))
    R = np.reshape(R, (mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL))
    multiplier = np.divide(dot_products, np.array([change_zero(d) for d in np.power(depths, 3)]))
    multiplier = np.reshape(multiplier, (mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL))
    return R, multiplier, dp_mask

  def read_batch(self, shuffle=True):
    """Read a batch of lidar data including labels. Data formated as numpy array
    of shape: height x width x {x, y, z, intensity, range, label}.
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      lidar_per_batch: LiDAR input. Shape: batch x height x width x 5.
      lidar_mask_per_batch: LiDAR mask, 0 for missing data and 1 otherwise.
        Shape: batch x height x width x 1.
      bin_per_batch: point-wise labels. Shape: batch x height x width.
    """
    mc = self.mc
    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    lidar_per_batch = []
    lidar_mask_per_batch = []
    intensity_per_batch = []
    multiplier_per_batch = []
    delta_per_batch = []
    bin_per_batch = []

    for idx in batch_idx:
      # load data
      # loading from npy is 30x faster than loading from pickle
      record = np.load(self._lidar_2d_path_at(idx, gta = False)).astype(np.float32, copy=False)
      INPUT_MEAN = mc.INPUT_MEAN_KITTI  
      INPUT_STD = mc.INPUT_STD_KITTI
      #comment this part out when saving the data for the first time
      if mc.DATA_AUGMENTATION:
        if mc.RANDOM_FLIPPING:
          if np.random.rand() > 0.5:
            # flip y
            record = record[:, ::-1, :]
            record[:, :, 1] *= -1
            INPUT_MEAN[:,:,1] *= -1
            flip = True
          else:
            flip = False
      
      lidar = record[:, :, :5] # x, y, z, intensity, depth
      intensity = record[:, :, 3]
      if os.path.exists(self._R_path_at(idx, gta = False)) and os.path.exists(self._multiplier_path_at(idx, gta = False))\
           and os.path.exists(self._mask_path_at(idx, gta = False)):
        multiplier = np.load(self._multiplier_path_at(idx, gta = False)).astype(np.float32, copy = False)
        lidar_mask = np.load(self._mask_path_at(idx, gta = False)).astype(bool, copy = False)
        R = np.load(self._R_path_at(idx, gta = False)).astype(np.float32, copy = False)
        if flip:
          multiplier = multiplier[:,::-1]
          lidar_mask = lidar_mask[:,::-1]
          R = R[:,::-1]
      else:
        depths = record[:,:,4]
        position_vectors = lidar[:,:,[0, 1, 2]]
        normals = self.calculate_normals(position_vectors)
        R, multiplier, dp_mask = self.calculate_RAndMulti(depths, normals, position_vectors, intensity)
        depth_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL]
        )
        R_mask = np.reshape((R[:,:,] < 1000), [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL])
        lidar_mask = np.logical_and(dp_mask, np.logical_and(depth_mask, R_mask))
        if flip:
          np.save(self._R_path_at(idx, gta = False), R[:,::-1])
          np.save(self._multiplier_path_at(idx, gta = False), multiplier[:,::-1])
          np.save(self._mask_path_at(idx, gta = False), lidar_mask[:,::-1])
        else:
          np.save(self._R_path_at(idx, gta = False), R)
          np.save(self._multiplier_path_at(idx, gta = False), multiplier)
          np.save(self._mask_path_at(idx, gta = False), lidar_mask)
      lidar_mask = np.reshape(lidar_mask, [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1])
      lidar = (lidar - INPUT_MEAN) / INPUT_STD     
      lidar = np.delete(lidar, 3, 2)
      bin_label = np.zeros((mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, mc.NUM_BIN))
      delta_label = np.zeros((mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, mc.NUM_BIN))
      for l in range(mc.NUM_BIN):
        bin_mask = np.logical_and(
           R >= mc.MID_VALUES[l] - mc.BIN_INTERVALS[l]/2,
           R < mc.MID_VALUES[l] + mc.BIN_INTERVALS[l]/2)
        bin_label[:, :, l] = 1.0 * bin_mask
        delta_label[:, :, l] = (R - mc.MID_VALUES[l]) * bin_mask
      bin_label = scipy.ndimage.filters.gaussian_filter1d(bin_label, mc.SOFT_LABEL_SIGMA)
      # Append all the data
      lidar_per_batch.append(lidar)
      lidar_mask_per_batch.append(lidar_mask)
      bin_per_batch.append(bin_label)
      delta_per_batch.append(delta_label)
      intensity_per_batch.append(intensity)
      multiplier_per_batch.append(multiplier)

    if len(lidar_per_batch) != mc.BATCH_SIZE or \
           len(lidar_mask_per_batch) != mc.BATCH_SIZE or \
           len(intensity_per_batch) != mc.BATCH_SIZE or \
           len(multiplier_per_batch) != mc.BATCH_SIZE or \
           len(bin_per_batch) != mc.BATCH_SIZE or \
           len(delta_per_batch) != mc.BATCH_SIZE:
      print(batch_idx)
      
    assert len(lidar_per_batch) == mc.BATCH_SIZE and \
           len(lidar_mask_per_batch) == mc.BATCH_SIZE and \
           len(intensity_per_batch) == mc.BATCH_SIZE and \
           len(multiplier_per_batch) == mc.BATCH_SIZE and \
           len(bin_per_batch) == mc.BATCH_SIZE and \
           len(delta_per_batch) == mc.BATCH_SIZE, \
           'imdb: data batch size error'   
    return np.array(lidar_per_batch), np.array(lidar_mask_per_batch), \
            np.array(intensity_per_batch), np.array(multiplier_per_batch), \
            np.array(bin_per_batch), np.array(delta_per_batch)
    

  def read_batch_with_filepath(self, shuffle=True):
    """Read a batch of lidar data including labels. Data formated as numpy array
    of shape: height x width x {x, y, z, intensity, range, label}.
    Args:
      shuffle: whether or not to shuffle the dataset
    Returns:
      lidar_per_batch: LiDAR input. Shape: batch x height x width x 5.
      lidar_mask_per_batch: LiDAR mask, 0 for missing data and 1 otherwise.
        Shape: batch x height x width x 1.
      bin_per_batch: point-wise labels. Shape: batch x height x width.
    """
    mc = self.mc

    if shuffle:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        self._shuffle_image_idx()
      batch_idx = self._perm_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
      self._cur_idx += mc.BATCH_SIZE
    else:
      if self._cur_idx + mc.BATCH_SIZE >= len(self._image_idx):
        batch_idx = self._image_idx[self._cur_idx:] \
            + self._image_idx[:self._cur_idx + mc.BATCH_SIZE-len(self._image_idx)]
        self._cur_idx += mc.BATCH_SIZE - len(self._image_idx)
      else:
        batch_idx = self._image_idx[self._cur_idx:self._cur_idx+mc.BATCH_SIZE]
        self._cur_idx += mc.BATCH_SIZE

    lidar_per_batch = []
    lidar_mask_per_batch = []
    intensity_per_batch = []
    multiplier_per_batch = []
    delta_per_batch = []
    bin_per_batch = []
    filepath_per_batch = []
    record_per_batch = []
    
    for idx in batch_idx:
      # load data
      # loading from npy is 30x faster than loading from pickle
      filepath = os.path.join(self._output_path, idx+'.npy')
      filepath_per_batch.append(filepath)
      record = np.load(self._lidar_2d_path_at(idx, gta = True)).astype(np.float32, copy=False)
      record_per_batch.append(record)
      INPUT_MEAN = mc.INPUT_MEAN_GTAV
      INPUT_STD = mc.INPUT_STD_GTAV
      #comment this part out when saving the data for the first time
      
      lidar = record[:, :, :5] # x, y, z, intensity, depth
      intensity = record[:, :, 3]
      if os.path.exists(self._R_path_at(idx, gta = True)) and os.path.exists(self._multiplier_path_at(idx, gta = True))\
           and os.path.exists(self._mask_path_at(idx, gta = True)):
        multiplier = np.load(self._multiplier_path_at(idx, gta = True)).astype(np.float32, copy = False)
        lidar_mask = np.load(self._mask_path_at(idx, gta = True)).astype(bool, copy = False)
        R = np.load(self._R_path_at(idx, gta = True)).astype(np.float32, copy = False)
      else:
        depths = record[:,:,4]
        position_vectors = lidar[:,:,[0, 1, 2]]
        normals = self.calculate_normals(position_vectors)
        R, multiplier, dp_mask = self.calculate_RAndMulti(depths, normals, position_vectors, intensity)
        depth_mask = np.reshape(
            (lidar[:, :, 4] > 0),
            [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL]
        )
        R_mask = np.reshape((R[:,:,] < 1000), [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL])
        lidar_mask = np.logical_and(dp_mask, np.logical_and(depth_mask, R_mask))
        np.save(self._R_path_at(idx, gta = True), R)
        np.save(self._multiplier_path_at(idx, gta = True), multiplier)
        np.save(self._mask_path_at(idx, gta = True), lidar_mask)
      
      lidar_mask = np.reshape(lidar_mask, [mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, 1])
      lidar = (lidar - INPUT_MEAN) / INPUT_STD     
      lidar = np.delete(lidar, 3, 2)
      bin_label = np.zeros((mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, mc.NUM_BIN))
      delta_label = np.zeros((mc.ZENITH_LEVEL, mc.AZIMUTH_LEVEL, mc.NUM_BIN))
      for l in range(mc.NUM_BIN):
        bin_mask = np.logical_and(
           R >= mc.MID_VALUES[l] - mc.BIN_INTERVALS[l]/2,
           R < mc.MID_VALUES[l] + mc.BIN_INTERVALS[l]/2)
        bin_label[:, :, l] = 1.0 * bin_mask
        delta_label[:, :, l] = (R - mc.MID_VALUES[l]) * bin_mask
      bin_label = scipy.ndimage.filters.gaussian_filter1d(bin_label, mc.SOFT_LABEL_SIGMA)

      # Append all the data
      lidar_per_batch.append(lidar)
      lidar_mask_per_batch.append(lidar_mask)
      bin_per_batch.append(bin_label)
      delta_per_batch.append(delta_label)
      intensity_per_batch.append(intensity)
      multiplier_per_batch.append(multiplier)

    assert len(lidar_per_batch) == mc.BATCH_SIZE and \
           len(lidar_mask_per_batch) == mc.BATCH_SIZE and \
           len(intensity_per_batch) == mc.BATCH_SIZE and \
           len(multiplier_per_batch) == mc.BATCH_SIZE and \
           len(bin_per_batch) == mc.BATCH_SIZE and \
           len(delta_per_batch) == mc.BATCH_SIZE and \
           len(record_per_batch) == mc.BATCH_SIZE and \
           len(filepath_per_batch) == mc.BATCH_SIZE, \
           'imdb: data batch size error'   

    return np.array(lidar_per_batch), np.array(lidar_mask_per_batch), \
            np.array(intensity_per_batch), np.array(multiplier_per_batch), \
            np.array(bin_per_batch), np.array(delta_per_batch),\
            np.array(filepath_per_batch), np.array(record_per_batch)
                
  def evaluate_detections(self):
    raise NotImplementedError
