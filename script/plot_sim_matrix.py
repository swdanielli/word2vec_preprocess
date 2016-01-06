#! /usr/bin/env python

import numpy as np
import math
import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

def quantiztion(val, quantiztion_bins=None):
  if math.isnan(float(val)):
    return 0
  elif quantiztion_bins:
    val = int(float(val)*quantiztion_bins)/float(quantiztion_bins-1)
    return 1 if val > 1 else val
  else:
    return float(val)

def plot_heat_map(cos_sim_matrix, filename, is_tx):
  fig = plt.figure()
  plt.clf()
  ax = fig.add_subplot(111)
  #if is_tx:
  #  res = ax.imshow(np.array(cos_sim_matrix).transpose(), cmap=plt.cm.Greys_r, interpolation='nearest')
  #else:
  res = ax.imshow(np.array(cos_sim_matrix), cmap=plt.cm.Greys_r, interpolation='nearest')
  #res = ax.imshow(np.array(cos_sim_matrix), cmap=plt.cm.jet, interpolation='nearest')
  ax.set_aspect('auto')

  cb = fig.colorbar(res)
  alphabet = '123456789'
  if not is_tx:
    width = len(cos_sim_matrix[0])
    plt.xticks(range(width), alphabet[:width])
  plt.savefig('%s.pdf' % filename, format='pdf')
  plt.close()

def _main( ):
  mapping = sys.argv[1]
  word2vec_fea_dir = sys.argv[2]
  word2vec_plot_dir = sys.argv[3]
  quantiztion_bins = None
  is_tx = False
  if len(sys.argv) > 4:
    quantiztion_bins = int(sys.argv[4])
  if len(sys.argv) > 5:
    is_tx = bool(int(sys.argv[5]))

  if is_tx:
    cos_sim_all = []
  for l_id in open(mapping):
    l_id = l_id.strip()
    with open('%s/%s' % (word2vec_fea_dir, l_id)) as f:
      try:
        cos_sim_matrix = [[quantiztion(y, quantiztion_bins) for y in x.strip().split('\t')] for x in f.readlines()]
        if is_tx:
          cos_sim_all += cos_sim_matrix
      except ValueError:
        print '%s/%s' % (word2vec_fea_dir, l_id)

    plot_heat_map(cos_sim_matrix, '%s/%s' % (word2vec_plot_dir, l_id), is_tx)

  if is_tx:
    plot_heat_map(cos_sim_all, '%s/accumulation' % word2vec_plot_dir, is_tx)

if __name__ == '__main__':
  _main( )
