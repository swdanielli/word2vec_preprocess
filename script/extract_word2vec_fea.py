#! /usr/bin/env python

nlp_util_dir = '/usr/users/swli/program/nlp_util'

# import modules & set up logging
import gensim, logging
import math
import numpy as np
import re
from scipy import spatial
import sys
sys.path.append(nlp_util_dir)

import load_courseware
import my_util

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

trans_doc_prefix = 'TRANS'
slides_doc_prefix = 'SLIDES'
tx_doc_prefix = 'TX'

class my_sentences(object):
  def __init__(self, corpus_name):
    self.corpus_name = corpus_name

  def __iter__(self):
    with open(self.corpus_name, 'r') as f:
      for line in f:
        yield line.strip().split('\t')[0].split(' ')

def mean(data):
  """Return the sample arithmetic mean of data."""
  n = len(data)
  if n < 1:
    raise ValueError('mean requires at least one data point')
  return sum(data)/float(n) # in Python 2 use sum(data)/float(n)

def _ss(data):
  """Return sum of square deviations of sequence data."""
  c = mean(data)
  ss = sum((x-c)**2 for x in data)
  return ss

def std(data):
  """Calculates the population standard deviation."""
  n = len(data)
  if n < 2:
    raise ValueError('variance requires at least two data points')
  ss = _ss(data)
  pvar = ss/float(n-1) # the population variance
  return pvar**0.5

def get_doc_size(doc_prefix, l_id, labels):
  doc_size = 1
  while get_doc_tag(doc_prefix, l_id, doc_size) in labels:
    doc_size += 1
  return doc_size - 1

def get_doc_tag(doc_prefix, l_id, index=None):
  if doc_prefix in [trans_doc_prefix, slides_doc_prefix]:
    return '%s_%s_%d' % (doc_prefix, l_id, index)
  elif doc_prefix in [tx_doc_prefix]:
    return '%s_%s' % (doc_prefix, l_id)

def filter_doc_by_pos(doc, pos_tags, pos_type):
  if len(doc) != len(pos_tags):
    raise ValueError
  words = []
  if pos_type in ['type_1']:
    retained_pos = ['CD', 'FW', 'JJ', 'JJR', 'JJS', 'LS', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
  elif pos_type in ['type_2']:
    retained_pos = ['CD', 'FW', 'JJ', 'JJR', 'JJS', 'LS', 'NN', 'NNP', 'NNPS', 'NNS', 'RB', 'RBR', 'RBS']
  elif pos_type in ['type_3']:
    retained_pos = ['CD', 'FW', 'JJ', 'JJR', 'JJS', 'LS', 'NN', 'NNP', 'NNPS', 'NNS']
  elif pos_type in ['type_4']:
    retained_pos = ['CD', 'FW', 'LS', 'NN', 'NNP', 'NNPS', 'NNS']
  elif pos_type in ['None']:
    retained_pos = None

  for index in range(0, len(doc)):
    if not retained_pos or pos_tags[index] in retained_pos:
      words.append(doc[index])
  return words

def get_docs(doc_prefix, l_id, corpus, corpus_pos, pos_type):
  docs = []
  doc_size = get_doc_size(doc_prefix, l_id, corpus.keys())
  for index in range(1, doc_size+1):
    doc_tag = get_doc_tag(doc_prefix, l_id, index)
    docs.append(filter_doc_by_pos(corpus[doc_tag], corpus_pos[doc_tag], pos_type))
  return docs

def get_tx_docs(corpus, corpus_pos, pos_type, tx_list_name):
  docs = []
  for line in open(tx_list_name):
    l_id = re.match('C(\d+_\d+).html\t', line).group(1)
    doc_tag = get_doc_tag(tx_doc_prefix, l_id)
    docs.append(filter_doc_by_pos(corpus[doc_tag], corpus_pos[doc_tag], pos_type))
  return docs

def load_corpus(corpus_name):
  corpus = {}
  for line in open(corpus_name):
    items = line.strip().split('\t')
    corpus[items[1]] = items[0].split(' ')
  return corpus

def windowing(docs_temp, window_size):
  docs = []
  for center in range(len(docs_temp)):
    doc = []
    for index in range(-1*window_size+center, window_size+center+1):
      if index < 0 or index >= len(docs_temp):
        continue
      doc += docs_temp[index][:]
    docs.append(doc)
  return docs

def window_by_seg(docs_temp, seg_filename):
  docs = []
  prev_label = '0'
  doc_index = 0
  for label in open(seg_filename):
    label = label.strip()
    if label != prev_label:
      docs.append([])
      prev_label = label
    docs[-1] += docs_temp[doc_index][:]
    doc_index += 1
  return docs

def get_one_hot_vec(doc, stop_list, oov_dict):
  arr = [0 for _ in range(len(oov_dict))]
  for word in doc:
    if word in stop_list:
      continue
    if word in oov_dict:
      arr[oov_dict.index(word)] += 1
  return arr

def get_words2vecs(doc, model, dim, stop_list, is_normalization_word=False, fix_stem=None):
  if fix_stem:
    (refined_oov, stem_map, oov_map) = fix_stem
    doc_oov_refined = []
    for word in doc:
      if word not in model and word not in refined_oov:
        continue
      doc_oov_refined += (re.split('\s+', refined_oov[word]) if (word in refined_oov and bool(refined_oov[word])) else [word])

    doc = []
    for word in doc_oov_refined:
      stemmed_word = (' '.join(my_util.preprocess_content(word, stop_list, is_math=True, is_stemming=True))).strip()
      if stemmed_word:
        if stemmed_word in stem_map:
          doc.append(stem_map[stemmed_word])
        elif stemmed_word in oov_map:
          doc.append(oov_map[stemmed_word])

  arr = np.empty((0, (dim + len(oov_map.keys()) if fix_stem else dim)))
  for word in doc:
    if word in stop_list or (not fix_stem and word not in model):
      continue
    if not fix_stem:
      arr = np.vstack(( arr, normalize_sum(model[word], is_normalization_word) ))
    elif isinstance(word, (int, long)):
      word_arr = np.zeros( dim+len(oov_map.keys()) )
      word_arr[word] = 1.0
      arr = np.vstack((arr, word_arr))
    else:
      # consider normalize sum(model[word]) to 1
      arr = np.vstack((
        arr,
        np.hstack((
          np.zeros( len(oov_map.keys()) ),
          normalize_sum(model[word], is_normalization_word)
        ))
      ))
  return arr

def normalize_sum(arr, is_normalize):
  return arr/abs(sum(arr)) if ( is_normalize and bool(sum(arr)) ) else arr

def get_doc2vec(arr, is_normalization=False):
  arr = np.mean(arr, axis=0)
  # normalizing doc2vec
  if is_normalization:
    h = max(arr)
    l = min(arr)
    if h == l:
      arr -= h
    else:
      arr = 2*(arr-l)/(h-l)-1
  return arr

def compute_cos_sim_one_hot(doc_1, doc_2, stop_list, oov_dict):
  vec_1 = get_one_hot_vec(doc_1, stop_list, oov_dict)
  vec_2 = get_one_hot_vec(doc_2, stop_list, oov_dict)
  cos_sim = 1 - spatial.distance.cosine(vec_1, vec_2)
  return cos_sim if not math.isnan(cos_sim) else 0

def compute_cos_sim(doc_1, doc_2, model, dim, stop_list, is_normalization=False, is_normalization_word=False, fix_stem=None):
  words2vecs_1 = get_words2vecs(doc_1, model, dim, stop_list, is_normalization_word, fix_stem)
  words2vecs_2 = get_words2vecs(doc_2, model, dim, stop_list, is_normalization_word, fix_stem)

  if words2vecs_1.shape[0] == 0 or words2vecs_2.shape[0] == 0:
    return 0

  doc2vec_1 = get_doc2vec(words2vecs_1, is_normalization)
  doc2vec_2 = get_doc2vec(words2vecs_2, is_normalization)
  return 1 - spatial.distance.cosine(doc2vec_1, doc2vec_2)

def normalize_cos_sim(cos_sims, is_normalization_cos_sim):
  cos_sims = np.array(cos_sims)
  if is_normalization_cos_sim == 'shift':
    h = np.max(cos_sims, axis=0)
    l = np.min(cos_sims, axis=0)
    for c in range(np.size(cos_sims, axis=1)):
      cos_sims[:, c] = (cos_sims[:, c]-l[c])/(h[c]-l[c])
  elif is_normalization_cos_sim == 'shift_v2':
    for c in range(np.size(cos_sims, axis=1)):
      try:
        h = np.max(filter(lambda x: x > 0, cos_sims[:, c]))
        l = np.min(filter(lambda x: x > 0, cos_sims[:, c]))
      except ValueError:
        (h, l) = (0, 0)
      for r in range(np.size(cos_sims, axis=0)):
        if cos_sims[r, c] > 0 and h != l:
          cos_sims[r, c] = (cos_sims[r, c]-l)/(h-l)
        else:
          cos_sims[r, c] = 0
  elif is_normalization_cos_sim == 'shift_v3':
    for c in range(np.size(cos_sims, axis=1)):
      try:
        p_h = np.max(filter(lambda x: x > 0, cos_sims[:, c]))
        p_l = np.min(filter(lambda x: x > 0, cos_sims[:, c]))
      except ValueError:
        (p_h, p_l) = (0, 0)
      try:
        n_h = np.max(filter(lambda x: x < 0, cos_sims[:, c]))
        n_l = np.min(filter(lambda x: x < 0, cos_sims[:, c]))
      except ValueError:
        (n_h, n_l) = (0, 0)
      for r in range(np.size(cos_sims, axis=0)):
        if cos_sims[r, c] > 0:
          if p_h == p_l:
            cos_sims[r, c] = 0
          else:
            cos_sims[r, c] = (cos_sims[r, c]-p_l)/(p_h-p_l)
        else:
          if n_h == n_l:
            cos_sims[r, c] = 0
          else:
            cos_sims[r, c] = -1*(cos_sims[r, c]-n_h)/(n_l-n_h)
  elif is_normalization_cos_sim == 'shift_v4':
    h = np.max(cos_sims, axis=0)
    l = np.min(cos_sims, axis=0)
    for c in range(np.size(cos_sims, axis=1)):
      for r in range(np.size(cos_sims, axis=0)):
        if cos_sims[r, c] > (h[c]+l[c])/2:
          cos_sims[r, c] = 2*(cos_sims[r, c]-l[c])/(h[c]-l[c])-1
        else:
          cos_sims[r, c] = 0
  elif is_normalization_cos_sim == 'std_norm':
    for c in range(np.size(cos_sims, axis=1)):
      cos_sims[:, c] = (cos_sims[:, c]-mean(cos_sims[:, c]))/std(cos_sims[:, c])
  return cos_sims

def print_fea(filename, trans, targets, model, dim, stop_list, is_normalization_doc2vec, is_normalization_cos_sim, oov_dict=None, is_normalization_word=False, fix_stem=None):
  f_o = open(filename, 'w')
  f_cos_sims = []
  for trans_doc in trans:
    f_cos_sim = []
    for target_doc in targets:
      if oov_dict:
        f_cos_sim.append(compute_cos_sim_one_hot(trans_doc, target_doc, stop_list, oov_dict))
      else:
        f_cos_sim.append(compute_cos_sim(trans_doc, target_doc, model, dim, stop_list, is_normalization=is_normalization_doc2vec, is_normalization_word=is_normalization_word, fix_stem=fix_stem))
    f_cos_sims.append(f_cos_sim)
  f_cos_sims = normalize_cos_sim(f_cos_sims, is_normalization_cos_sim)

  for f_cos_sim in f_cos_sims:
    f_o.write('\t'.join([str(x) for x in f_cos_sim]) + '\n')
  f_o.close()

def assign_argv(argv, argv_index, assign_fn=lambda x: x):
  return (assign_fn(argv[argv_index]), argv_index+1)

def _main( ):
  task_type = sys.argv[1]
  corpus_name = sys.argv[2]
  model_name = sys.argv[3]
  lecture_list_name = sys.argv[4]
  fea_dir = sys.argv[5]
  dim = int(sys.argv[6])

  is_normalization_doc2vec = bool(int(sys.argv[7]))
  is_normalization_cos_sim = sys.argv[8]
  is_remove_stop = bool(int(sys.argv[9]))
  corpus_pos_name = sys.argv[10]
  pos_type = sys.argv[11]

  corpus = load_corpus(corpus_name)
  corpus_pos = load_corpus(corpus_pos_name)
  use_external_model = False
  fix_stem = False
  fix_math = False
  keyword_only = False
  is_normalization_word = False
  fix_stem_info = None
  is_pool = False
  if re.match('.*external_model', task_type):
    model = gensim.models.Word2Vec.load_word2vec_format(model_name, binary=True)
    use_external_model = True
    if 'fix_stem' in task_type:
      fix_stem = True
      if 'pool' in task_type:
        is_pool = True

      if 'fix_stem_math' in task_type:
        fix_math = True
      elif 'fix_stem_keyword' in task_type:
        keyword_only = True
  else:
    model = gensim.models.Word2Vec.load(model_name)

  stop_list = []
  if is_remove_stop or fix_stem:
    stop_list = load_courseware.load_stop_list()

  if re.match('.*textbook_lecture', task_type):
    argv_index = 12
    tx_list_name = sys.argv[12]
    seg_dir = sys.argv[13]
    tx_docs = get_tx_docs(corpus, corpus_pos, pos_type, tx_list_name)
    if use_external_model:

      if fix_stem:
        is_normalization_word = bool(int(sys.argv[14]))
        oov_refined = sys.argv[15]
        stem_to_raw = sys.argv[16]


        if fix_math:
          math_words = sys.argv[17]
        elif keyword_only:
          keyword_list = sys.argv[17]
      else:
        oov_dict_name = sys.argv[14]
  elif re.match('.*slides_trans', task_type):
    window_size = int(sys.argv[12])
    if use_external_model:
      if fix_stem:
        is_normalization_word = bool(int(sys.argv[13]))
        oov_refined = sys.argv[14]
        stem_to_raw = sys.argv[15]
        if fix_math:
          math_words = sys.argv[16]
        elif keyword_only:
          keyword_list = sys.argv[16]
      else:
        oov_dict_name = sys.argv[13]

  if use_external_model:
    if fix_stem:
      refined_oov = {}
      keywords = []
      for line in open(keyword_list):
        keywords += my_util.preprocess_content(line.strip(), stop_list, is_math=True, is_stemming=True)
      keywords = list(set(keywords))

      for line in open(oov_refined):
        items = line.strip().split('\t')
        refined_oov[items[0]] = (items[1] if len(items) > 1 else '')

      stem_map = {}
      oov_map = {}
      oov_size = 0
      for line in open(stem_to_raw):
        items = line.strip().split('\t')
        # remove words not in positive list (keyword list)
        if keywords and items[0] not in keywords:
          continue
        if re.match('n', items[2]):
          stem_map[items[0]] = re.split('\s+', items[1])[0]
        else:
          oov_map[items[0]] = oov_size
          oov_size += 1
      # add math words to oov
      if fix_math:
        for line in open(math_words):
          stemmed_word = (' '.join(my_util.preprocess_content(line.strip(), stop_list, is_math=True, is_stemming=True))).strip()
          # remove words not in positive list (keyword list)
          if stemmed_word and keywords and stemmed_word not in keywords:
            continue
          if stemmed_word and stemmed_word not in (stem_map.keys()+oov_map.keys()):
            oov_map[stemmed_word] = oov_size
            oov_size += 1
      fix_stem_info = (refined_oov, stem_map, oov_map)
    else:
      oov_dict = my_util.load_stopwords(oov_dict_name)

  for l_id in open(lecture_list_name):
    l_id = l_id.strip()
    trans_temp = get_docs(trans_doc_prefix, l_id, corpus, corpus_pos, pos_type)
    if re.match('.*slides_trans', task_type):
      trans = windowing(trans_temp, window_size)
      targets = get_docs(slides_doc_prefix, l_id, corpus, corpus_pos, pos_type)
    elif re.match('.*textbook_lecture', task_type):
      trans = window_by_seg(trans_temp, '%s/%s' % (seg_dir, l_id))
      targets = tx_docs

    print_fea('%s/%s' % (fea_dir, l_id), trans, targets, model, dim, stop_list, is_normalization_doc2vec, is_normalization_cos_sim, is_normalization_word=is_normalization_word, fix_stem=fix_stem_info)
    
    if use_external_model and not fix_stem:
      print_fea('%s/%s.oov' % (fea_dir, l_id), trans, targets, model, dim, stop_list, is_normalization_doc2vec, is_normalization_cos_sim, oov_dict=oov_dict)

if __name__ == '__main__':
  _main( )
