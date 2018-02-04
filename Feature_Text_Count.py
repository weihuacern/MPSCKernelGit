
"""
__file__

  Feature_Text_Count.py

__description__

  This file generates the following features for each run and fold, and for the entire training and testing set.

    1. Basic Counting Features
            
      1. Count of n-gram in name/item_description

      2. Count & Ratio of Digit in name/item_description

      3. Count & Ratio of Unique n-gram in name/item_description

    2. Intersect Counting Features

      1. Count & Ratio of a's n-gram in b's n-gram

    3. Intersect Position Features

      1. Statistics of Positions of a's n-gram in b's n-gram

      2. Statistics of Normalized Positions of a's n-gram in b's n-gram

__author__

  Hua Wei < hua.wei@cern.ch >

"""

import re
import sys
import numpy as np
import pandas as pd
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams


def get_position_list(target, obs):
  """
  Get the list of positions of obs in target
  """
  pos_of_obs_in_target = [0]
  if len(obs) != 0:
    pos_of_obs_in_target = [j for j,w in enumerate(obs, start=1) if w in target]
    if len(pos_of_obs_in_target) == 0:
      pos_of_obs_in_target = [0]
  return pos_of_obs_in_target


#####################
## Preprocess data ##
#####################
token_pattern = "[a-zA-Z0-9'`]+"
#token_pattern = r"(?u)\b\w\w+\b"
def TextPreProcessing(line,
                      token_pattern=token_pattern):
  ## tokenize
  tokenizer = RegexpTokenizer(token_pattern)
  tokens = tokenizer.tokenize( line.lower() )
  return tokens
'''
def Myngrams(text, min=1, max=4):
  s = []
  for n in range(min, max+1):
    for ngram in ngrams(text, n):
      s.append(' '.join(str(i) for i in ngram))

  #s = list(set(s)) # unique string in the list
  return s
'''
def Myngrams(text, nfold):
  s = []
  for ngram in ngrams(text, nfold):
    s.append(' '.join(str(i) for i in ngram))

  #s = list(set(s)) # unique string in the list
  return s

def DumpTextCountFeatures(df):
  ## unigram
  print("Generate one gram...")
  df["title_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_title"]), axis=1))
  df["description_unigram"] = list(df.apply(lambda x: preprocess_data(x["product_description"]), axis=1))
  ## bigram
  print("Generate two gram...")
  join_str = "_"
  df["title_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["title_unigram"], join_str), axis=1))
  df["description_bigram"] = list(df.apply(lambda x: ngram.getBigram(x["description_unigram"], join_str), axis=1))
  ## trigram
  print("Generate three gram...")
  join_str = "_"
  df["title_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["title_unigram"], join_str), axis=1))
  df["description_trigram"] = list(df.apply(lambda x: ngram.getTrigram(x["description_unigram"], join_str), axis=1))

  ################################
  ## word count and digit count ##
  ################################
  print("generate basic word count features")
  '''
  feat_names = ["name", "item_description"]
  grams = ["unigram", "bigram", "trigram"]
  count_digit = lambda x: sum([1. for w in x if w.isdigit()])
  for feat_name in feat_names:
    for gram in grams:
      ## word count
      df["count_of_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(x[feat_name+"_"+gram]), axis=1))
      df["count_of_unique_%s_%s"%(feat_name,gram)] = list(df.apply(lambda x: len(set(x[feat_name+"_"+gram])), axis=1))
      df["ratio_of_unique_%s_%s"%(feat_name,gram)] = map(try_divide, df["count_of_unique_%s_%s"%(feat_name,gram)], df["count_of_%s_%s"%(feat_name,gram)])

    ## digit count
    df["count_of_digit_in_%s"%feat_name] = list(df.apply(lambda x: count_digit(x[feat_name+"_unigram"]), axis=1))
    df["ratio_of_digit_in_%s"%feat_name] = map(try_divide, df["count_of_digit_in_%s"%feat_name], df["count_of_%s_unigram"%(feat_name)])

  ## description missing indicator
  df["description_missing"] = list(df.apply(lambda x: int(x["description_unigram"] == ""), axis=1))
  '''
  ##########################
  ## intersect word count ##
  ##########################
  print("generate intersect word count features")
  #### unigram
  '''
  for gram in grams:
    for obs_name in feat_names:
      for target_name in feat_names:
        if target_name != obs_name:
          ## query
          df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = list(df.apply(lambda x: sum([1. for w in x[obs_name+"_"+gram] if w in set(x[target_name+"_"+gram])]), axis=1))
          df["ratio_of_%s_%s_in_%s"%(obs_name,gram,target_name)] = map(try_divide, df["count_of_%s_%s_in_%s"%(obs_name,gram,target_name)], df["count_of_%s_%s"%(obs_name,gram)])

    ## some other feat
    df["title_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_title_%s_in_query"%gram], df["count_of_query_%s"%gram])
    df["title_%s_in_query_div_query_%s_in_title"%(gram,gram)] = map(try_divide, df["count_of_title_%s_in_query"%gram], df["count_of_query_%s_in_title"%gram])
    df["description_%s_in_query_div_query_%s"%(gram,gram)] = map(try_divide, df["count_of_description_%s_in_query"%gram], df["count_of_query_%s"%gram])
    df["description_%s_in_query_div_query_%s_in_description"%(gram,gram)] = map(try_divide, df["count_of_description_%s_in_query"%gram], df["count_of_query_%s_in_description"%gram])
  '''

  ##################################
  ## intersect word position feat ##
  ##################################
  print("generate intersect word position features")
  '''
  for gram in grams:
    for target_name in feat_names:
      for obs_name in feat_names:
        if target_name != obs_name:
          pos = list(df.apply(lambda x: get_position_list(x[target_name+"_"+gram], obs=x[obs_name+"_"+gram]), axis=1))
          ## stats feat on pos
          df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(np.min, pos)
          df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(np.mean, pos)
          df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(np.median, pos)
          df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(np.max, pos)
          df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(np.std, pos)
          ## stats feat on normalized_pos
          df["normalized_pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_min" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
          df["normalized_pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_mean" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
          df["normalized_pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_median" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
          df["normalized_pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_max" % (obs_name, gram, target_name)], df["count_of_%s_%s" % (obs_name, gram)])
          df["normalized_pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] = map(try_divide, df["pos_of_%s_%s_in_%s_std" % (obs_name, gram, target_name)] , df["count_of_%s_%s" % (obs_name, gram)])
  '''
  return

if __name__ == "__main__":

  ###############
  ## Load Data ##
  ###############
  ## load data
  df_train = pd.read_csv('../data/train.tsv', sep='\t')
  df_train.set_index('train_id')

  '''
    with open(config.processed_train_data_path, "rb") as f:
        dfTrain = cPickle.load(f)
    with open(config.processed_test_data_path, "rb") as f:
        dfTest = cPickle.load(f)
    ## load pre-defined stratified k-fold index
    with open("%s/stratifiedKFold.%s.pkl" % (config.data_folder, config.stratified_label), "rb") as f:
            skf = cPickle.load(f)

    ## file to save feat names
    feat_name_file = "%s/counting.feat_name" % config.feat_folder
  '''

  #######################
  ## Generate Features ##
  #######################
  print("==================================================")
  print("Generate counting features from text...")


  '''
    extract_feat(dfTrain)
    feat_names = [
        name for name in dfTrain.columns \
            if "count" in name \
            or "ratio" in name \
            or "div" in name \
            or "pos_of" in name
    ]
    feat_names.append("description_missing")


    print("For cross-validation...")
    for run in range(config.n_runs):
        ## use 33% for training and 67 % for validation
        ## so we switch trainInd and validInd
        for fold, (validInd, trainInd) in enumerate(skf[run]):
            print("Run: %d, Fold: %d" % (run+1, fold+1))
            path = "%s/Run%d/Fold%d" % (config.feat_folder, run+1, fold+1)
              
            #########################
            ## get word count feat ##
            #########################
            for feat_name in feat_names:
                X_train = dfTrain[feat_name].values[trainInd]
                X_valid = dfTrain[feat_name].values[validInd]
                with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    cPickle.dump(X_train, f, -1)
                with open("%s/valid.%s.feat.pkl" % (path, feat_name), "wb") as f:
                    cPickle.dump(X_valid, f, -1)
    print("Done.")


    print("For training and testing...")
    path = "%s/All" % config.feat_folder
    ## use full version for X_train
    extract_feat(dfTest)
    for feat_name in feat_names:
        X_train = dfTrain[feat_name].values
        X_test = dfTest[feat_name].values
        with open("%s/train.%s.feat.pkl" % (path, feat_name), "wb") as f:
            cPickle.dump(X_train, f, -1)
        with open("%s/test.%s.feat.pkl" % (path, feat_name), "wb") as f:
            cPickle.dump(X_test, f, -1)
            
    ## save feat names
    print("Feature names are stored in %s" % feat_name_file)
    ## dump feat name
    dump_feat_name(feat_names, feat_name_file)
  '''
  print("All Done.")
