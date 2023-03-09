# Create TF Records
import os
from pathlib import Path
import ftfy
import tensorflow as tf
from lm_dataformat import Reader
from tokenizers import Tokenizer
from transformers import GPT2TokenizerFast

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from itertools import repeat
import re
import logging

INPUT_DIR = ''
FILES_PER = None
NAME = None
OUTPUT_DIR = None
ENCODER_PATH = None
MIN_SIZE = 100
USE_FTFY = False
SEPERATOR = [50256]
CHUNK_SIZE = 2048
WRITE_DATASET_CONFIG = True
PROCESSES = 2

def _int64_feature(value):
  ''' Return an int64_list from a bool / enum / int / uint '''
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

def write_to_file(writer, data):
  ''' Writes data to .tfrecords file '''
  feature = {"text": _int64_feature(data)}
  tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
  writer.write(tf_example)

def get_tokenizer(args):
  if args.encoder_path is None:
    return GPTTokenizerFast.from_pretrained('gpt2')
  else:
    return Tokenizer.from_file(ENCODER_PATH)

def split_list(l, n):
  # splits list/string into n size chunks
  return [l[i:i + n] for i in range(0, len(l), n)]

def archive_to_tokens(f, encoder, args, prefix=[]):
  reader = Reader(f)
  for doc in reader.stream_data(threaded=False):
    if USE_FTFY:
      doc = ftfy.fix_text(doc, normalization="NFKC")
    doc = encoder.encode(doc) + SEPERATOR
    yield split_list(prefix + doc, CHUNK_SIZE)
    prefix = []

def write_files(files, files_per, output_dir, out_name, start_no, write_remainder=False, process_no=None):
  if files == None:
    return
  chunks = split_list(files, files_per)
  if not chunks:
    return

  if len(chunks[-1]) != files_per and not write_remainder:
    remainder = chunks.pop(-1)
  else:
    remainder = None
    files_per = len(chunks[-1])

  for files in chunks:
    fp = f"{output_dir}/{out_name}_{start_no}"
    if process_no is not None:
      fp += f"_{process_no}"
    fp += f"_{files_per}"
    fp += ".tfrecords"
    with tf.io.TFRecordWriter(fp) as writer:
      for f in files:
        write_to_file(writer, f)
    start_no += 1

  return start_no, remainder

def get_files(input_dir, filetypes=None):
  if filetypes == None:
    filetypes = ['json1.zst', '.txt', '.xz', '.tar.gz']
  files = [list(Path(input_dir).glob(f"*{ft}")) for ft in filetypes]
  flattened_list = [str(item) for sublist in files for item in sublist]
  if not flattened_list:
    raise Exception(f""" did not find any files at this path {input_dir}, please also ensure your files are in format {filetypes}""")
  return flattened_list

def read_checkpoint(checkpoint_path, resume_from_checkpoint=True):
  if resume_from_checkpoint and os.path.isfile(checkpoint_path):
    try:
      resume_files_processed, tfrecord_count = [int(i) for i in open(checkpoint_path, "r").read().split(", ")]
      print(f"\nResuming from tfrecord no. {tfrecord_count} / file no. {resume_files_processed}")
      return resume_files_processed, tfrecord_count
    except:
      pass

  return 0, 0

def create_tfrecords(params, writer_remainder=True, write_every_n_files=1, save_checkpoints=False, resume_from_checkpoint=False, display_pbar=False):
  