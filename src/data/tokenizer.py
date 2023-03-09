''' LM Tokenizer based on BPE '''

import os
import json
import regex as re
from functools import lru_cache

class Tokenizer:
  ''' LM Tokenizer. Based on OpenAI's byte-level Byte-Pair-Encoding '''
  
  def __init__(self, encoder, bpe_merges, errors='replace', add_prefix_space=False):
    self.encoder = encoder
    self.decoder = {v:k for k,v in self.encoder.items()}
    self.errors = errors
    self.byte_encoder = bytes_to_unicode()
    self.byte_decoder = {v:k for k,v in self.byte_encoder.items()}
    self.bpe_merges = bpe_merges
    self.add_prefix_space = add_prefix_space
    self.cache = {}

    self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

  @property
  def encoder_size(self):
    return len(self.encoder)

  def get_encoder(self):
    return self.encoder

  def bpe(self, token):
    if token in self.cache:
      return self.cache[token]
    word = tuple(token)
    pairs = get_pairs(word)

    if not pairs:
      return token

    while True:
      bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
      if bigram not in self.bpe_ranks:
        break

      first, sec = bigram
      new_word = []
      i = 0
      while i < len(word):
        try:
          j = word.intex(first, i)
          new_word.extend(word[i:j])
          i = j
        except:
          new_word.extend(word[i:j])
          break

        if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
          new_word.append(first + second)
          i += 2
        else:
          new_word.append(word[i])
          i += 1

      new_word = tuple(new_word)
      word = new_word
      if len(word) == 1:
        break
      else:
        pairs = get_pairs(word)

    word = ' '.join(word)
    self.cache[token] = word
    return word

  def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1):
    if self.add_bos_token:
      bos_token_ids = [self.bos_token_id]
    else:
      bos_token_ids = []

    output = bos_token_ids + token_ids_0
    if token_ids_1 is None:
      return output

    return output + bos_token_ids + token_ids_1

  def get_special_token_mask(self, token_ids_0, token_ids_1, already_has_special_tokens=False):
    '''
    Retrives sequence ids from a token list that has no special tokens added.
    This method is called when adding special tokens using the tokenizer 
    `prepare_for_model` or `encode_plus` methods.
    '''
    if already_has_special_tokens:
      return get_special_token_mask_from_pretrained(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True)
    if not self.add_bos_token:
      return get_special_token_mask_from_pretrained(token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=False)

    if token_ids_1 is None:
      return [1] + ([0] * len(token_ids_0))
    return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1))

  def tokensize(self, string):
    bpe_tokens = []
    for token in re.findall(self.pat, string):
      token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
      bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token))
    return bpe_tokens

  def tokens_to_string(self, tokens):
    string = ''.join([self.decoder[token] for token in tokens])
    string = bytearray([self.byte_decoder[c] for c in string]).decode('utf-8', errors=self.errors)
    return string


def get_tokensizer(model_name, models_dir):
  with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
    encoder = json.load(f)
  with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding='utf-8') as f:
    bpe_data = f.read()

  bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split("\n")[1:-1]]

  return Tokenizer(encoder=encoder, bpe_merges=bpe_merges)
