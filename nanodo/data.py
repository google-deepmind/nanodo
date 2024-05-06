# Copyright 2024 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Data pipeline."""

from collections.abc import Sequence
import dataclasses
import enum
import functools
from typing import Iterator

import grain.python as grain
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_text as tftxt

import sentencepiece as spm


PAD_ID = 0
EOS_ID = 1
BOS_ID = 2

### pure python helpers for use with grain ###


class Preprocess(enum.Enum):
  NOAM_PACKED = 1
  PADDED = 2


def py_batched_tfds(
    *,
    tfds_name: str,
    split: str,
    context_size: int,
    worker_count: int,
    vocab_path: str,
    batch_size: int,
    seed: int | None = 1234,
    num_epochs: int | None = None,
    num_records: int | None = None,
    preprocessing: Preprocess = Preprocess.NOAM_PACKED,
    worker_buffer_size: int = 2,
) -> Iterator[grain.Record]:
  """Returns iterator for regularly batched text examples."""
  datasource = tfds.data_source(tfds_name, split=split)
  index_sampler = grain.IndexSampler(
      num_records=num_records if num_records is not None else len(datasource),
      num_epochs=num_epochs,
      shard_options=grain.NoSharding(),
      shuffle=True,
      seed=seed,
  )
  spt = _SPTokenizer(vocab_path)

  pad_len = None if preprocessing == Preprocess.NOAM_PACKED else context_size
  pygrain_ops = [
      grain.MapOperation(
          map_function=functools.partial(
              _py_tokenize, spt=spt, pad_id=PAD_ID, pad_len=pad_len
          )
      )
  ]
  if preprocessing == Preprocess.NOAM_PACKED:
    pygrain_ops.append(_NoamPack(context_size=context_size))
  elif preprocessing == Preprocess.PADDED:
    pygrain_ops.append(grain.MapOperation(map_function=np.array))
  else:
    raise ValueError(f'Unknown preprocessing: {preprocessing}')
  pygrain_ops.append(grain.Batch(batch_size=batch_size, drop_remainder=True))
  batched_dataloader = grain.DataLoader(
      data_source=datasource,
      operations=pygrain_ops,
      sampler=index_sampler,
      worker_count=worker_count,
      worker_buffer_size=worker_buffer_size,
  )
  return iter(batched_dataloader)


def py_batched_tfds_for_eval(
    *,
    tfds_name: str,
    split: str,
    tokenizer: tftxt.SentencepieceTokenizer,
    batch_size: int,
    context_length: int,
) -> Iterator[np.ndarray]:
  """Returns iterator for batched eval datasets."""
  data_source = tfds.data_source(tfds_name, split=split)
  sampler = grain.IndexSampler(
      num_records=len(data_source),
      shard_options=grain.NoSharding(),
      shuffle=False,
  )
  operations = text_preprocess_batched_operations(
      tokenizer=tokenizer,
      context_length=context_length,
      batch_size=batch_size,
  )
  batched_dataloader = grain.DataLoader(
      data_source=data_source,
      operations=operations,
      sampler=sampler,
      # The tokenizer is not serializable, so we don't use multiprocessing. This
      # shouldn't be a problem for the eval with simple transformations.
      worker_count=0,
  )
  return iter(batched_dataloader)


def _pad_or_crop(x: np.ndarray, *, context_length: int) -> np.ndarray:
  """Either pads or crops the tokenized text input to context_length."""
  if len(x.shape) != 1:
    raise ValueError(f'Expected 1D array of tokenized text, got {x.shape}')
  if len(x) < context_length:
    pad_width = [(0, context_length - len(x))]
    return np.pad(x, pad_width, mode='constant', constant_values=PAD_ID)
  else:
    return x[:context_length]


def text_preprocess_batched_operations(
    tokenizer: tftxt.SentencepieceTokenizer,
    context_length: int,
    batch_size: int,
) -> Sequence[grain.Operation | grain.Transformation]:
  """Returns all operations to produce batched text tokens."""
  operations = []
  operations.append(
      grain.MapOperation(map_function=lambda x: tokenizer.tokenize(x['text']))
  )
  pad_or_crop = functools.partial(_pad_or_crop, context_length=context_length)
  operations.append(grain.MapOperation(map_function=pad_or_crop))
  operations.append(grain.Batch(batch_size=batch_size, drop_remainder=False))
  return operations


### TF / tf.data helpers (to be deprecated) ###


def get_tokenizer(model_path: str) -> tftxt.SentencepieceTokenizer:
  with tf.io.gfile.GFile(model_path, 'rb') as model_fp:
    spmodel = model_fp.read()
  sp_tokenizer = tftxt.SentencepieceTokenizer(
      model=spmodel, add_bos=True, add_eos=True, reverse=False
  )
  assert sp_tokenizer.string_to_id('</s>') == EOS_ID
  assert sp_tokenizer.string_to_id('<s>') == BOS_ID
  assert sp_tokenizer.string_to_id('<pad>') == PAD_ID
  return sp_tokenizer


def text_preprocess_packed(
    ds: tf.data.Dataset,
    tokenizer: tftxt.SentencepieceTokenizer,
    context_length: int,
    batch_size: int,
    shuffle: bool = True,
    buffer_size: int = 100_000,
) -> tf.data.Dataset:
  """Returns ds with ("noam") packed text tokens (training only)."""
  # Here packed means examples are all concatenated together separated by EOS,
  # and modeling examples have fixed length formed via sliding context
  # window. This is similar to PaLM packing.
  assert tokenizer.add_eos
  ds = ds.map(
      lambda x: tokenizer.tokenize(x['text']),
      num_parallel_calls=tf.data.AUTOTUNE,
  )
  if shuffle:
    ds = ds.shuffle(buffer_size, reshuffle_each_iteration=True)
  ds = _noam_pack(ds, context_length)
  ds = ds.batch(batch_size, drop_remainder=True).prefetch(tf.data.AUTOTUNE)
  return ds


def _get_py_tokenizer(path: str) -> spm.SentencePieceProcessor:
  sp = spm.SentencePieceProcessor()
  sp.Load(path)
  assert sp.bos_id() == BOS_ID
  assert sp.eos_id() == EOS_ID
  assert sp.pad_id() == PAD_ID
  return sp


# Need this because we can't pickle SentencePieceProcessor object
class _SPTokenizer:
  """Wrapper class for SentencePiece tokenizer."""

  def __init__(self, vocab_path):
    self._tokenizer = None
    self._vocab_path = vocab_path

  def get_tokenizer(self) -> spm.SentencePieceProcessor:
    if not self._tokenizer:
      self._tokenizer = _get_py_tokenizer(self._vocab_path)
    return self._tokenizer


def _py_tokenize(
    features: dict[str, str],
    spt: _SPTokenizer,
    pad_len: int | None = None,
    pad_id: int = PAD_ID,
) -> list[int]:
  """Tokenizes text into ids, optionally pads or truncates to pad_len."""
  text = features['text']
  ids = spt.get_tokenizer().EncodeAsIds(text)
  ids.insert(0, BOS_ID)
  ids.append(EOS_ID)
  if pad_len is not None:
    if len(ids) < pad_len:
      ids.extend([pad_id] * (pad_len - len(ids)))
    elif len(ids) > pad_len:
      ids = ids[:pad_len]
  return ids


@dataclasses.dataclass
class _NoamPack:
  """Pygrain operation for tokenizing and Noam packing text."""

  context_size: int

  def __call__(
      self, idseq_iterator: Iterator[grain.Record]
  ) -> Iterator[grain.Record]:
    packed_ids = []
    for input_record in idseq_iterator:
      start = 0
      while start < len(input_record.data):
        rem_data = input_record.data[start:]
        if len(packed_ids) + len(rem_data) < self.context_size:
          packed_ids.extend(rem_data)  # use rest of example, move-on
          break
        else:
          take = self.context_size - len(packed_ids)
          packed_ids.extend(rem_data[:take])
          last_record_key = input_record.metadata.remove_record_key()
          yield grain.Record(
              last_record_key, np.array(packed_ids, dtype=np.int32)
          )
          start += take
          packed_ids = []
          # Drop remainder for simplicity.
          # We lose the rest of the example on restore.


def _noam_pack(ds: tf.data.Dataset, context_length: int) -> tf.data.Dataset:
  """Concatenates all examples and generates sliding-window examples."""
  ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
  # Turn flat sequence into context-length examples
  ds = ds.batch(context_length, drop_remainder=True).prefetch(
      tf.data.AUTOTUNE
  )  # no need to pad
  return ds


def _multi_epoch_ds_gen(ds: tf.data.Dataset, epochs: int | None = None):
  # TF2 way to do multiple epochs rather than repeat.
  # Ensures examples don't mix between epochs.
  epoch_num = 0
  while epochs is None or epoch_num < epochs:
    for x in ds.as_numpy_iterator():
      yield x
    epoch_num += 1


# pylint: disable=invalid-name


def get_in_out(in_BxL: jax.Array) -> tuple[jax.Array, jax.Array, jax.Array]:
  # Assumes input of the form <BOS> <IDs> <EOS> for eval.
  x_BxL = in_BxL
  y_BxL = jnp.pad(
      in_BxL[:, 1:],
      ((0, 0), (0, 1)),
      mode='constant',
      constant_values=PAD_ID,
  )
  weights_BxL = jnp.where(y_BxL != PAD_ID, 1, 0).astype(jnp.float32)

  return x_BxL, y_BxL, weights_BxL
