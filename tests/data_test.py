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
"""Tests for `../data.py`."""

# pylint: disable=invalid-name

import os

from absl.testing import absltest
from absl.testing import parameterized
import chex
import grain.python as grain
import jax
import jax.numpy as jnp
from nanodo import data
import numpy as np
import tensorflow_datasets as tfds


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


def _get_vocab_path():
  return os.path.join(
      os.path.dirname(__file__),
      "testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model",
  )


def _get_spm():
  return data.get_tokenizer(_get_vocab_path())


def _assert_grain_records(records: list[grain.Record], expected: np.ndarray):
  actual = [r.data for r in records]
  np.testing.assert_equal(actual, expected)


class DataTest(parameterized.TestCase):

  def test_py_batched_tfds(self):
    num_examples = 100
    with tfds.testing.mock_data(num_examples=num_examples):
      context_size = 512
      batch_size = 2
      ds = data.py_batched_tfds(
          tfds_name="lm1b",
          split="train",
          context_size=context_size,
          worker_count=0,
          vocab_path=_get_vocab_path(),
          batch_size=batch_size,
      )
      self.assertEqual((batch_size, context_size), next(iter(ds)).shape)

  def test_py_noam_pack(self):
    records = [[2, 3, 4, 1], [5, 6, 7, 8, 9, 10, 11, 1]]
    pyg_records = [
        grain.Record(metadata=grain.RecordMetadata(index=i), data=records[i])
        for i in range(len(records))
    ]
    npack = data._NoamPack(4)
    _assert_grain_records(
        list(npack(iter(pyg_records))),
        np.array([[2, 3, 4, 1], [5, 6, 7, 8], [9, 10, 11, 1]]),
    )

  def test_py_batched_tfds_noam_packed(self):
    with tfds.testing.mock_data():
      ds = data.py_batched_tfds(
          tfds_name="lm1b",
          split="train",
          context_size=1024,
          batch_size=2,
          worker_count=0,
          vocab_path=_get_vocab_path(),
          num_records=10,
          preprocessing=data.Preprocess.NOAM_PACKED,
      )
      it = iter(ds)
      b = next(it)
      self.assertEqual(b.shape, (2, 1024))
      self.assertEqual(np.sum(b == data.PAD_ID), 0)
      b = next(it)
      self.assertEqual(b.shape, (2, 1024))
      self.assertEqual(np.sum(b == data.PAD_ID), 0)

  def test_py_batched_tfds_padded(self):
    with tfds.testing.mock_data():
      ds = data.py_batched_tfds(
          tfds_name="lm1b",
          split="train",
          context_size=1024,
          batch_size=2,
          worker_count=0,
          vocab_path=_get_vocab_path(),
          num_records=10,
          preprocessing=data.Preprocess.PADDED,
      )
      it = iter(ds)
      b = next(it)
      self.assertEqual(b.shape, (2, 1024))
      self.assertGreater(np.sum(b == data.PAD_ID), 0)  # sanity check

  def test_py_tokenize(self):
    tok = data._SPTokenizer(_get_vocab_path())
    ids = data._py_tokenize({"text": "some text"}, spt=tok)
    self.assertNotEmpty(ids)
    ids = data._py_tokenize(
        {"text": "some text"}, spt=tok, pad_len=128, pad_id=0
    )
    self.assertLen(ids, 128)

  def test_get_in_out(self):
    rng = jax.random.PRNGKey(42)
    length = 256
    batch_size = 8
    in_BxL = jax.random.randint(
        rng, shape=(batch_size, length), minval=1, maxval=256
    )
    x_BxL, y_BxL, weights_BxL = data.get_in_out(in_BxL)
    self.assertEqual(x_BxL.shape, in_BxL.shape)
    self.assertEqual(y_BxL.shape, in_BxL.shape)
    self.assertEqual(weights_BxL.shape, in_BxL.shape)
    chex.assert_trees_all_equal(x_BxL, in_BxL)
    chex.assert_trees_all_equal(y_BxL[:, : length - 1], in_BxL[:, 1:length])
    chex.assert_trees_all_equal(
        y_BxL[:, length - 1],
        jnp.ones_like(y_BxL[:, length - 1]) * data.PAD_ID,
    )
    chex.assert_trees_all_equal(
        weights_BxL[:, : length - 1],
        jnp.ones_like(weights_BxL[:, : length - 1]),
    )
    chex.assert_trees_all_equal(
        weights_BxL[:, length - 1], jnp.zeros_like(weights_BxL[:, length - 1])
    )


if __name__ == "__main__":
  absltest.main()
