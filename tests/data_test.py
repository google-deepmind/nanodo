"""Tests for `../data.py`."""

# pylint: disable=invalid-name,g-import-not-at-top

import os
from typing import TYPE_CHECKING

from absl.testing import absltest
from absl.testing import parameterized
import chex
import grain.python as grain
import jax
import jax.numpy as jnp
from nanodo import data
from nanodo.google.configs import default
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

if TYPE_CHECKING:
  import ml_collections


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


def _get_vocab_path():
  return  os.path.join(
      os.path.dirname(__file__),
      "testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model",
  )


def _get_spm():
  return data.get_tokenizer(_get_vocab_path())


def _get_config() -> "ml_collections.ConfigDict":
  """Get the default hyperparameter configuration."""
  c = default.get_config()
  c.opt.num_train_steps = 1
  c.batch_size = 2
  c.eval_steps = 1
  return c


def _assert_grain_records(records: list[grain.Record], expected: np.ndarray):
  actual = [r.data for r in records]
  np.testing.assert_equal(actual, expected)


class DataTest(parameterized.TestCase):

  def test_get_data(self):
    with tfds.testing.mock_data(num_examples=5):
      ds = data.get_data("mnist", split="train")
    self.assertIsNotNone(ds)

  def test_text_preprocess_batched(self):
    with tfds.testing.mock_data(num_examples=100):
      ds = tfds.load("lm1b", split="train")
      c = _get_config()
      ds = data.text_preprocess_batched(
          ds,
          _get_spm(),
          context_length=c.eval_max_target_length,
          batch_size=c.batch_size,
      )
      all_ds = list(ds)
      self.assertLen(all_ds, 50)
      self.assertEqual((2, c.eval_max_target_length), all_ds[0].shape)

  def test_noam_pack(self):
    ds = tf.data.experimental.from_list(
        [[2, 3, 4, 1], [5, 6, 7, 8, 9, 10, 11, 1]]
    )
    ds4 = data._noam_pack(ds, 4)
    np.testing.assert_equal(
        list(ds4.as_numpy_iterator()),
        [[2, 3, 4, 1], [5, 6, 7, 8], [9, 10, 11, 1]],
    )

    ds = tf.data.experimental.from_list(
        [[2, 3, 4, 1], [12, 13, 14, 1], [5, 6, 7, 8, 9, 10, 11, 1]]
    )
    ds8 = data._noam_pack(ds, 8)
    np.testing.assert_equal(
        list(ds8.as_numpy_iterator()),
        [[2, 3, 4, 1, 12, 13, 14, 1], [5, 6, 7, 8, 9, 10, 11, 1]],
    )

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

  def test_multi_epoch_gen(self):
    ds = tf.data.Dataset.from_tensor_slices([1, 2, 3])
    self.assertLen(list(data._multi_epoch_ds_gen(ds, 1)), 3)
    self.assertLen(list(data._multi_epoch_ds_gen(ds, 2)), 6)

  def test_py_batched_tfds_noam_packed(self):
    with tfds.testing.mock_data():
      it = data.py_batched_tfds(
          tfds_name="lm1b",
          split="train",
          context_size=1024,
          batch_size=2,
          worker_count=0,
          vocab_path=_get_vocab_path(),
          num_records=10,
          preprocessing=data.Preprocess.NOAM_PACKED,
      )
      b = next(it)
      self.assertEqual(b.shape, (2, 1024))
      self.assertEqual(np.sum(b == data.PAD_ID), 0)
      b = next(it)
      self.assertEqual(b.shape, (2, 1024))
      self.assertEqual(np.sum(b == data.PAD_ID), 0)

  def test_py_batched_tfds_padded(self):
    with tfds.testing.mock_data():
      it = data.py_batched_tfds(
          tfds_name="lm1b",
          split="train",
          context_size=1024,
          batch_size=2,
          worker_count=0,
          vocab_path=_get_vocab_path(),
          num_records=10,
          preprocessing=data.Preprocess.PADDED,
      )
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
