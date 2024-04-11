"""Tests for `../train.py`."""

# pylint: disable=invalid-name,g-importing-member,g-import-not-at-top

import os

from typing import TYPE_CHECKING

from absl import logging
from absl.testing import parameterized
import chex
from flax.training.train_state import TrainState
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh
from nanodo import data
from nanodo import loss as loss_lib
from nanodo import model
from nanodo import optimizer as opt
from nanodo import train
from nanodo.google.configs import default
import tensorflow_datasets as tfds

from absl.testing import absltest

if TYPE_CHECKING:
  import ml_collections


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


_VOCAB_PATH = "testdata/sentencepiece_cc_all.32000.100extra-sentencepiece.model"


def _get_config() -> "ml_collections.ConfigDict":
  """Get the default hyperparameter configuration."""
  c = default.get_config()
  c.vocab_path = os.path.join(os.path.dirname(__file__), _VOCAB_PATH)

  c.opt.peak_learning_rate = 0.01
  c.opt.init_learning_rate = 0.001
  c.opt.final_learning_rate = 0.0001
  c.opt.num_train_steps = 1
  c.opt.warmup_steps = 10
  c.opt.decay_steps = 100

  c.opt.b1 = 0.9
  c.opt.b2 = 0.98
  c.opt.eps = 1e-9
  c.opt.weight_decay = 0.1

  c.batch_size = 2
  c.eval_steps = 1
  c.checkpoint_every_steps = 1
  c.pygrain_worker_count = 2
  c.vocab_size = 32

  c.context_length = 64
  c.model_dim = 32
  c.mlp_dim = 128
  c.num_layers = 2
  c.num_heads = 4

  return c


class TrainTest(parameterized.TestCase):

  @parameterized.parameters(True, False)
  def test_trainer(self, fsdp_enabled: bool = False):
    c = _get_config()
    cfg = model.DoConfig(
        D=16, H=4, L=16, N=2, V=100, F=2, fsdp_enabled=fsdp_enabled)
    m = model.TransformerDo(cfg)
    rng = jax.random.PRNGKey(42)
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
    shardings, state = train._init_train_state(c, m, rng, mesh=mesh)
    t = train.Trainer(c, state, mesh, shardings, loss_lib.get_default_loss_fn)

    self.assertEqual(t.step, 0)

  def test_train_step(self):
    c = _get_config()
    docfg = model.DoConfig(
        D=c.model_dim,
        H=c.num_heads,
        L=c.context_length,
        N=c.num_layers,
        V=c.vocab_size,
        F=c.mlp_dim,
    )
    m = model.TransformerDo(docfg)
    init_rng, data_rng = jax.random.split(jax.random.PRNGKey(42))
    in_BxL = jax.random.randint(
        data_rng,
        (2, c.context_length),
        0,
        c.vocab_size,
        jnp.int32,
    )
    params = jax.jit(m.init)(init_rng, in_BxL)
    optimizer = opt.get_optimizer(c.opt)
    state = TrainState.create(
        apply_fn=m.apply, params=params["params"],
        tx=optimizer,
    )

    self.assertEqual(state.step, 0)
    state, metrics = train._train_step(
        state, in_BxL, c, loss_lib.get_default_loss_fn
    )
    self.assertEqual(state.step, 1)

    reference = {
        "__train_loss": 3.945808,
        "train_loss": 3.945808,
        "train_ntokens": 124,

        "grads/all/rms": 0.01341779,
        "grads/all/mean": 3.026796e-05,
        "grads/all/std": 0.01341776,

        "updates/all/rms": 0.99979043,
        "updates/all/mean": -0.00446726,
        "updates/all/std": 0.9997804,

        "params/all/rms": 0.16059065,
        "params/all/mean": 0.00510256,
        "params/all/std": 0.16050959,

        "learning_rate": c.opt.init_learning_rate,

        "train_fraction": 0,
        "train_tokens_seen": 0,
    }
    metrics_subset = {k: v for k, v in metrics.items() if k in reference}
    print(metrics_subset)
    warning = (
        " metric after doing a single gradient step of the default model "
        "have changed. If you did not intend to change the model's behavior "
        "(e.g. refactoring), this may indicate a bug. If the change is "
        "expected (e.g. change in parameterization, default hyperparameters, "
        "random seed, renaming or removing metrics, etc.), then please update "
        "the `reference` dictionary above with the new expected values."
    )

    jax.tree_util.tree_map_with_path(
        lambda k, x, y: self.assertAlmostEqual(
            x,
            y,
            places=2,
            msg=jax.tree_util.keystr(k) + warning),
        reference,
        metrics_subset,
    )

  @parameterized.parameters(data.Preprocess.NOAM_PACKED, data.Preprocess.PADDED)
  def test_train_and_evaluate(self, preprocessing):
    if jax.default_backend() == "cpu":
      self.skipTest("Skipping slow tests on CPU.")
    c = _get_config()
    c.checkpoint = True
    cfg = model.DoConfig(
        D=c.model_dim,
        H=c.num_heads,
        L=c.context_length,
        N=c.num_layers,
        V=c.vocab_size,
        F=c.mlp_dim,
    )
    m = model.TransformerDo(cfg)
    rng = jax.random.PRNGKey(42)
    mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
    _, state = train._init_train_state(c, m, rng, mesh)
    ckpt_dir = self.create_tempdir().full_path
    with tfds.testing.mock_data(num_examples=100):
      train_iter = data.py_batched_tfds(
          tfds_name=c.ds_name,
          split="train",
          context_size=c.context_length,
          worker_count=c.pygrain_worker_count,
          vocab_path=c.vocab_path,
          batch_size=c.batch_size,
          num_epochs=c.train_epochs,
          preprocessing=preprocessing,
      )
      train.train_and_evaluate(c, ckpt_dir)

      ckpt_mngr = train._get_ckpt_manager(ckpt_dir, c)

      self.assertEqual(ckpt_mngr.latest_step(), 1)
      # TODO: Check data iterator.
      restored_state, _ = train._restore_ckpt(ckpt_mngr, state, train_iter)
      self.assertEqual(restored_state.step, 1)

      logging.info("Trigger restore, check step is updated.")
      c.opt.num_train_steps = 2
      train.train_and_evaluate(c, ckpt_dir)
      ckpt_mngr = train._get_ckpt_manager(ckpt_dir, c)
      self.assertEqual(ckpt_mngr.latest_step(), 2)
      restored_state, _ = train._restore_ckpt(ckpt_mngr, state, train_iter)
      self.assertEqual(restored_state.step, 2)

  def test_train_step_remat(self):
    c = _get_config()
    docfg = model.DoConfig(
        D=128, H=16, L=256, N=4, V=1024, F=4 * 4, remat=False)
    m = model.TransformerDo(docfg)
    docfg_remat = model.DoConfig(
        D=128, H=16, L=256, N=4, V=1024, F=4 * 4, remat=True
    )
    m_remat = model.TransformerDo(docfg_remat)

    init_rng = jax.random.PRNGKey(42)
    in_BxL = jax.random.categorical(init_rng, jnp.ones((16, 256, 1024)))
    initial_variables = jax.jit(m.init)(
        init_rng,
        in_BxL,
    )
    optimizer = opt.get_optimizer(c.opt)
    state = TrainState.create(
        apply_fn=m.apply, params=initial_variables["params"],
        tx=optimizer,
    )

    state_remat = TrainState.create(
        apply_fn=m_remat.apply, params=initial_variables["params"],
        tx=optimizer,
    )
    new_state, metrics = train._train_step(state, in_BxL, c)
    new_state_remat, metrics_remat = train._train_step(state_remat, in_BxL, c)

    # Check metrics agreement
    chex.assert_trees_all_close(metrics, metrics_remat, rtol=1e-2, atol=1e-1)
    # Check updated params agreement
    chex.assert_trees_all_close(
        new_state.params, new_state_remat.params, rtol=1e-2, atol=1e-1
    )
    # Check optimizer state agreement
    chex.assert_trees_all_close(
        new_state_remat.opt_state,
        new_state_remat.opt_state,
        rtol=1e-2,
        atol=1e-1,
    )


if __name__ == "__main__":
  absltest.main()
