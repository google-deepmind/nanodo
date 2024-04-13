"""Tests for `../metrics.py`."""

# pylint: disable=invalid-name,g-importing-member,g-import-not-at-top

from typing import TYPE_CHECKING

from absl.testing import absltest
from absl.testing import parameterized
import chex
from flax.training.train_state import TrainState
import jax
from jax import random
import jax.numpy as jnp
from nanodo import loss as loss_lib
from nanodo import metrics as metrics_lib
from nanodo import model
from nanodo import optimizer as opt
from nanodo import train


if TYPE_CHECKING:
  import ml_collections


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


def _get_config() -> "ml_collections.ConfigDict":
  """Get the default hyperparameter configuration."""
  c = default.get_config()

  c.batch_size = 2
  c.eval_steps = 1
  c.vocab_size = 32

  c.context_length = 64
  c.model_dim = 32
  c.mlp_dim = 128
  c.num_layers = 2
  c.num_heads = 4

  return c


class MetricsTest(parameterized.TestCase):

  def test_aggregate_microbatch_metrics(self):
    c = _get_config()
    docfg = model.DoConfig(D=128, H=16, L=256, N=4, V=1024, F=4 * 4)
    m = model.TransformerDo(docfg)
    init_rng = jax.random.PRNGKey(42)
    in_BxL = jax.random.categorical(init_rng, jnp.ones((16, 256, 1024)))

    initial_variables = jax.jit(m.init)(
        init_rng,
        in_BxL,
    )

    state_single = TrainState.create(
        apply_fn=m.apply, params=initial_variables["params"],
        tx=opt.get_optimizer(c.opt),
    )

    state_single, metrics_single = train._train_step(
        state_single, in_BxL, c, loss_lib.get_default_loss_fn
    )
    metrics_single = metrics_lib.aggregate_microbatch_metrics([metrics_single])

    grad_accumulation_steps = 4
    c.opt.grad_accumulation_steps = grad_accumulation_steps

    state_multistep = TrainState.create(
        apply_fn=m.apply, params=initial_variables["params"],
        tx=opt.get_optimizer(c.opt),
    )

    microbatch_train_metrics = []
    for sub_in_BxL in jnp.array_split(in_BxL, grad_accumulation_steps, axis=0):
      state_multistep, metrics = train._train_step(
          state_multistep, sub_in_BxL, c, loss_lib.get_default_loss_fn
      )
      microbatch_train_metrics.append(metrics)
    metrics_multistep = metrics_lib.aggregate_microbatch_metrics(
        microbatch_train_metrics)

    self.assertEqual(state_single.step, state_multistep.step)
    # Check metrics agreement
    chex.assert_trees_all_close(
        metrics_single, metrics_multistep, rtol=1e-2, atol=1e-1)
    # Check updated params agreement
    chex.assert_trees_all_close(
        state_single.params, state_multistep.params, rtol=1e-2, atol=1e-1)
    # Check optimizer state agreement
    chex.assert_trees_all_close(
        state_single.opt_state, state_multistep.opt_state, rtol=1e-2, atol=1e-1)

  @parameterized.product(
      x_shape=(
          (3, 10),
          (5, 10)
      ),
      y_shape=(
          (10, 2),
          (10, 7)
      ),
      fn_and_cost_fn=(
          (lambda x, y: x @ y,
           # Cost of (n, m) @ (m, k) is (n m k) * 2 (multiplies and additions).
           # Memory is storing reading two matrices + writing the result.
           # Each 32-bit entry is 4 bytes.
           lambda x, y: (  # pylint: disable=g-long-lambda
               2 * x[0] * x[1] * y[1],
               4 * ((x[0] * x[1]) + (y[0] * y[1]) + (x[0] * y[1])),
           )
           ),
          (lambda x, y: jnp.maximum(x @ y, 0.),
           # Cost of matmul + cost of ReLU = size of the output matrix.
           # Memory is store all inputs (incl a scalar) + 5 * output size:
           # 1) write and read the output of matmul
           # 2) write and read a matrix of zeros
           # 3) write a matrix of results
           lambda x, y: (  # pylint: disable=g-long-lambda
               2 * x[0] * x[1] * y[1] + x[0] * y[1],
               4 * ((x[0] * x[1]) + (y[0] * y[1]) + 5 * (x[0] * y[1]) + 1),
           )
           ),
      )
  )
  def test_get_flops(
      self,
      x_shape,
      y_shape,
      fn_and_cost_fn
  ):
    x = random.normal(random.PRNGKey(1), x_shape)
    y = random.normal(random.PRNGKey(2), y_shape)

    fn, cost_fn = fn_and_cost_fn
    costs = metrics_lib._get_costs(fn, x, y)
    flops_ref, memory_ref = cost_fn(x.shape, y.shape)
    self.assertEqual(costs["flops_lowered"], flops_ref)

    if jax.config.read("jax_enable_x64"):
      memory_ref *= 2

    if jax.default_backend() != "tpu":
      # TODO: understand TPU memory cost.
      self.assertEqual(costs["bytes accessed_lowered"], memory_ref)

    if jax.default_backend() != "gpu":
      # TODO: revisit after https://github.com/google/jax/issues/16008).
      self.assertEqual(costs["flops_compiled"], flops_ref)
      # TODO: understand compiled memory cost.
      # self.assertEqual(costs["bytes accessed_compiled"], memory_ref)

  def test_gaussian(self):
    rng = jax.random.PRNGKey(0)
    data = jax.random.normal(rng, (100,))
    average = None

    for x in data:
      update = metrics_lib.Average.from_array(x)
      average = update if average is None else average.merge(update)

    self.assertIsNotNone(average)

    self.assertAlmostEqual(
        average.mean,
        0.0,
        delta=3 * average.sem,
    )

    full_average = metrics_lib.Average.from_array(data)

    self.assertAlmostEqual(
        full_average.mean,
        0.0,
        delta=3 * full_average.sem,
    )

    # agreement
    self.assertAlmostEqual(
        average.mean,
        full_average.mean,
        delta=(average.sem ** 2 + full_average.sem ** 2) ** 0.5,
    )


if __name__ == "__main__":
  absltest.main()
