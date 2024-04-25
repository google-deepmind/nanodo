"""Tests for `../evaluate.py`."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
from nanodo import evaluate
from nanodo import metrics as metrics_lib
from nanodo import model


jax.config.parse_flags_with_absl()
jax.config.update("jax_numpy_rank_promotion", "raise")


class EvalTest(parameterized.TestCase):

  def test_eval_step(self):
    docfg = model.DoConfig(D=128, H=16, L=256, N=4, V=1024, F=4 * 4)
    m = model.TransformerDo(docfg)
    rng = jax.random.PRNGKey(42)
    _, init_rng = jax.random.split(rng)
    input_shape = (2, 256)
    x = jnp.ones(input_shape, dtype=jnp.int32)
    initial_variables = jax.jit(m.init)(init_rng, x)
    metrics = metrics_lib.Average()
    for _ in range(3):
      step_metrics = evaluate._eval_step(initial_variables["params"], x, m)
      metrics = metrics.merge(step_metrics)

    self.assertGreater(metrics.mean, 0)
    self.assertGreater(metrics.sem, 0)
    self.assertGreater(metrics.variance, 0)
    self.assertGreater(metrics.count, 0)


if __name__ == "__main__":
  absltest.main()
