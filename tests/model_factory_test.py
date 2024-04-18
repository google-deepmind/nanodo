"""Test experimental models and model factory."""

# pylint: disable=invalid-name

from absl.testing import absltest
from absl.testing import parameterized
import chex
import flax
import jax
import jax.numpy as jnp
from nanodo import model as default_model
from nanodo import model_factory
from nanodo.configs import default


jax.config.parse_flags_with_absl()
jax.config.update('jax_numpy_rank_promotion', 'raise')


class ModelTest(parameterized.TestCase):

  def _default_output(self, rng):
    """Setup an example input, output, params and config."""
    B, L = (2, 128)
    # default model
    cfg = default_model.DoConfig(D=16, H=4, L=L, N=4, V=256, F=4 * 4)
    m = default_model.TransformerDo(cfg)
    rng, spl = jax.random.split(rng)
    x_BxL = jax.random.randint(
        rng, minval=0, maxval=cfg.V, dtype=jnp.int32, shape=(B, L)
    )
    params = m.init(spl, x_BxL)
    default_model_out = m.apply(params, x_BxL)

    # equality test for experimental models
    # first set up config
    c = default.get_config()
    c.model_dim = cfg.D
    c.num_heads = cfg.H
    c.context_length = cfg.L
    c.num_layers = cfg.N
    c.vocab_size = cfg.V
    c.mlp_dim = cfg.F
    c.dtype = 'float32'

    return default_model_out, params, x_BxL, c

  def test_default_model(self):
    rng = jax.random.PRNGKey(42)
    default_model_out, params, x_BxL, c = self._default_output(rng)
    m, _ = model_factory.get_model_and_loss(c, c.vocab_size)
    self.assertTrue(jnp.allclose(m.apply(params, x_BxL), default_model_out))

if __name__ == '__main__':
  absltest.main()
