"""Optimizer."""

# pylint: disable=g-import-not-at-top

import functools
from typing import Iterable, TYPE_CHECKING

import jax
import optax

if TYPE_CHECKING:
  import ml_collections


def get_optimizer(c: "ml_collections.ConfigDict") -> optax.MultiSteps:
  """Get optimizer."""
  optimizer = _get_base_optimizer(c)

  if c.get("optimizer", "adamw") == "adamw_mup":
    scale_dict = {"kernel": c.layerwise_lr_multiplier.kernel}
    optimizer = optax.chain(optimizer, _scale_by_dict(scale_dict))

  clip_by_global_norm = c.get("clip_by_global_norm", None)
  if clip_by_global_norm:
    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_by_global_norm), optimizer)

  # Multistep gradient accumulation
  optimizer = optax.MultiSteps(optimizer, c.get("grad_accumulation_steps", 1))

  return optimizer


def get_learning_rate_schedule(
    c: "ml_collections.ConfigDict",
) -> optax.Schedule:
  """Creates a learning rate schedule based on the config."""

  schedules = [
      optax.linear_schedule(
          init_value=c.init_learning_rate,
          end_value=c.peak_learning_rate,
          transition_steps=c.warmup_steps,
      )
  ]

  decay_type = c.get("decay_type", "cosine")

  if decay_type == "rsqrt":
    schedules.append(
        _rsqrt_schedule(
            init_value=c.peak_learning_rate,
            shift=1 + c.warmup_steps,
        )
    )

  elif decay_type == "cosine":
    decay_steps = c.get("decay_steps", c.num_train_steps - c.warmup_steps)
    schedules.append(
        optax.cosine_decay_schedule(
            init_value=c.peak_learning_rate,
            decay_steps=decay_steps,
            alpha=c.final_learning_rate / c.peak_learning_rate,
            exponent=1.0,
        )
    )

  elif decay_type == "linear":
    schedules.append(
        optax.linear_schedule(
            init_value=c.peak_learning_rate,
            end_value=c.final_learning_rate,
            transition_steps=c.num_train_steps - c.warmup_steps,
        )
    )

  elif decay_type == "constant_without_warmup":
    return optax.constant_schedule(value=c.peak_learning_rate)

  elif decay_type == "constant":
    schedules.append(optax.constant_schedule(value=c.peak_learning_rate))

  elif decay_type.startswith("constant_linear_decay_"):
    if decay_type.endswith("p"):
      percent_decay = float(decay_type.split("_")[-1].split("p")[0]) / 100
      if  percent_decay < 0 or percent_decay > 1:
        raise ValueError(f"Invalid decay % provided in {decay_type}")
      transition_steps = int(c.num_train_steps * percent_decay)
    else:
      decay_steps = int(decay_type.split("_")[-1])
      if decay_steps < 0 or decay_steps > c.num_train_steps:
        raise ValueError(f"Invalid decay steps provided in {decay_type}")
      transition_steps = decay_steps
    schedules += [
        optax.constant_schedule(value=c.peak_learning_rate),
        optax.linear_schedule(
            init_value=c.peak_learning_rate,
            end_value=c.final_learning_rate,
            transition_steps=transition_steps,
        )
    ]
    return optax.join_schedules(schedules, boundaries=[
        c.warmup_steps, c.num_train_steps - transition_steps])

  else:
    raise NotImplementedError(f"Unsupported decay type: {c.decay_type}")

  return optax.join_schedules(schedules, boundaries=[c.warmup_steps])


def _rsqrt_schedule(*, init_value: float, shift: int) -> optax.Schedule:
  """Constructs a schedule with reciprocal sqrt decay."""

  def schedule(count):
    return init_value * (count + shift) ** -0.5 * shift**0.5

  return schedule


def _params_mask(
    params: optax.Params, exclude_names: Iterable[str] = ("bias", "scale")
) -> optax.Params:
  """Generate boolean mask for params PyTree with `exclude_names` parameters."""
  def _check_key_contain_exclude_names(key_path):
    return any([
        x in "/".join([k.key for k in key_path if hasattr(k, "key")])
        for x in exclude_names
    ])

  # Mask should return True for parameters that does not match patterns inside
  # `exclude_names`.
  return jax.tree_util.tree_map_with_path(
      lambda key_path, _: not _check_key_contain_exclude_names(key_path), params
  )


def _get_base_optimizer(
    c: "ml_collections.ConfigDict",
) -> optax.GradientTransformation:
  """Get base optimizer."""
  learning_rate_fn = get_learning_rate_schedule(c)
  optimizer_type = c.get("optimizer", "adamw")
  weight_decay_exclusion_names = c.get("weight_decay_exclusion_names", [])

  if optimizer_type == "adafactor":
    base_optimizer = optax.adafactor(
        learning_rate_fn,
        multiply_by_parameter_scale=c.get(
            "multiply_by_parameter_scale", True),
        decay_rate=c.get("decay_rate", 0.8),
        momentum=c.get("momentum", None),
        factored=c.get("factored", True),
        eps=c.get("eps", 1e-30),
        weight_decay_rate=c.weight_decay,
        weight_decay_mask=functools.partial(
            _params_mask, exclude_names=weight_decay_exclusion_names))

  elif optimizer_type in ("adamw", "adamw_mup"):
    if c.get("independent_weight_decay", False):
      weight_decay = c.weight_decay / c.peak_learning_rate
    else:
      weight_decay = c.weight_decay
    base_optimizer = optax.adamw(
        learning_rate_fn,
        b1=c.get("b1", 0.9),
        b2=c.get("b2", 0.98),
        eps=c.get("eps", 1e-9),
        weight_decay=weight_decay,
        mask=functools.partial(
            _params_mask, exclude_names=weight_decay_exclusion_names),
    )

  else:
    raise ValueError(optimizer_type)

  return base_optimizer


def _scale_by_dict(
    scale_dict: dict[str, float]) -> optax.GradientTransformation:
  """Optax transform for performing layerwise learning rate rescaling.

  Args:
    scale_dict: a dictionary that determines which parameters to apply
    learning rate rescaling, e.g., {"kernel": 3.} means using a 3X learning rate
    for all parameters whose name contain "kernel".

  Returns:
    An Optax transform suitable for chaining (should be applied after the
    optimizer).
  """

  def init_fn(_):
    return optax.EmptyState()

  def update_fn(updates, state, params=None):
    del params

    def scale(keys, x):
      # Convert to str "module_name_1/module_name_2/.../kernel"
      str_keys = "/".join([k.key for k in keys if hasattr(k, "key")])
      for which_to_rescale, multiplier in scale_dict.items():
        if which_to_rescale in str_keys:
          return x * multiplier
      return x

    updates = jax.tree_util.tree_map_with_path(scale, updates)
    return updates, state

  return optax.GradientTransformation(init_fn, update_fn)
