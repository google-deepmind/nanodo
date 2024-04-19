"""Training loop."""

# pylint: disable=invalid-name,g-importing-member,g-import-not-at-top

import functools
import time
from typing import Any, Iterator, TYPE_CHECKING

from absl import logging
from clu import metric_writers
from clu import periodic_actions
from flax import linen as nn
from flax.training.train_state import TrainState
import grain.python as grain
import jax
from jax.experimental import mesh_utils
import jax.numpy as jnp
from jax.sharding import Mesh
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P
from nanodo import data
from nanodo import evaluate
from nanodo import loss as loss_lib
from nanodo import metrics as metrics_lib
from nanodo import model_factory
from nanodo import optimizer
import optax
import orbax.checkpoint as ocp
import tensorflow as tf


if TYPE_CHECKING:
  import ml_collections


PyTree = Any


def train_and_evaluate(c: "ml_collections.ConfigDict", workdir: str):
  """Train loop."""

  # Prevent tensorflow from fragmenting GPU memory.
  # See https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html.
  tf.config.set_visible_devices([], "GPU")

  mesh = Mesh(mesh_utils.create_device_mesh((jax.device_count(),)), ("data",))
  # For multistep gradient accumulator to simulate large batch sizes.
  grad_accumulation_steps = c.get("grad_accumulation_steps", 1)
  micro_batch_size, r = divmod(c.batch_size, grad_accumulation_steps)
  if grad_accumulation_steps > 1:
    logging.info("Gradient accumulation steps: %d", grad_accumulation_steps)
    logging.info(
        "Using total batch size = %d, micro batch size = %d",
        c.batch_size, micro_batch_size
    )
  if r:
    raise ValueError(
        "Batch size must be divisible by the gradient accumulation steps."
    )
  if micro_batch_size % jax.device_count() != 0:
    raise ValueError("Batch size must be divisible by the number of devices.")

  tf.io.gfile.makedirs(workdir)
  rng = jax.random.PRNGKey(c.seed)

  tokenizer = data.get_tokenizer(c.vocab_path)
  vocab_size = tokenizer.vocab_size()

  model, get_loss_fn = model_factory.get_model_and_loss(c, vocab_size)

  tic = time.time()
  shardings, state = _init_train_state(c, model, rng=rng, mesh=mesh)
  init_time = time.time() - tic
  logging.info("[TIMING]: get_new_state (jit init) time: %.2fs", init_time)

  train_iter = data.py_batched_tfds(
      tfds_name=c.ds_name,
      split="train",
      context_size=c.context_length,
      worker_count=c.pygrain_worker_count,
      vocab_path=c.vocab_path,
      batch_size=micro_batch_size,
      num_epochs=c.train_epochs,
      preprocessing=data.Preprocess.NOAM_PACKED,
      worker_buffer_size=c.pygrain_worker_buffer_size,
  )

  if c.checkpoint:
    ckpt_mngr = _get_ckpt_manager(workdir, c)
    if c.checkpoint_restore_dir is not None:
      logging.info("Restoring checkpoint from %s", c.checkpoint_restore_dir)
      ex_ckpt_mngr = _get_ckpt_manager(c.checkpoint_restore_dir, c)
      state, train_iter = _restore_ckpt(ex_ckpt_mngr, state, train_iter)

    elif ckpt_mngr.latest_step() is not None:
      latest_step = ckpt_mngr.latest_step()
      logging.info("Restoring checkpoint %d from %s", latest_step, workdir)
      state, train_iter = _restore_ckpt(ckpt_mngr, state, train_iter)

  trainer = Trainer(
      c=c,
      state=state,
      mesh=mesh,
      shardings=shardings,
      get_loss_fn=get_loss_fn,
  )

  # We may evaluate on larger context length than training to measure length
  # generalization.
  if c.context_length < c.eval_max_target_length:
    logging.warning(
        "context_length %d is smaller than eval_max_target_length %d",
        c.context_length,
        c.eval_max_target_length,
    )
  eval_batch_size = c.get("eval_batch_size", micro_batch_size)
  if eval_batch_size % jax.device_count() != 0:
    raise ValueError(
        "Eval Batch size must be divisible by the number of devices.")

  # TODO: Also use pygrain for eval data; remove tf.data.
  eval_ds = data.get_data(
      c.ds_name,
      c.eval_split,
      functools.partial(
          data.text_preprocess_batched,  # Different from training
          tokenizer=tokenizer,
          batch_size=eval_batch_size,
          context_length=c.eval_max_target_length,
          shuffle=False,
      ),
  )
  evaluator = evaluate.Evaluator(c, model, eval_ds, mesh, shardings)

  writer = metric_writers.create_default_writer(
      workdir,
      just_logging=jax.process_index() > 0,
      write_to_xm_measurements=c.get("write_to_xm_measurements", True),
      write_to_datatable=True
  )
  if trainer.step == 0:
    writer.write_hparams(dict(c))
    writer.write_scalars(trainer.step, {"jit_compilation_time": init_time})

  report_progress = periodic_actions.ReportProgress(
      num_train_steps=c.opt.num_train_steps,
      writer=writer,
      every_steps=c.write_perf_metrics_every_steps,
      every_secs=None,
  )

  if jax.process_index() == 0:
    hooks = [
        report_progress,
        periodic_actions.Profile(logdir=workdir, num_profile_steps=5),
    ]
  else:
    hooks = []

  with metric_writers.ensure_flushes(writer):
    with jax.spmd_mode("allow_all"):  # 🔥🔥 TODO: remove

      def _eval():
        with report_progress.timed("eval"):
          step = trainer.step
          eval_metrics = evaluator.eval(trainer.state.params)
          writer.write_scalars(step, eval_metrics)

      def _checkpoint():
        if c.checkpoint:
          step = trainer.step
          logging.info("Saving last checkpoint step %d", step)
          ckpt_mngr.save(step, {"state": trainer.state, "data": train_iter})

      def _process_metrics(step, microbatch_metrics):
        if microbatch_metrics and step % c.write_train_metrics_every_steps == 0:
          microbatch_metrics = [trainer.get_metrics(step, m)
                                for m in microbatch_metrics]
          metrics = metrics_lib.aggregate_microbatch_metrics(microbatch_metrics)
          writer.write_scalars(step, metrics)
          # Simple check for NaN/Inf for early termination.
          loss = metrics["train_loss"]
          if jnp.isnan(loss) or jnp.isinf(loss):
            # Terminate training. The next step has already been dispatched.
            logging.error(
                "[TRAINING ERROR] Nan/Inf encountered in training loop.\n "
                "Terminating training loop at step: %d", step + 1
            )
            _eval()
            raise FloatingPointError(step + 1, loss)

      pending_microbatch_metrics = []
      for step in range(trainer.step, c.opt.num_train_steps + 1):
        is_final_step = step == c.opt.num_train_steps
        if step % c.eval_every_steps == 0 or is_final_step:
          _eval()
        if step % c.checkpoint_every_steps == 0 or is_final_step:
          _checkpoint()

        for h in hooks:
          h(step)

        # Schedule this step's tasks.
        # Initialize metrics for microbatch accumulation.
        new_microbatch_metrics = []
        for _ in range(grad_accumulation_steps):
          try:
            in_BxL = next(train_iter)
          except StopIteration:
            logging.warning("Ran out of data at step %d. Stopping.", step)
            break
          # Async dispatch next step.
          new_microbatch_metrics.append(trainer.do_step(step, in_BxL))

        # Download to host and process the previous step's metrics after having
        # asynchronously dispatched the new step.
        _process_metrics(step - 1, pending_microbatch_metrics)
        pending_microbatch_metrics = new_microbatch_metrics
        logging.log_first_n(
            logging.INFO, "Finished training step %d.", 5, step - 1)
      # Download to host and process the final step's metrics.
      _process_metrics(c.opt.num_train_steps, pending_microbatch_metrics)

  if c.checkpoint:
    ckpt_mngr.close()


class Trainer:
  """Executes training step."""

  def __init__(
      self,
      c: "ml_collections.ConfigDict",
      state: TrainState,
      mesh: Mesh,
      shardings: PyTree,
      get_loss_fn: loss_lib.LossFnFactory = loss_lib.get_default_loss_fn,
  ):
    self.state = state
    self.init_metrics = None

    # In the jit call below, in_shardings and out_shardings specify the
    # shardings of the input and output of the jitted function.
    # There is just as many in_shardings as input arguments, and likewise for
    # outputs. "shardings" is the shardings of the state, P("data") denotes
    # that the argument is split along the data axis (in this case the
    # input data), and P() denotes that the result is replicated on each
    # device (in this case the train metrics).
    self.step_fn = jax.jit(
        functools.partial(
            _train_step,
            c=c,
            get_loss_fn=get_loss_fn,
            mesh=mesh,
        ),
        in_shardings=(
            shardings,
            NamedSharding(mesh, P()),
        ),
        out_shardings=(shardings, NamedSharding(mesh, P())),
        donate_argnames=("state", "in_BxL"),
    )

  @property
  def step(self) -> int:
    return int(self.state.step)

  def get_metrics(
      self, step: int, metrics: dict[str, float]
  ) -> dict[str, float]:
    # Grab the (possibly previous step's) metrics from device.
    metrics = jax.device_get(metrics)
    if step == 0:
      metrics |= self.init_metrics
    metrics["total_flops"] = self.init_metrics["flops_lowered"] * step
    return metrics

  def do_step(self, step: int, in_BxL: jax.Array) -> dict[str, float]:
    """Async dispatch one training step and return metrics."""
    # Note that the device may be busy with the previous step.
    # Avoid calling self.step as that would block until the device is ready.
    if step == 0 or self.init_metrics is None:
      self.init_metrics = metrics_lib.get_init_metrics(
          self.step_fn, self.state, in_BxL)

    self.state, metrics = self.step_fn(self.state, in_BxL)
    return metrics


def _train_step(
    state: TrainState,
    in_BxL: jax.Array,
    c: "ml_collections.ConfigDict",
    get_loss_fn: loss_lib.LossFnFactory = loss_lib.get_default_loss_fn,
    mesh: Mesh | None = None,
) -> tuple[TrainState, dict[str, float | jax.Array]]:
  """One forward/backward pass."""
  if mesh is not None:
    # B becomes local batch size here.
    in_BxL = jax.lax.with_sharding_constraint(
        in_BxL, NamedSharding(mesh, P("data"))
    )
  grad_fn = jax.value_and_grad(
      get_loss_fn(in_BxL, state.apply_fn, c), has_aux=True
  )
  (loss, aux_data), grads = grad_fn(state.params)

  # Access to optax updates.
  updates, new_opt_state = state.tx.update(grads, state.opt_state, state.params)
  new_params = optax.apply_updates(state.params, updates)
  new_state = state.replace(
      # Keep gradient_step as Trainer's step.
      step=state.opt_state.gradient_step + 1,  # pytype: disable=attribute-error
      params=new_params,
      opt_state=new_opt_state,
  )

  metrics = metrics_lib.get_metrics(aux_data, c, loss, state, grads, updates)
  return new_state, metrics


def _init_train_state(
    c: "ml_collections.ConfigDict",
    model: nn.Module,
    rng: jax.Array,
    mesh: Mesh,
) -> tuple[PyTree, TrainState]:
  """Creates a sharding and model state."""
  inputs = jax.ShapeDtypeStruct(shape=(1, c.context_length), dtype=jnp.int32)

  def init(rng, inputs):
    params = model.init(rng, inputs)
    return TrainState.create(
        apply_fn=model.apply,
        params=params["params"],
        tx=optimizer.get_optimizer(c.opt),
    )

  params = jax.eval_shape(init, rng, inputs)
  shardings = nn.get_sharding(params, mesh)
  state = jax.jit(init, out_shardings=shardings)(rng, inputs)
  return shardings, state


def _get_ckpt_manager(
    ckpt_dir: str,
    c: "ml_collections.ConfigDict",
) -> ocp.CheckpointManager:
  options = ocp.CheckpointManagerOptions(max_to_keep=c.max_to_keep)
  checkpointers = dict(
      state=ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler()),
      data=ocp.Checkpointer(grain.PyGrainCheckpointHandler()),  # pytype:disable=wrong-arg-types
  )
  return ocp.CheckpointManager(ckpt_dir, checkpointers, options)


def _restore_ckpt(
    ckpt_mngr: ocp.CheckpointManager,
    state: TrainState,
    train_iter: Iterator[jax.Array],
    step: int | None = None,
) -> tuple[TrainState, Iterator[jax.Array]]:
  """Restore a checkpoint."""
  restore_args = ocp.checkpoint_utils.construct_restore_args(state)
  restore_kwargs = {"state": {"restore_args": restore_args}}
  restored = ckpt_mngr.restore(
      ckpt_mngr.latest_step() if step is None else step,
      items={"state": state, "data": train_iter},
      restore_kwargs=restore_kwargs,
  )
  return restored["state"], restored["data"]
