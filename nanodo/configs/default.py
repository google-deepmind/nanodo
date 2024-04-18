"""Default Hyperparameter configuration.

Usage:
/bin/bash third_party/py/nanodo/google/run.sh --config=default
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Get the default hyperparameter configuration."""
  cfg = ml_collections.ConfigDict()
  # Global batch size. This must be divisible by the number of devices.
  cfg.batch_size = 256
  cfg.context_length = 512
  cfg.seed = 42

  # Data
  cfg.ds_name = "lm1b:1.1.0"
  cfg.vocab_path = (
      # /bigstore/t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model
      # Same as above but with BOS at position 2.
      "/cns/rs-d/home/pagi/vocabs/cc_all.32000.100extra.bos.model"
  )

  cfg.train_epochs = None  # None=>infinite

  # Optimizer
  cfg.opt = ml_collections.ConfigDict()

  # Note: lm1b has 30,301,028 training examples
  cfg.opt.num_train_steps = 100_000
  # Note: evaluation, checkpointing, and train and performance metrics logging
  # happens at regular frequencies specified below. If any of them doesn't
  # divide `num_train_steps`, the final result will not be saved / logged etc.
  # E.g., if `num_train_steps == 15` but `checkpoint_every_steps == 10`, only
  # checkpoints at steps 0 and 10 will be saved.

  cfg.opt.peak_learning_rate = 0.0016
  cfg.opt.init_learning_rate = 0.00016
  cfg.opt.final_learning_rate = 0.00016  # currently only "cosine" uses it
  cfg.opt.warmup_steps = 1000

  cfg.opt.decay_type = "cosine"
  cfg.opt.weight_decay = 0.1
  # It is common practice to set this to 1.0 for many well-known LLM trainings.
  cfg.opt.clip_by_global_norm = None
  # type of optimizer -- defaults to adamw if not specified.
  # options: ["adamw", "adafactor"]
  cfg.opt.optimizer = "adamw"

  # Transformer
  cfg.model_dim = 512
  cfg.mlp_dim = 2048
  cfg.num_heads = 8
  cfg.num_layers = 6
  cfg.dtype = "bfloat16"

  # Transformer block rematerialization / gradient checkpointing to save memory.
  # Ref: https://arxiv.org/abs/1604.06174
  # Clears all the intermediate activations inside Transformer blocks.
  # Set to False if speed is important (naive compute for one update step is
  # 6N (remat=False) vs 8N (remat=True) due to extra fwd pass), use
  # this when activation is causing HBM OOM error.
  cfg.remat = False

  # Checkpointing
  cfg.checkpoint = True
  cfg.checkpoint_every_steps = 2000
  # Path to the checkpoint to be restored. Current behavior is to restore from
  # this checkpoint but new checkpoints will be saved to the new workdir.
  cfg.checkpoint_restore_dir = None
  cfg.max_to_keep = 100

  # Eval
  cfg.eval_every_steps = 100
  cfg.eval_split = "test"  # 306,688 examples
  cfg.eval_steps = 100  # less if this exceeds 1 epoch
  cfg.eval_max_target_length = 512

  # FSDP
  # whether to shard model -- does not need to be specified, defaults to True.
  cfg.fsdp_enabled = True

  # Logging
  cfg.write_train_metrics_every_steps = 1  # train loss, gradient norms, etc.
  cfg.write_perf_metrics_every_steps = 100  # steps_per_sec, uptime.
  # Options to turn on/off writing to XM measurements.
  # For Vizier interface, we currently require write_to_xm_measurements=True
  cfg.write_to_xm_measurements = True
  # Option to turn on internal statistics: rms_norm, mean, std of per-layer,
  # module-wise statistics. Due to high-load, when setting this to True consider
  # turning off writing to XM measurements and rely on Datatables.
  cfg.log_internal_metrics = True

  # pygrain
  cfg.pygrain_worker_count = 16  # might increase this if input-bound
  # Buffer size (in unit of batches) for the data loader. Default to 2 so we
  # always prefetch another batch
  cfg.pygrain_worker_buffer_size = 2

  # Hardware / scheduling requirements. If not set or set to `""` or `0`, flags
  # passed to `xm_launch.py` are used.
  cfg.platform = ""
  cfg.priority = 0
  cfg.cell = ""
  return cfg
