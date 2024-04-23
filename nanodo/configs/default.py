"""Default Hyperparameter configuration.

Usage:
/bin/bash third_party/py/nanodo/run.sh --config=default
"""

import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """Get the default hyperparameter configuration."""
  cfg = ml_collections.ConfigDict()
  cfg.seed = 42

  # Data
  cfg.batch_size = 256  # Global batch size. Must be divisible by the #devices.
  cfg.train_epochs = None  # None=>infinite
  cfg.ds_name = "lm1b:1.1.0"
  cfg.vocab_path = "/cns/rs-d/home/pagi/vocabs/cc_all.32000.100extra.bos.model"
  # Same as /bigstore/t5-data/vocabs/cc_all.32000.100extra/sentencepiece.model,
  # but with BOS at position 2.

  # Transformer
  cfg.model = ml_collections.config_dict.create(
      D=512,  # model/embed dim  = qkv dim
      H=8,  # num attention heads
      L=512,  # max context/sequence length (move out of config?)
      N=6,  # number of transformer block layers
      F=2048,  # FF inner dimension
      dtype="bfloat16",  # computation dtype.
      fsdp_enabled=True,  # True to shard the model.
      remat=False,  # Transformer block gradient checkpointing to save memory.
  )

  # Optimizer
  cfg.opt = ml_collections.config_dict.create(
      num_train_steps=100_000,  # Note: lm1b has 30,301,028 training examples
      peak_learning_rate=0.0016,
      init_learning_rate=0.00016,
      final_learning_rate=0.00016,
      warmup_steps=1000,
      decay_type="cosine",
      weight_decay=0.1,
      clip_by_global_norm=None,  # 1.0 is common for many well-known LLMs.
      optimizer="adamw",
  )

  # Checkpointing
  cfg.checkpoint = True
  cfg.checkpoint_every_steps = 2000
  # Path to the checkpoint to be restored. Note than new checkpoints will be
  # saved to the new workdir.
  cfg.checkpoint_restore_dir = None
  cfg.max_to_keep = 100

  # Eval
  cfg.eval_every_steps = 100
  cfg.eval_split = "test"  # 306,688 examples
  cfg.eval_steps = 100  # less if this exceeds 1 epoch
  cfg.eval_max_target_length = 512

  # Logging
  cfg.write_train_metrics_every_steps = 1  # train loss, gradient norms, etc.
  cfg.write_perf_metrics_every_steps = 100  # steps_per_sec, uptime.
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
