#pip install cloud-tpu-client


#pip install gin-config && pip install tensorflow-datasets && pip install tf-keras-nightly --no-deps

#pip install https://storage.googleapis.com/tensorflow-public-build-artifacts/prod/tensorflow/official/release/nightly/linux_x86_tpu/wheel_py310/749/20240915-062017/github/tensorflow/build_output/tf_nightly_tpu-2.18.0.dev20240915-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

git clone https://github.com/tensorflow/recommenders.git

git clone https://github.com/tensorflow/models.git

export PYTHONPATH=~/recommenders/:~/models/


export TF_XLA_FLAGS='--tf_mlir_enable_mlir_bridge=true --tf_xla_sparse_core_disable_table_stacking=true --tf_mlir_enable_convert_control_to_data_outputs_pass=true --tf_mlir_enable_merge_control_flow_pass=true'


TF_USE_LEGACY_KERAS=1 TPU_LOAD_LIBRARY=0 python3 ./models/official/recommendation/ranking/train.py  --mode=train     --model_dir=gs://ptxla-debug/tf/sc/dlrm/runs/2/ --params_override="
runtime:
  distribution_strategy: tpu
  mixed_precision_dtype: 'mixed_bfloat16'
task:
  use_synthetic_data: false
  use_tf_record_reader: true
  train_data:
    input_path: 'gs://trillium-datasets/criteo/train/day_*/*'
    global_batch_size: 16384
    use_cached_data: true
  validation_data:
    input_path: 'gs://trillium-datasets/criteo/eval/day_*/*'
    global_batch_size: 16384
    use_cached_data: true
  model:
    num_dense_features: 13
    bottom_mlp: [512, 256, 128]
    embedding_dim: 128
    interaction: 'multi_layer_dcn'
    dcn_num_layers: 3
    dcn_low_rank_dim: 512
    size_threshold: 8000
    top_mlp: [1024, 1024, 512, 256, 1]
    use_multi_hot: true
    concat_dense: false
    dcn_use_bias: true
    vocab_sizes: [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36]
    multi_hot_sizes: [3,2,1,2,6,1,1,1,1,7,3,8,1,6,9,5,1,1,1,12,100,27,10,3,1,1]
    max_ids_per_chip_per_sample: 128
    max_ids_per_table: [280, 128, 64, 272, 432, 624, 64, 104, 368, 352, 288, 328, 304, 576, 336, 368, 312, 392, 408, 552, 2880, 1248, 720, 112, 320, 256]
    max_unique_ids_per_table: [104, 56, 40, 32, 72, 32, 40, 32, 32, 144, 64, 192, 32, 40, 136, 32, 32, 32, 32, 240, 1352, 432, 120, 80, 32, 32]
    use_partial_tpu_embedding: false
    size_threshold: 0
    initialize_tables_on_host: true
trainer:
  train_steps: 10000
  validation_interval: 1000
  validation_steps: 660
  summary_interval: 1000
  steps_per_loop: 1000
  checkpoint_interval: 0
  optimizer_config:
    embedding_optimizer: 'Adagrad'
    dense_optimizer: 'Adagrad'
    lr_config:
      decay_exp: 2
      decay_start_steps: 70000
      decay_steps: 30000
      learning_rate: 0.025
      warmup_steps: 0
    dense_sgd_config:
      decay_exp: 2
      decay_start_steps: 70000
      decay_steps: 30000
      learning_rate: 0.00025
      warmup_steps: 8000
  train_tf_function: true
  train_tf_while_loop: true
  eval_tf_while_loop: true
  use_orbit: true
  pipeline_sparse_and_dense_execution: true"
