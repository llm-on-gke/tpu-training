pip install cloud-tpu-client

pip install https://storage.googleapis.com/tensorflow-public-build-artifacts/prod/tensorflow/official/release/nightly/linux_x86_tpu/wheel_py310/749/20240915-062017/github/tensorflow/build_output/tf_nightly_tpu-2.18.0.dev20240915-cp310-cp310-manylinux_2_17_x86_64.manylinux2014_x86_64.whl -f https://storage.googleapis.com/libtpu-tf-releases/index.html --force

export TPU_NAME=<NAME of the TPU>
export TPU_LOAD_LIBRARY=0
