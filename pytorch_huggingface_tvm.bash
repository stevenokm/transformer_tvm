if [ ! -d "./.venv" ]; then
    echo "please create virtual environment with python3.8 "
    echo "cmd: python3.8 -m venv .venv; source .venv/bin/activate"
    exit 1
fi
# pytorch
# pip3 install torch==2.0.1+cu117 torchvision==0.15.2+cu117 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu117 # torch 2.0.1 @ cu117
pip3 install torch==2.0.0+cu117 torchvision==0.15.1+cu117 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu117 # torch 2.0.0 @ cu117
# pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113 # torch 1.12 @ cu113

# huggingface
pip3 install transformers==4.34.0 accelerate==0.20.3 datasets bitsandbytes

# tvm
# pip3 install apache-tvm-cu116-cu116 -f https://tlcpack.ai/wheels # tvm 0.9 @ cu116
# pip3 install apache-tvm-cu113-cu113 -f https://tlcpack.ai/wheels # tvm 0.9 @ cu113
# pip3 install tlcpack-cu116==0.11.1 -f https://tlcpack.ai/wheels # tvm 0.11.1 @ cu116
# built from source
# install dependency packages
# sudo apt-get update
# sudo apt-get install -y python3 python3-dev python3-setuptools gcc \
#     libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev
# sudo apt-get install -y clang-10 libpapi-dev

# build tvm on .venv
if [ ! -d "./.venv" ]; then
    echo "Please create virtual environment first!"
    exit 1
fi

if [ ! -d "./.venv/lib/tvm" ]; then
    git clone --recursive https://github.com/apache/tvm ./.venv/lib/tvm
fi
# ln -s python3.8/site-packages/nvidia/cudnn ./.venv/lib/cudnn
cd ./.venv/lib/tvm
git checkout v0.13.0
if [ ! -d "./build-release" ]; then
    mkdir ./build-release
fi
if [ ! -d "./build-debug" ]; then
    mkdir ./build-debug
fi

# NOTE: set(USE_LLVM ON) for CPU, set(USE_CUDA) for GPU
# NOTE: set(USE_CUDNN ON) for cuDNN
# NOTE: set(USE_LLVM /usr/bin/llvm-config-10) for llvm-10 on ubuntu
# NOTE: set(USE_PAPI ON) for detail profiling
# NOTE: set(USE_RELAY_DEBUG ON) for debug

function set_cmake_config {
    var_name=$1
    var_value=$2
    conf_file=$3
    echo "set(${var_name} ${var_value})" >>${conf_file}
}

# import default config.cmake
source ../../../config_cmake.bash

cd build-release/
config_cmake_dict["USE_CUDA"]="ON"
config_cmake_dict["USE_GRAPH_EXECUTOR_CUDA_GRAPH"]="ON"
config_cmake_dict["USE_LLVM"]="/usr/bin/llvm-config-10"
config_cmake_dict["USE_CUDNN"]="ON"
config_cmake_dict["USE_CUBLAS"]="ON"
config_cmake_dict["USE_CURAND"]="ON"
config_cmake_dict["USE_CUTLASS"]="ON"
config_cmake_dict["SUMMARIZE"]="ON"
if [ ! -f "config.cmake" ]; then
    for key in "${!config_cmake_dict[@]}"; do
        set_cmake_config "${key}" "${config_cmake_dict[${key}]}" "config.cmake"
    done
fi
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=86
# NOTE: for RTX 3090 or later, reffer to https://developer.nvidia.com/cuda-gpus
cd ..
cd build-debug/
config_cmake_dict["USE_RELAY_DEBUG"]="ON"
if [ ! -f "config.cmake" ]; then
    for key in "${!config_cmake_dict[@]}"; do
        set_cmake_config "${key}" "${config_cmake_dict[${key}]}" "config.cmake"
    done
fi
cmake .. \
    -DCMAKE_CUDA_ARCHITECTURES=86
# NOTE: for RTX 3090 or later, reffer to https://developer.nvidia.com/cuda-gpus
cd ..

sleep 10

TVM_BUILD_PATH="build-debug" make -j8

sleep 10

TVM_BUILD_PATH="build-release" make -j8

cd ../../..
pip3 install python-dotenv

TVM_HOME=./.venv/lib/tvm
echo 'TVM_LOG_DEBUG=1' >>.env
echo "TVM_HOME=${TVM_HOME}" >>.env
echo "TVM_LIBRARY_PATH=${TVM_HOME}/build-debug" >>.env
echo "PYTHONPATH=\${PYTHONPATH}:${TVM_HOME}/python" >>.env

pip3 install numpy decorator attrs
pip3 install tornado psutil xgboost cloudpickle
pip3 install pytest pandas scipy

# ecco
pip3 install matplotlib ipython scikit-learn seaborn pytest PyYAML captum
echo "PYTHONPATH=\${PYTHONPATH}:./ecco" >>.env
