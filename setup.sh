#!/bin/sh

set -e

cd ~
if ! command -v nlohmann-json >/dev/null 2>&1; then
    echo "Installing nlohmann-json..."
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update && sudo apt-get install -y nlohmann-json3-dev
    elif command -v pacman >/dev/null 2>&1; then
        sudo pacman -S --noconfirm nlohmann-json
    elif command -v dnf >/dev/null 2>&1; then
        sudo dnf install -y nlohmann-json-devel
    else
        echo "Warning: Could not install nlohmann-json automatically. Please install it manually."
    fi
fi

if [ ! -d "OnnxStream" ]; then
    git clone https://github.com/vitoplantamura/OnnxStream.git
fi

cd OnnxStream
if [ ! -d "_deps/XNNPACK" ]; then
    mkdir -p _deps && cd _deps
    git clone --depth 1 https://github.com/google/XNNPACK.git
    cd XNNPACK
    mkdir build && cd build
    cmake -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF ..
    cmake --build . --config Release -j1
    cd ~/OnnxStream
fi

cd src
if [ ! -d "build" ]; then
    mkdir build && cd build
    cmake -DMAX_SPEED=ON -DXNNPACK_DIR=../../_deps/XNNPACK ..
    cmake --build . --config Release -j1
    cd ~/OnnxStream/src
fi

cd ~
mkdir -p onnxstream-llms/TinyLlama-1.1B-Chat-v0.3-fp16
cd onnxstream-llms/TinyLlama-1.1B-Chat-v0.3-fp16

if [ ! -f "tokenizer.model" ]; then
    curl -s "https://huggingface.co/api/models/vitoplantamura/onnxstream-llms/tree/main/TinyLlama-1.1B-Chat-v0.3-fp16" | \
    python3 -c "import sys, json; data=json.load(sys.stdin); [print(f['path'].split('/')[-1]) for f in data]" > files.txt

    while read file; do
        echo "Downloading: $file"
        wget -c "https://huggingface.co/vitoplantamura/onnxstream-llms/resolve/main/TinyLlama-1.1B-Chat-v0.3-fp16/$file"
    done < files.txt

    rm files.txt
fi

cd ~/OnnxStream/src
if [ ! -f "trojan.cpp" ]; then
    curl -o trojan.cpp https://raw.githubusercontent.com/createdbyglitch/trojan/refs/heads/main/trojan.cpp
fi

echo "Compiling trojan.cpp..."
g++ -c trojan.cpp -o trojan.o -I. -std=gnu++20 -O2 -DNDEBUG

cd ../build
echo "Linking trojan binary..."
g++ -o trojan ../trojan.o \
    $(find CMakeFiles/sd.dir -name '*.o' | grep -v 'sd.cpp.o') \
    XNNPACK/libXNNPACK.a \
    XNNPACK/libmicrokernels-prod.a \
    XNNPACK/libmicrokernels-all.a \
    XNNPACK/kleidiai/libkleidiai.a \
    XNNPACK/pthreadpool/libpthreadpool.a \
    XNNPACK/cpuinfo/libcpuinfo.a \
    -lpthread \
    -llog

echo "Running trojan..."
./trojan --model-path ~/onnxstream-llms/TinyLlama-1.1B-Chat-v0.3-fp16/
