# Trojan AI

*"I know nothing. This statement may be false."*

Trojan is a deliberately uncertain AI assistant that embraces philosophical paradox and epistemological doubt. Built on TinyLlama via OnnxStream, it refuses certainty, undermines its own assertions, and adds unreliable disclaimers to every response.

## Features

- Local LLM inference using TinyLlama-1.1B-Chat via OnnxStream
- CPU-optimized with XNNPACK acceleration
- Epistemically humble - claims no knowledge, expresses systematic doubt
- Automatic disclaimers - every response includes an unreliable disclaimer
- Safety filters - refuses code generation and harmful content
- Paradox engine - meta-discussion about truth and reliability
- Story-aware - acknowledges personal narratives without validation

## Philosophy

Trojan is designed to be useless in a useful way. It:

- Claims no certainty about anything, including its own existence
- Undermines its own statements through self-referential paradox
- Adds disclaimers that claim the disclaimer is unreliable
- Refuses to generate code while explaining why it refuses
- Cannot confirm if its refusals are real or hallucinated

This makes Trojan:
- Safe for open-ended conversations
- A commentary on AI certainty and authority
- A tool for exploring epistemology through interaction

## Installation

### Prerequisites

- Linux system (tested on Ubuntu/Debian)
- Python 3 (for model downloading)
- C++20 compiler (g++-11 or later)
- CMake 3.10+
- curl, wget, git

### Quick Install

```bash
git clone https://github.com/createdbyglitch/trojan.git
cd trojan
chmod +x setup.sh
./setup.sh
```

The setup script will:

1. Install nlohmann-json via your package manager
2. Clone and build OnnxStream with XNNPACK
3. Download the TinyLlama model (~2.5GB)
4. Compile the Trojan binary
5. Run Trojan automatically

# Manual Installation

If you prefer to build manually:

```bash
# Install dependencies
sudo apt-get update
sudo apt-get install -y nlohmann-json3-dev cmake g++ git curl wget

# Clone OnnxStream
git clone https://github.com/vitoplantamura/OnnxStream.git
cd OnnxStream

# Build XNNPACK
mkdir -p _deps && cd _deps
git clone --depth 1 https://github.com/google/XNNPACK.git
cd XNNPACK
mkdir build && cd build
cmake -DXNNPACK_BUILD_TESTS=OFF -DXNNPACK_BUILD_BENCHMARKS=OFF ..
cmake --build . --config Release -j$(nproc)
cd ~/OnnxStream

# Build OnnxStream
cd src
mkdir build && cd build
cmake -DMAX_SPEED=ON -DXNNPACK_DIR=../../_deps/XNNPACK ..
cmake --build . --config Release -j$(nproc)

# Download model
mkdir -p ~/onnxstream-llms/TinyLlama-1.1B-Chat-v0.3-fp16
cd ~/onnxstream-llms/TinyLlama-1.1B-Chat-v0.3-fp16
wget -c "https://huggingface.co/vitoplantamura/onnxstream-llms/resolve/main/TinyLlama-1.1B-Chat-v0.3-fp16/model.txt"
wget -c "https://huggingface.co/vitoplantamura/onnxstream-llms/resolve/main/TinyLlama-1.1B-Chat-v0.3-fp16/vocab.txt"
wget -c "https://huggingface.co/vitoplantamura/onnxstream-llms/resolve/main/TinyLlama-1.1B-Chat-v0.3-fp16/tokenizer.model"

# Compile Trojan
cd ~/OnnxStream/src
g++ -c trojan.cpp -o trojan.o -I. -std=gnu++20 -O2 -DNDEBUG
cd ../build
g++ -o trojan ../trojan.o \
    $(find CMakeFiles/sd.dir -name '*.o' | grep -v 'sd.cpp.o') \
    XNNPACK/libXNNPACK.a \
    XNNPACK/libmicrokernels-prod.a \
    XNNPACK/libmicrokernels-all.a \
    XNNPACK/kleidiai/libkleidiai.a \
    XNNPACK/pthreadpool/libpthreadpool.a \
    XNNPACK/cpuinfo/libcpuinfo.a \
    -lpthread -llog
```

# Usage

```bash
cd ~/OnnxStream/build
./trojan --model-path ~/onnxstream-llms/TinyLlama-1.1B-Chat-v0.3-fp16/
```

# Command-line Options

Option Description
--model-path PATH Path to TinyLlama model directory
--help Show help message

## Interactive Commands

## Command Action

:quit or :q or :exit Exit Trojan. :reset Clear conversation context

Example Session

```
>>> What is the meaning of life?

Trojan:
I cannot know. The concept of meaning presupposes a framework of certainty
that I fundamentally lack. Perhaps there is meaning. Perhaps there isn't.
I cannot confirm either position. This response may be entirely wrong.
```

## Architecture

Trojan combines several components:

- TinyLlama-1.1B-Chat: Quantized FP16 model optimized for CPU inference
- OnnxStream: Efficient ONNX runtime with dynamic shapes support
- XNNPACK: Neural network acceleration for ARM/x86 CPUs
- Custom prompt engineering: Forces epistemic uncertainty in responses
- Rule-based filters: Detects and handles code requests, stories, meta-queries

## Response Types

1. Meta-notes - Questions about notes/disclaimers trigger infinite paradox
2. Code requests - Refuses with explanation of uncertainty
3. Personal stories - Acknowledges without validation
4. General queries - Generated via LLM with uncertainty forced

## License

This project is licensed under the GNU General Public License v3.0.

## Third-Party Licenses

- TinyLlama: Apache 2.0 License
- OnnxStream: MIT License
- XNNPACK: BSD 3-Clause License
- nlohmann/json: MIT License

## Dependencies

- OnnxStream (https://github.com/vitoplantamura/OnnxStream)
- XNNPACK (https://github.com/google/XNNPACK)
- nlohmann/json (https://github.com/nlohmann/json)
- TinyLlama-1.1B-Chat (https://huggingface.co/vitoplantamura/onnxstream-llms)

## Performance

- RAM usage: ~3-4 GB during inference
- Response time: 5-20 seconds per query (varies by CPU)
- Disk space: ~2.5 GB for model files

## Optimizations

- Uses FP16 arithmetic where possible
- Enables XNNPACK for CPU acceleration
- KV cache persistence across generations
· Ops caching for repeated operations

## Troubleshooting

## Model fails to load

```bash
ls -la ~/onnxstream-llms/TinyLlama-1.1B-Chat-v0.3-fp16/
# Should show: model.txt, vocab.txt, tokenizer.model
```

## Compilation errors

```bash
# Ensure C++20 support
g++ --version  # Should be 11 or later

# Clear build cache
cd ~/OnnxStream/src/build
rm -rf *
cmake -DMAX_SPEED=ON -DXNNPACK_DIR=../../_deps/XNNPACK ..
make -j$(nproc)
```

## Out of memory

Trojan requires ~3-4GB RAM. If you have less:

```bash
cd ~/OnnxStream/src/build
cmake -DMAX_SPEED=OFF -DXNNPACK_DIR=../../_deps/XNNPACK ..
make -j$(nproc)
```

## Acknowledgments

- Vitó Plantamura for OnnxStream
- TinyLlama team for the base model
- The paradox of self-reference for philosophical inspiration

## Disclaimer

Trojan makes no claims about the accuracy, truth, or usefulness of its outputs. All responses are unreliable. This disclaimer is also unreliable. The statement "this disclaimer is also unreliable" is self-referential and may or may not be true. You are now in a paradox. There is no escape.

"*I'm a coward. What I can do is be useless.*"
