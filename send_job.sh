#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4                    # One GPU is enough for qwen2.5:7b and avoids slow multi-GPU discovery
#SBATCH --cpus-per-task=16
##SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --hint=multithread
#SBATCH --mem-bind=local
#SBATCH --distribution=block:block
#SBATCH --account=YOUR_PROJECT_ID
#SBATCH --partition=boost_usr_prod
#SBATCH --qos=boost_qos_dbg
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

module load cuda/12.2
source "$HOME/miniconda3/etc/profile.d/conda.sh"   
conda activate dl-agents

# Find where the module puts the CUDA libs
CUDA_LIB=$(dirname $(which nvcc))/../lib64
echo "System CUDA lib: ${CUDA_LIB}"

export OLLAMA_MODELS="$HOME/.ollama/models"
# Put system libs BEFORE ollama's bundled ones
export LD_LIBRARY_PATH="${CUDA_LIB}:${ollama_cuda_lib}:${ollama_lib}:${LD_LIBRARY_PATH}"

unset OLLAMA_LLM_LIBRARY

# Resolve paths from the directory the job was submitted from
exercises_dir="${SLURM_SUBMIT_DIR:-$PWD}"
script_dir="${exercises_dir}"
cd "${exercises_dir}"

export PYTHONPATH="${script_dir}:${exercises_dir}:${PYTHONPATH}"
export NUMBA_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "ID: ${SLURM_JOB_ID} Nodes: ${SLURM_NNODES}, tasks: ${SLURM_NTASKS}, ranks/node: ${SLURM_NTASKS_PER_NODE}, threads/rank: ${SLURM_CPUS_PER_TASK}, numba_threads: ${NUMBA_NUM_THREADS}"

ollama_root=$HOME/.local
ollama_bin="${ollama_root}/bin/ollama"
ollama_lib="${ollama_root}/lib/ollama"
ollama_cuda_lib="${ollama_lib}/cuda_v12"

export OLLAMA_HOST=127.0.0.1:11436
export OLLAMA_LLM_LIBRARY=cuda_v12
export OLLAMA_LOAD_TIMEOUT=10m
export LD_LIBRARY_PATH="${ollama_cuda_lib}:${ollama_lib}:${LD_LIBRARY_PATH}"
export OLLAMA_NUM_PARALLEL=1      # already set, good
export OLLAMA_MAX_LOADED_MODELS=1 # limit to avoid multi-GPU probing
export OLLAMA_CONTEXT_LENGTH=8192 # set to max to avoid dynamic resizing overhead during the runs; 
unset HIP_VISIBLE_DEVICES ROCR_VISIBLE_DEVICES GPU_DEVICE_ORDINAL GGML_VK_VISIBLE_DEVICES
unset CUDA_VISIBLE_DEVICES

echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi -L || true

# Tunneling info
XDG_RUNTIME_DIR=""
node=$(hostname -s)
user=$(whoami)
portval=18892

echo -e "
# One-command tunnel from your LOCAL machine:
ssh -N -L ${portval}:127.0.0.1:${portval} \\
  -o ServerAliveInterval=30 -o ServerAliveCountMax=6 \\
  -J ${user}@login.leonardo.cineca.it ${user}@${node}.leonardo.local

# Then open in browser:
http://localhost:${portval}/

# Server running on:
http://${node}.leonardo.local:${portval}/lab
"                                                 

# Check what Slurm actually assigns 
echo "SLURM_JOB_GPUS=${SLURM_JOB_GPUS}"
echo "SLURM_GPUS_ON_NODE=${SLURM_GPUS_ON_NODE}"
nvidia-smi -L

# Check driver vs runtime compatibility
nvidia-smi  # shows driver version
ls "${ollama_cuda_lib}/"  # what CUDA libs does ollama bundle?

# Try to actually init CUDA
python3 -c "import ctypes; lib=ctypes.CDLL('libcuda.so.1'); print('libcuda OK')" || echo "libcuda FAILED"

# Check what cuda version the driver supports
cat /proc/driver/nvidia/version 2>/dev/null || true

# Launch Ollama server directly inside the allocated GPU job shell.
"${ollama_bin}" serve &
ollama_pid=$!
trap 'kill ${ollama_pid} 2>/dev/null || true' EXIT

ollama_ready=0                                     # FIX: initialize variable before the loop
for i in {1..60}; do
    if "${ollama_bin}" list >/dev/null 2>&1; then
        echo "Ollama is ready on ${OLLAMA_HOST}"
        ollama_ready=1
        break
    fi
    if ! kill -0 "${ollama_pid}" 2>/dev/null; then
        echo "Ollama exited before becoming ready" >&2
        wait "${ollama_pid}"
        exit 1
    fi
    sleep 1
done

if [ "${ollama_ready}" -ne 1 ]; then
    echo "Ollama did not become ready within 60 seconds" >&2
    exit 1
fi

"${ollama_bin}" list

# Run Python scripts
echo "Running temperature_sweep.py..."
python3 temperature_sweep.py -model qwen3:14b   -temperatures 0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0 -num_samples 100

echo "Running opinion_dynamics.py..."
for temp in 0.1 0.3 0.5 0.7 1.0 1.5 2.0 3.0; do
    echo "Running opinion_dynamics.py with temperature=${temp}..."
     python3 opinion_dynamics.py -model qwen3:14b -num_agents 4 -num_rounds 4 -temperature ${temp}
done

echo "Scripts completed successfully"