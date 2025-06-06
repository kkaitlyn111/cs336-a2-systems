#!/bin/bash
#SBATCH --job-name=torchFA
#SBATCH --partition=a1-batch
#SBATCH --qos=a1-batch-qos
#SBATCH -c 1
#SBATCH --time=00:10:00
#SBATCH --output=torchFA_%j.out
#SBATCH --error=torchFA_%j.err

uv run pytest -k test_flash_forward_pass_pytorch

