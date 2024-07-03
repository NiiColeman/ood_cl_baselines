#!/bin/bash 
#SBATCH -A IscrC_FoundCL
#SBATCH -p boost_usr_prod
#SBATCH --time=24:00:00   # format: HH:MM:SS
#SBATCH -N 4                # 1 node
#SBATCH --ntasks-per-node=8 # 4 tasks out of 32
#SBATCH --gres=gpu:4      # 4 gpus per node out of 4
#SBATCH --mem=64GB          # memory per node out of 494000MB 
#SBATCH --job-name=lora_masks
#SBATCH --output=/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/baselines/outs/ft_lora_fgvc_aircraft_exp_%j.out
#SBATCH --error=/leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/baselines/outs/ft_lora_fgvc_aircraft_exp_%j.err


export CUDA_HOME=/leonardo/prod/opt/compilers/cuda/12.1/none
export TORCH_CUDA_ARCH_LIST="5.2 6.0 6.1 7.0 7.5 8.0 8.6+PTX"
source /leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/l2/bin/activate


# python main.py --config resources/config.yaml --data_path /leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/data/cub200/CUB_200_2011 --dataset cub200  --experiment vit_lora_finetuning
# python main.py --config resources/config.yaml --data_path /leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/data/tinyimagenet/tiny-imagenet-200  --dataset tinyimagenet  --experiment vit_lora_finetuning
python main.py --config resources/config.yaml --data_path  /leonardo_scratch/fast/IscrC_FoundCL/cl/lora-CL/data/fgvc_aircraft/fgvc-aircraft-2013b/data  --dataset fgvc_aircraft  --experiment vit_lora_finetuning