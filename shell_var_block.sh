#!/bin/bash
#SBATCH -J train_owt_bd3lm_varblock
#SBATCH -o watch_folder/%x_%j.out
#SBATCH -e watch_folder/%x_%j.err
#SBATCH -N 1
#SBATCH --mem=32G
#SBATCH -t 72:00:00
#SBATCH --partition=a30
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

cd /vol/bitbucket/hk1122/bd3lms
source .venv/bin/activate

export HF_HOME=/vol/bitbucket/hk1122/.cache/huggingface
export HF_DATASETS_CACHE=$HF_HOME/datasets
export TRANSFORMERS_CACHE=$HF_HOME/transformers

BLOCK_SIZES="[64,32,16,8]"
BLOCK_SIZE=16

PRETRAIN_CKPT=null
# PRETRAIN_CKPT=kuleshov-group/bd3lm-owt-block_size1024-pretrain

python -u main.py \
    loader.global_batch_size=512 \
    loader.eval_global_batch_size=512 \
    loader.batch_size=8 \
    loader.eval_batch_size=8 \
    model=small \
    algo=bd3lm \
    algo.clip_search_widths=[0.5,0.6,0.7,0.8,0.9] \
    data=openwebtext-split \
    data.insert_train_special=False \
    data.insert_valid_special=False \
    data.insert_valid_eos=False \
    model.length=1024 \
    block_size=${BLOCK_SIZE} \
    "block_sizes=${BLOCK_SIZES}" \
    "wandb.name=bd3lm-owt-block_sizes${BLOCK_SIZES}" \
    mode=train \
    model.attn_backend=flex \
    training.resample=True \
    training.from_pretrained=$PRETRAIN_CKPT \
    data.cache_dir=/vol/bitbucket/hk1122/.cache/bd3lm_datasets