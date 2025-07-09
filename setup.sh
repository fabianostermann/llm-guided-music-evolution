#!/bin/bash

# clamp page: https://ai-muzic.github.io/clamp/
# clamp code: https://github.com/microsoft/muzic/tree/main/clamp

model_dir="clamp_sander-wood"
mkdir -p $model_dir

# clamp model 1: https://huggingface.co/sander-wood/clamp-small-512
model_name="clamp-small-512"
mkdir -p $model_dir/$model_name/
wget -nc https://huggingface.co/sander-wood/$model_name/resolve/main/config.json -P $model_dir/$model_name/
wget -nc https://huggingface.co/sander-wood/$model_name/resolve/main/pytorch_model.bin -P $model_dir/$model_name/

# clamp model 2: https://huggingface.co/sander-wood/clamp-small-1024
model_name="clamp-small-1024"
mkdir -p $model_dir/$model_name/
wget -nc https://huggingface.co/sander-wood/$model_name/resolve/main/config.json -P $model_dir/$model_name/
wget -nc https://huggingface.co/sander-wood/$model_name/resolve/main/pytorch_model.bin -P $model_dir/$model_name/

echo "Please also install:"
echo "$ sudo apt install abcm2ps abcmidi"
