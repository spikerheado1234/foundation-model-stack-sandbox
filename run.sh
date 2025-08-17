#! /bin/bash

python scripts/train_causal.py \
    --architecture=llama \
    --variant=micro \
    --tokenizer=char_tokenizer \
    --dataset_style=mock \
    --dataset_path="" \
    --max_steps=10 \
    --report_steps=5 \
    --batch_size=1 \
    --device_type=cuda \
    --epochs=1
