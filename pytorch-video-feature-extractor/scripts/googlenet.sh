#!/usr/bin/bash
set -x

echo $(pwd)
cd ../ && python main.py \
    --video_dpath /mnt/g/Github/video_summarizer/sample_video \
    --model googlenet \
    --batch_size 50 \
    --stride 5 \
    --out /mnt/g/Github/video_summarizer/sample_feature/sample_GoogleNet.h5