#!/usr/bin/env bash
python train_source.py --gpu_id 0 --src rc --tgt rl --tradeoff 0.001 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src rl --tgt rc --tradeoff 0.001 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src rl --tgt t --tradeoff 0.001 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src rc --tgt t --tradeoff 0.003 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src t --tgt rc --tradeoff 0.003 --tradeoff2 0.03 
python train_source.py --gpu_id 0 --src t --tgt rl --tradeoff 0.003 --tradeoff2 0.03 