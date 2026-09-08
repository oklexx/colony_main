@echo off
python train.py --steps 1000000 --envs 8 --n-steps 512 --batch-size 1024 --obs-mode flat --compile False --eval-freq 0 --name quick_start
