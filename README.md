# Rectified Flow From Scratch (2D Toy Implementation)

This repository provides a minimal yet complete implementation of Rectified Flow,
a modern generative modeling technique that learns deterministic transport maps
from noise to data.

We train a velocity field on a 2D moons dataset and visualize:

- Learned vector fields
- Flow trajectories
- Generated samples


python train.py
python sample.py
python train.py --conditional
python sample.py --checkpoint output/TwoMoons_cond.pth --gif output/TwoMoons_cond.gif

python train.py --dataset "ChessBoard" --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard.pth --gif output/ChessBoard.gif
python train.py --dataset "ChessBoard" --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3 --conditional
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard_cond.pth --gif output/ChessBoard_cond.gif

python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST.pth --gif output/MNIST.gif
python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256 --conditional
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST_cond.pth --gif output/MNIST_cond.gif

python train.py --DDIM
python sample.py --checkpoint output/TwoMoons_ddim.pth --gif output/TwoMoons_ddim.gif
python train.py --conditional --DDIM
python sample.py --checkpoint output/TwoMoons_cond_ddim.pth --gif output/TwoMoons_cond_ddim.gif

python train.py --dataset "ChessBoard" --DDIM --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard_ddim.pth --gif output/ChessBoard_ddim.gif
python train.py --dataset "ChessBoard" --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3 --conditional --DDIM
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard_cond_ddim.pth --gif output/ChessBoard_cond_ddim.gif


python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256 --DDIM
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST_ddim.pth --gif output/MNIST_ddim.gif
python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256 --conditional --DDIM
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST_cond_ddim.pth --gif output/MNIST_cond_ddim.gif
##






