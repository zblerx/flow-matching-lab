# FlowLab

A minimal **course + playground** to learn modern **generative flow models** from scratch.

FlowLab teaches how diffusion and flow-based generative models work by combining:

- **theory questions**
- **small experiments**
- **minimal PyTorch implementations**

The repository is designed so that you can **read, run, and modify everything**.

---

# Philosophy

Modern generative models are often hidden behind massive frameworks.

FlowLab takes the opposite approach:

- minimal code
- explicit training loops
- small datasets
- step-by-step experiments

The goal is to **understand the mechanics** of generative models.

---

# What you will learn

This course progressively builds the core ideas behind modern generative models.

Topics include:

- Flow Matching
- Conditional Generation
- Diffusion Models
- DDIM Sampling
- U-Net architectures
- Discrete Flow Matching

We start with **toy 2D datasets**, then scale to **images (MNIST)**.

---

# Course format

Each lesson follows the same structure:

### 1️⃣ Question

We start from a conceptual question.

Example:

> How can we transform a simple noise distribution into a complex data distribution?

---

### 2️⃣ Mathematical idea

We introduce the key equation behind the method.

Example:

\[
x_t = (1 - t)x_0 + t x_1
\]

---

### 3️⃣ Experiment

We run a small experiment to observe the behavior.

Example:

Train a flow model on TwoMoons:

```bash
python train.py




































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






