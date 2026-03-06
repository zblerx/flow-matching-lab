# Flow Matching Lab

A **minimal course + playground** for learning **generative flow models** from scratch.  
It focuses on **explicit training loops and small experiments** instead of large frameworks to reveal how generative models work.

## 1. Two Moons

The Two Moons dataset is a simple 2D toy problem that lets us visualize how flow models transform a simple distribution (like a Gaussian) into a more complex one.

- Unconditional flows learn the data distribution without any extra information.
- Conditional flows learn the distribution given a label, allowing us to control which mode is sampled.

### 1.1 Unconditional Flow
Run the training and sampling scripts:
```bash
# Rectified Flow
python train.py
python sample.py

# DDIM
python train.py --DDIM
python sample.py --checkpoint output/TwoMoons_ddim.pth --gif output/TwoMoons_ddim.gif
```

<table>
  <tr>
    <td align="center">
      <img src="output/TwoMoons.gif" width="300"><br>
      <b>Rectified Flow</b>
    </td>
    <td align="center">
      <img src="output/TwoMoons_ddim.gif" width="300"><br>
      <b>DDIM</b>
    </td>
  </tr>
</table>

### 1.2 Conditional
We can condition the flow on the moon label (0 or 1):
```bash
# Rectified Flow
python train.py --conditional
python sample.py --checkpoint output/TwoMoons_cond.pth --gif output/TwoMoons_cond.gif

# DDIM
python train.py --conditional --DDIM
python sample.py --checkpoint output/TwoMoons_cond_ddim.pth --gif output/TwoMoons_cond_ddim.gif
```

<table>
  <tr>
    <td align="center">
      <img src="output/TwoMoons_cond.gif" width="300"><br>
      <b>Rectified Flow</b>
    </td>
    <td align="center">
      <img src="output/TwoMoons_cond_ddim.gif" width="300"><br>
      <b>DDIM</b>
    </td>
  </tr>
</table>

### 1.3 Questions

#### 1. Flow Dynamics
- At inference, what does each step of the flow (or diffusion) do to the samples?
- How does the trajectory from `x_0` to `x_1` differ between Rectified Flow and DDIM?

2. Conditional vs Unconditional
- How does conditioning on the label affect the learned distribution?
- Why might a conditional flow produce cleaner or more separated modes?

3. Change the Model Size
- Modify the MLP and observe differences in sample quality.

4. Vary the Number of Steps
- Reduce --n_steps when sampling. How does it affect the smoothness of trajectories?

5. Noise Sensitivity
- Change the noise parameter in TwoMoons data loader. How robust is your flow to noisy data?

6. Compare Rectified Flow and DDIM interpolation trajectories.
- Select a starting point `x_0` from the base distribution and its corresponding target `x_1` from the data. Then, compare the trajectory produced by the flow model to the straight-line path from `x_0` to `x_1`, for both Rectified Flow and DDIM.


## 2. Chessboard
### Unconditional
```bash
# Rectified Flow
python train.py --dataset "ChessBoard" --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard.pth --gif output/ChessBoard.gif

# DDIM
python train.py --dataset "ChessBoard" --DDIM --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard_ddim.pth --gif output/ChessBoard_ddim.gif
```

<table>
  <tr>
    <td align="center">
      <img src="output/ChessBoard.gif" width="300"><br>
      <b>Rectified Flow</b>
    </td>
    <td align="center">
      <img src="output/ChessBoard_ddim.gif" width="300"><br>
      <b>DDIM</b>
    </td>
  </tr>
</table>

### Conditional
```bash
# Rectified Flow
python train.py --dataset "ChessBoard" --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3 --conditional
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard_cond.pth --gif output/ChessBoard_cond.gif

# DDIM
python train.py --dataset "ChessBoard" --model_config '{"model" : "MLP", "h" : 512}' --lr 1e-3 --conditional --DDIM
python sample.py --n_samples 50000 --model_config '{"model" : "MLP", "h" : 512}' --checkpoint output/ChessBoard_cond_ddim.pth --gif output/ChessBoard_cond_ddim.gif
```

<table>
  <tr>
    <td align="center">
      <img src="output/ChessBoard_cond.gif" width="300"><br>
      <b>Rectified Flow</b>
    </td>
    <td align="center">
      <img src="output/ChessBoard_cond_ddim.gif" width="300"><br>
      <b>DDIM</b>
    </td>
  </tr>
</table>

## 3. MNIST
### Unconditional
```bash
# Rectified Flow
python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST.pth --gif output/MNIST.gif

# DDIM
python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256 --DDIM
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST_ddim.pth --gif output/MNIST_ddim.gif
```

<table>
  <tr>
    <td align="center">
      <img src="output/MNIST.gif" width="300"><br>
      <b>Rectified Flow</b>
    </td>
    <td align="center">
      <img src="output/MNIST_ddim.gif" width="300"><br>
      <b>DDIM</b>
    </td>
  </tr>
</table>

### Conditional
```bash
# Rectified Flow
python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256 --conditional
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST_cond.pth --gif output/MNIST_cond.gif

# DDIM
python train.py --dataset "MNIST" --model_config '{"model" : "UNet"}' --lr 1e-3 --batch_size=256 --conditional --DDIM
python sample.py --n_samples 16 --model_config '{"model" : "UNet"}' --checkpoint output/MNIST_cond_ddim.pth --gif output/MNIST_cond_ddim.gif
```

<table>
  <tr>
    <td align="center">
      <img src="output/MNIST_cond.gif" width="300"><br>
      <b>Rectified Flow</b>
    </td>
    <td align="center">
      <img src="output/MNIST_cond_ddim.gif" width="300"><br>
      <b>DDIM</b>
    </td>
  </tr>
</table>


