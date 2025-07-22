# Directory Description

This directory stores **training scripts** that initiate the learning process for specific environments and models.

These scripts load environments (e.g., `CartPole-v1`), initialize the corresponding neural network architectures (from `agents/`), define loss functions, and implement training loops.

### Typical responsibilities of a file in this directory:
- Create the environment via `gym.make()`
- Instantiate the model (e.g., `A2CBaselineCartPole`)
- Set up optimizers and hyperparameters
- Run the training loop with reward tracking
- Optionally save logs or model checkpoints
