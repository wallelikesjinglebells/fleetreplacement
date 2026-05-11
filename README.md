# RL for Fleet Replacement 🚚

Reinforcement learning agents trained with [Stable Baselines3](https://stable-baselines3.readthedocs.io/en/master/) and a custom [Gymnasium](https://gymnasium.farama.org/) environment to find cost-optimal strategies for transitioning heavy-duty road freight fleets from diesel trucks to battery electric trucks. Five distinct electrification scenarios are explored.

## Environment Setup
<!-- ### Conda
Create new environment
```bash
conda create --name <env-name>
```
Activate new environment
```bash
conda activate <env-name>
``` -->
Python
```bash
conda install -c anaconda python=3.12.12
```
Gymnasium
```bash
pip install gymnasium
```
Stable Baselines3
```bash
pip install stable-baselines3[extra]
```
`extra` for [ProgressBarCallback](https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#progressbarcallback)

[Stable Baselines3 Contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html) for action masking via [MaskablePPO](https://sb3-contrib.readthedocs.io/en/master/modules/ppo_mask.html#maskableppo-policies)
```bash
pip install sb3-contrib
```
TensorBoard
```bash
pip install tensorboard
```
TensorFlow
```bash
pip install tensorflow
```
## Training 🏃‍➡️

### Train on Your Machine
```bash
python train.py <scenario>
```

| Scenario                               | Command-line argument |
| -------------------------------------- | ------------ |
| Status Quo                             | SQ           |
| Scenario 1: Technological Stalemate    | S1           |
| Scenario 2: Technology without Mandate | S2           |
| Scenario 3: Ambition meets Reality     | S3           |
| Scenario 4: Autonomous Green Logistics | S4           |

### TensorBoard Visualization
```bash
tensorboard --logdir <log-directory>
```

### Train on [Modal](https://modal.com/)
#### 1. Install Modal
```bash
pip install modal
```
#### 2. Authenticate
```bash
modal token new
```
#### 3. Train 
Either train a specific scenario
```bash
modal run modal_train.py --scenario <scenario>
```
or all scenarios in parallel:
```bash
modal run --detach modal_train.py --scenario all
```

## Evaluation 🔍

### Evaluating the Trained Models

#### TensorBoard Visualization
```bash
tensorboard --logdir existing_logs/v3
```

#### Replacement Strategies
Run one evaluation episode:
```bash
python evaluate_timeline.py SQ <scenario> --v3
```
Visualize replacement for one scenario across 50 episodes:
```bash
python visualize_timeline.py <scenario> --v3
```
Visualize replacement for one scenario for one episode:
```bash
python visualize_timeline.py <scenario> --v3 --episodes 1
```

#### Performance Comparison to Baselines
Create a boxplot comparing the performance of the RL model to baselines within one scenario:
```bash
python compare_to_baselines.py <scenario> --v3
```

| Baseline | SQ | S1 | S2 | S3 | S4 | Description |
|---|:---:|:---:|:---:|:---:|:---:|---|
| Random valid-action | ✓ | ✓ | ✓ | ✓ | ✓ | Lower bound; selects uniformly from valid actions at each step |
| EOL pure electric | ✓ | ✓ | ✓ | ✓ | ✓ | Hold until end-of-life, replace with BET throughout |
| EOL pure diesel | | ✓ | ✓ | | | Hold until end-of-life, replace with DT throughout (no purchase ban) |
| EOL diesel → post-ban electric | ✓ | | | ✓ | ✓ | Replace with DT at EOL until purchase ban, then BET |
| 5-year pure electric | ✓ | ✓ | ✓ | ✓ | ✓ | Replace at age 5 with BET throughout |
| 5-year pure diesel | | ✓ | ✓ | | | Replace at age 5 with DT throughout (no purchase ban) |
| 5-year diesel → post-ban electric | ✓ | | | ✓ | ✓ | Replace at age 5 with DT until purchase ban, then BET |
| Greedy electric | ✓ | ✓ | ✓ | ✓ | ✓ | Replace with BET whenever action mask permits |
| Greedy diesel | | ✓ | ✓ | | | Replace with DT whenever action mask permits (no purchase ban) |
| Cost-greedy | ✓ | ✓ | ✓ | ✓ | ✓ | Myopic: pick the lowest-cost valid action at each step |


### Evaluating Your Trained Models
Automatic saving of the best models from `train.py` to the correct folder is not implemented yet. Your models will be saved to `models\scenarios`. In order to evaluate, rename `scenarios` to anything from `v0` to `v38792879284` (except `v3`, which contains the provided trained models).

#### TensorBoard Visualization
```bash
tensorboard --logdir <log-directory>
```
#### Replacement Strategies
Run one evaluation episode of your best model:
```bash
python evaluate_timeline.py SQ <scenario> --<vN>
```
Visualize replacement using the best model for one scenario across 50 episodes:
```bash
python visualize_timeline.py <scenario> --<vN>
```
Visualize replacement using the best model for one scenario for one episode:
```bash
python visualize_timeline.py <scenario> --<vN> --episodes 1
```

#### Performance Comparison to Baselines
Create a boxplot comparing the performance of the RL model to baselines within one scenario:
```bash
python compare_to_baselines.py <scenario> --<vN>
```

