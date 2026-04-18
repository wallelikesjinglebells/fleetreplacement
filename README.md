## Environment setup
### Conda
Create new environment
```bash
conda create --name <env-name>
```
Activate new environment
```bash
conda activate <env-name>
```
### Install 
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

[Stable Baselines3 Contrib](https://stable-baselines3.readthedocs.io/en/master/guide/sb3_contrib.html) for action masking
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
### Using
```bash
python train.py <scenario>
```

| Scenario                               | Command-line |
| -------------------------------------- | ------------ |
| Status Quo                             | SQ           |
| Scenario 1: Technological Stalemate    | S1           |
| Scenario 2: Technology without Mandate | S2           |
| Scenario 3: Ambition meets Reality     | S3           |
| Scenario 4: Autonomous Green Logistics | S4           |

This is the test branch