# gym-trussbot

#### Installation
```shell script
cd ..
pip install -e gym-trussbot
```

#### Create Environment
```python
import gym
env = gym.make('gym_trussbot:trussbot-v0')
```

#### Usage
```
env.step(np.array([0,0,0,0.2,0.3,0.2]))     # shrinkage of beams

env.render()        # visualize a paused bot
# 'p': pause/run
# 'a': actuate/release
```
