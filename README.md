# Let jax spin up Deep RL

  [OpenAI Spinning Up](https://github.com/openai/spinningup.git) is my bible.

## Installation

We recommend using [uv](https://docs.astral.sh/uv/guides/install-python/) to install this package.

```sh
git clone https://github.com/linZHank/spinupax.git
cd spinupax
uv sync
```

## Example

### Train an Advantage Actor Critic (A2C) agent to play CartPole-v1

```python
from spinupax.a2c import agent

agent.learn()
```
