from lite_agents.vpg.components import VPGBuffer, ExperienceBatch, CategoricalActor
from flax import nnx
import optax

learning_journal = {
    "episode_idx": 0,
    "step_idx": 0,
    "episode_len": 0,
    "episode_return": 0.0,
    "deposit_return": [],
    "averaged_return": [],
}


if __name__=="__main__":
    import gymnasium as gym
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    rngs = nnx.Rngs(25)
    buffer = VPGBuffer([], [], [], [])
    actor = CategoricalActor(rngs, env.observation_space.shape[0], env.action_space.n, hidden_sizes=(128, 128))
    nnx.display(actor)
    learning_rate = 3e-4
    optimizer = nnx.Optimizer(actor, optax.adamw(learning_rate))
    max_epochs = 256
    last_obs, info = env.reset()
