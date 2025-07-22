import gymnasium as gym
import numpy as np
from lite_agents.vpg.components import VPGBuffer, ExperienceBatch, CategoricalActor
import jax.numpy as jnp
from flax import nnx
import optax


@nnx.jit
def make_decision(rngs: nnx.Rngs, actor, obs: np.ndarray):
    log_prob, pi = actor(obs)  # policy: log(pi(a|s))
    act = pi.sample(seed=rngs)
    return act, jnp.exp(log_prob)


@nnx.jit
def objective_fn(actor, experience_batch: ExperienceBatch):
    _, policies = actor(experience_batch.obs)
    logpa_batch = policies.log_prob(experience_batch.act)
    objective_batch = experience_batch.ret * logpa_batch  # NOT expected return

    return -objective_batch.mean()


@nnx.jit
def update_params(actor, optimizer, experience_batch):
    grad_fn = nnx.value_and_grad(objective_fn)
    objective, grads = grad_fn(actor, experience_batch)
    optimizer.update(grads)  # In-place updates.


def learn(
    env_name: str = "CartPole-v1",
    seed: int = 0,
    max_epochs: int = 64,
    learning_rate: float = 3e-4,
    discount: float = 0.99,
    hidden_sizes: tuple = (64, 64),
    min_epoch_episodes: int = 10,  # minimal episodes per epoch
):
    # SETUP
    env = gym.make(env_name, render_mode="rgb_array")
    rngs = nnx.Rngs(seed)
    buffer = VPGBuffer([], [], [], [])
    actor = CategoricalActor(
        rngs,
        env.observation_space.shape[0],
        env.action_space.n,
        hidden_sizes,
    )
    nnx.display(actor)
    optimizer = nnx.Optimizer(actor, optax.adamw(learning_rate))
    learning_journal = {
        "episode_idx": 0,
        "step_idx": 0,
        "episode_len": [0],
        "deposit_return": [0.0],
        "averaged_return": [],
    }
    last_obs, info = env.reset()

    # LOOP
    for e in range(max_epochs):
        for st in range(
            (min_epoch_episodes + 1) * env.spec.max_episode_steps
        ):  # at least 10 finished episodes
            act, _ = make_decision(rngs, actor, last_obs)
            next_obs, rew, term, trunc, info = env.step(np.array(act))
            buffer.store_step(last_obs, act, rew)
            # TODO: update journal in a util function
            learning_journal["step_idx"] += 1
            learning_journal["episode_len"][-1] += 1
            learning_journal["deposit_return"][-1] += rew
            last_obs = next_obs.copy()
            if term or trunc:
                buffer.wrapup_episode(learning_journal["episode_len"][-1], discount)
                # Episode statistics
                learning_journal["episode_idx"] += 1
                learning_journal["averaged_return"].append(
                    sum(learning_journal["deposit_return"])
                    / len(learning_journal["deposit_return"])
                )
                # TODO: need a logger
                print(
                    f"---\nepisode: {learning_journal['episode_idx']}, length: {learning_journal['episode_len'][-1]}, return: {learning_journal['deposit_return'][-1]}\n---\n"
                )
                # Reset episode
                last_obs, _ = env.reset()
                learning_journal["episode_len"].append(0)
                learning_journal["deposit_return"].append(0.0)
                if st > min_epoch_episodes * env.spec.max_episode_steps:
                    break
        # Epoch statistics
        print(
            f"===\nepoch {e + 1} \n\ttotal steps: {learning_journal['step_idx']}\n\taveraged return: {learning_journal['averaged_return'][-1]}\n==="
        )
        exp_batch = buffer.extract_experience()
        update_params(actor, optimizer, exp_batch)
        buffer = VPGBuffer([], [], [], [])
    # TODO: need a plotter
    import matplotlib.pyplot as plt
    from pathlib import Path

    plt.plot(learning_journal["averaged_return"])
    plt.ylim(0, 200)
    plt.yticks(np.arange(0, 200, 20))
    plt.grid(visible=True, axis="y")
    plt.show()
    # plt.savefig(Path(__file__).parent.joinpath(f"{env_name}.png"))


if __name__ == "__main__":
    learn(max_epochs=128)
