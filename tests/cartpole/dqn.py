import gymnasium as gym

# SETUP
journal_learn = {
    "episode_idx": 0,
    "step_idx": 0,
    "episode_len": [0],
    "deposit_return": [0.0],
    "averaged_return": [],
}


# LOOP
env = gym.make("CartPole-v1", render_mode="rgb_array")
last_obs, info = env.reset()
for st in range(1000 * env.spec.max_episode_steps):
    # Play a step
    # pred_act, q_val = resolve_and_assess(rngs, explore_epsilon, critic, last_obs)
    act = pred_act.squeeze()
    next_obs, rew, term, trunc, info = env.step(np.array(act))
    buffer.store_step(last_obs, pred_act, rew, term, next_obs)
    last_obs = next_obs.copy()
    # Step statistics
    journal_learn["step_idx"] += 1  # TODO: update journal in a util function
    journal_learn["episode_len"][-1] += 1
    journal_learn["deposit_return"][-1] += rew
