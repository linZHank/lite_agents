from spinupax.a2c import agent

agent.learn(
    max_epochs=128,
    eval_flag=True,
)
# agent.play()
