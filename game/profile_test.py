import jax.numpy as jnp
import jax
import numpy as np

from FootballGame import FootballGame, Action

@jax.jit
def test():
    env = FootballGame()
    state = env.reset()

    def f(carry, _):
        image = env.render(carry)
        return env.step(
            carry, 
            Action(move=jnp.array((0, 1)), kick=0),
            Action(move=jnp.array((0, -1)), kick=0)
        ), jnp.sum(image.flatten())

    state, data = jax.lax.scan(f, state, (), length=10000)

    image = env.render(state)
    return image, data

import time
start_time = time.time()

with jax.log_compiles():
    image, data = test()

print(data)

print(time.time() - start_time)

print(image)