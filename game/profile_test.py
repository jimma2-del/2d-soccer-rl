import cv2
import jax.numpy as jnp
import jax
import numpy as np

from FootballGame import FootballGame

@jax.jit
def test():
    env = FootballGame()
    state = env.reset()

    def f(carry, _):
        image = env.render(carry)
        return env.step(carry), jnp.sum(image.flatten())

    state, data = jax.lax.scan(f, state, (), length=1)

    image = env.render(state)
    return image, data

import time
start_time = time.time()

image, data = test()
print(data)

print(time.time() - start_time)

WINDOW_NAME = "Game"

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
cv2.imshow(WINDOW_NAME, np.array(image))

cv2.waitKey()