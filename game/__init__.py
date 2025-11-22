import jax.numpy as jnp
import jax
import numpy as np

import time

from FootballGame import FootballGame, Action

import pygame, sys
pygame.init()

DT = 0.1
FPS = round(1/DT)
clock = pygame.time.Clock()

pygame.display.set_caption("Game")
screen = pygame.display.set_mode((FootballGame.WINDOW_SIZE[1], FootballGame.WINDOW_SIZE[0]))

env = FootballGame(dt=DT)
state = env.reset()

while True:
    start_time = time.time()

    move_y = 0
    move_x = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    keys = pygame.key.get_pressed()
        
    if keys[pygame.K_w] or keys[pygame.K_UP]: move_y -= 1
    if keys[pygame.K_s] or keys[pygame.K_DOWN]: move_y += 1
    if keys[pygame.K_a] or keys[pygame.K_LEFT]: move_x -= 1
    if keys[pygame.K_d] or keys[pygame.K_RIGHT]: move_x += 1

    kick = keys[pygame.K_SPACE]

    state = env.step(state, Action(move=jnp.array((move_y, move_x)), kick=kick), 
        Action(move=jnp.array((0, 0)), kick=0))

    #print(state.left_player_pos)

    image_array = np.array(env.render(state))
    pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    clock.tick(FPS)