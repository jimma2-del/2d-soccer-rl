import jax.numpy as jnp
import jax
import numpy as np

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
goal = 0

while True:
    left_player_move_y = 0
    left_player_move_x = 0

    right_player_move_y = 0
    right_player_move_x = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    keys = pygame.key.get_pressed()
        
    if keys[pygame.K_w]: left_player_move_y -= 1
    if keys[pygame.K_s]: left_player_move_y += 1
    if keys[pygame.K_a]: left_player_move_x -= 1
    if keys[pygame.K_d]: left_player_move_x += 1

    if keys[pygame.K_UP]: right_player_move_y -= 1
    if keys[pygame.K_DOWN]: right_player_move_y += 1
    if keys[pygame.K_LEFT]: right_player_move_x -= 1
    if keys[pygame.K_RIGHT]: right_player_move_x += 1

    left_player_kick = keys[pygame.K_LSHIFT] or keys[pygame.K_SPACE]
    right_player_kick = keys[pygame.K_KP0]

    if goal == 0:
        state, goal = env.step(state, 
            Action(move=jnp.array((left_player_move_y, left_player_move_x)), kick=left_player_kick), 
            Action(move=jnp.array((right_player_move_y, right_player_move_x)), kick=right_player_kick))
        
        if goal != 0: print(goal)

    #print(state.left_player_pos)

    image_array = np.array(env.render(state, left_player_kick, right_player_kick))
    pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    clock.tick(FPS)