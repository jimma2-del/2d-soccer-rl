import jax.numpy as jnp
import jax
import numpy as np

from FootballGame import FootballGame, Action, Settings

PLAYERS_PER_TEAM = 2

def make_action(players_action: dict[int, tuple[tuple[int, int], int]], players=PLAYERS_PER_TEAM):
    move = jnp.zeros((players, 2), dtype=jnp.float32)
    kick = jnp.zeros((players), dtype=jnp.float32)

    for i, action in players_action.items():
        move = move.at[i].set(jnp.array(action[0]))
        kick = kick.at[i].set(action[1])

    return Action(move=move, kick=kick)

import pygame, sys
pygame.init()

print("CONTROLS")
print("Player 1: E S D F to Move, SHIFT to Kick")
print("Player 1: P L SEMICOLON QUOTE to Move, RIGHT-SHIFT to Kick")

DT = 0.1
FPS = round(1/DT)
clock = pygame.time.Clock()

env = FootballGame(dt=DT, settings=Settings(players_per_team=PLAYERS_PER_TEAM))
state = env.reset()
goal = 0

pygame.display.set_caption("2D Football Game")
screen = pygame.display.set_mode((env._cached_consts.window_size[1], env._cached_consts.window_size[0]))

while True:
    left_player_move_y = 0
    left_player_move_x = 0

    right_player_move_y = 0
    right_player_move_x = 0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            sys.exit()

    keys = pygame.key.get_pressed()
        
    if keys[pygame.K_e]: left_player_move_y -= 1
    if keys[pygame.K_d]: left_player_move_y += 1
    if keys[pygame.K_s]: left_player_move_x -= 1
    if keys[pygame.K_f]: left_player_move_x += 1

    if keys[pygame.K_p]: right_player_move_y -= 1
    if keys[pygame.K_SEMICOLON]: right_player_move_y += 1
    if keys[pygame.K_l]: right_player_move_x -= 1
    if keys[pygame.K_QUOTE]: right_player_move_x += 1

    left_player_kick = keys[pygame.K_LSHIFT] #or keys[pygame.K_SPACE]
    right_player_kick = keys[pygame.K_RSHIFT]

    left_player_action = make_action({ 1: ((left_player_move_y, left_player_move_x), left_player_kick) })
    right_player_action = make_action({ 1: ((right_player_move_y, right_player_move_x), right_player_kick) })

    if goal == 0:
        state, goal = env.step(state, left_player_action, right_player_action)
        if goal != 0: print(goal)

    #print(state.left_player_pos)

    image_array = np.array(env.render(state, left_player_action.kick, right_player_action.kick))
    pygame_surface = pygame.surfarray.make_surface(image_array.swapaxes(0,1))
    screen.blit(pygame_surface, (0,0))
    pygame.display.flip()

    clock.tick(FPS)