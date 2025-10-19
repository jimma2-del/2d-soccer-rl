from chex import dataclass
import chex
import jax.numpy as jnp
import jax
import functools
import dataclasses

from utils import draw_circle, coll_time_moving_circle_circle

@dataclass
class State:
    left_player_pos: chex.Array
    left_player_vel: chex.Array

    right_player_pos: chex.Array
    right_player_vel: chex.Array

    ball_pos: chex.Array
    ball_vel: chex.Array

@dataclass
class Action:
    move: chex.Array # (y: int -1 to 1, x: int -1 to 1)
    kick: chex.Array # 0=no kick, 1=kick

class FootballGame:
    FIELD_SIZE = (500, 1000) # height, width
    EXTRA_RADIUS = 100

    WINDOW_SIZE = (FIELD_SIZE[0] + 2*EXTRA_RADIUS, FIELD_SIZE[1] + 2*EXTRA_RADIUS)
    WINDOW_SHAPE = (*WINDOW_SIZE, 3)

    FIELD_BOUNDS = ((EXTRA_RADIUS, EXTRA_RADIUS), (EXTRA_RADIUS + FIELD_SIZE[0], EXTRA_RADIUS + FIELD_SIZE[1]))
    CENTER = [WINDOW_SIZE[0] // 2, WINDOW_SIZE[1] // 2]

    BALL_RADIUS = 10
    PLAYER_RADIUS = 15

    GOAL_WIDTH = 200
    GOALPOST_RADIUS = 10

    GOALPOST_Y1 = CENTER[0] - GOAL_WIDTH//2 - GOALPOST_RADIUS
    GOALPOST_Y2 = CENTER[0] + GOAL_WIDTH//2 + GOALPOST_RADIUS

    GOALPOST_TL_POS = jnp.array((GOALPOST_Y1, FIELD_BOUNDS[0][1]), dtype=jnp.int32)
    GOALPOST_BL_POS = jnp.array((GOALPOST_Y2, FIELD_BOUNDS[0][1]), dtype=jnp.int32)
    GOALPOST_TR_POS = jnp.array((GOALPOST_Y1, FIELD_BOUNDS[1][1]), dtype=jnp.int32)
    GOALPOST_BR_POS = jnp.array((GOALPOST_Y2, FIELD_BOUNDS[1][1]), dtype=jnp.int32)

    ZERO_VECTOR = jnp.zeros((2), dtype=jnp.float32)

    PLAYER_START_GOAL_DIST = 100
    LEFT_PLAYER_START_POS = jnp.array((CENTER[0]+1, FIELD_BOUNDS[0][1] + PLAYER_START_GOAL_DIST), dtype=jnp.float32)
    RIGHT_PLAYER_START_POS = jnp.array((CENTER[0]-1, FIELD_BOUNDS[1][1] - PLAYER_START_GOAL_DIST), dtype=jnp.float32)
        # adding a very small offset so x-coords don't line up perfectly and cause issues; temp fix

    BALL_START_POS = jnp.array(CENTER, dtype=jnp.float32)

    PLAYER_ACCEL = 450

    # percent velocity is reduced by per 1s; higher -> more friction
    PLAYER_FRICTION_DAMPENING = 0.97
    BALL_FRICTION_DAMPENING = 0.3

    BALL_MASS = 1
    PLAYER_MASS = 3

    KICK_REACH = 5
    KICK_TOTAL_RANGE = KICK_REACH + PLAYER_RADIUS + BALL_RADIUS
    KICK_IMPULSE_VEL = 250

    def __init__(self, dt=0.1):
        self.DT = dt

    @functools.partial(jax.jit, static_argnames=('self'))
    def reset(self) -> State:
        return State(
            left_player_pos=FootballGame.LEFT_PLAYER_START_POS,
            #left_player_pos=FootballGame.GOALPOST_TL_POS + jnp.array([50,0]),
            left_player_vel=FootballGame.ZERO_VECTOR,

            right_player_pos=FootballGame.RIGHT_PLAYER_START_POS,
            right_player_vel=FootballGame.ZERO_VECTOR,

            ball_pos=FootballGame.BALL_START_POS,
            ball_vel=FootballGame.ZERO_VECTOR,
        )

        # TODO: clipping issue; most easily recreatable from the perfectly aligned starting x-positions
            # temp fix to add a very small offset to the starting positions; still other ways to clip though
            # happens whenever the ball is squeezed between the players

    @functools.partial(jax.jit, static_argnames=('self'))
    def step(self, state: State, left_player_action: Action, right_player_action: Action) -> State:
        ### update velocities first, then positions (semi-implicit Euler method) ###

        left_player_nvel = state.left_player_vel
        right_player_nvel = state.right_player_vel
        ball_nvel = state.ball_vel

        left_player_move = left_player_action.move
        right_player_move = right_player_action.move

        # prevent moving faster in diagonals
        # left_player_move = jnp.nan_to_num(left_player_action.move / jnp.linalg.norm(left_player_action.move))
        # right_player_move = jnp.nan_to_num(right_player_action.move / jnp.linalg.norm(right_player_action.move))

        # update player velocities based on movement actions
        left_player_nvel += left_player_move*FootballGame.PLAYER_ACCEL*self.DT
        right_player_nvel += right_player_move*FootballGame.PLAYER_ACCEL*self.DT

        # apply kick TODO: fix double-kicking?
        left_player_to_ball = state.ball_pos - state.left_player_pos
        left_player_to_ball_dist = jnp.linalg.norm(left_player_to_ball)

        ball_nvel += jax.lax.cond(
            jnp.logical_and(left_player_action.kick, left_player_to_ball_dist <= FootballGame.KICK_TOTAL_RANGE), 
            lambda: FootballGame.KICK_IMPULSE_VEL * left_player_to_ball / left_player_to_ball_dist, 
            lambda: FootballGame.ZERO_VECTOR
        )

        right_player_to_ball = state.ball_pos - state.right_player_pos
        right_player_to_ball_dist = jnp.linalg.norm(right_player_to_ball)

        ball_nvel += jax.lax.cond(
            jnp.logical_and(right_player_action.kick, right_player_to_ball_dist <= FootballGame.KICK_TOTAL_RANGE),
            lambda: FootballGame.KICK_IMPULSE_VEL * right_player_to_ball / right_player_to_ball_dist, 
            lambda: FootballGame.ZERO_VECTOR
        )

        # apply friction, modelled as proportional to velocity (proportional to momentum for simplicity)
        left_player_nvel *= (1 - FootballGame.PLAYER_FRICTION_DAMPENING) ** self.DT
        right_player_nvel *= (1 - FootballGame.PLAYER_FRICTION_DAMPENING) ** self.DT

        ball_nvel *= (1 - FootballGame.BALL_FRICTION_DAMPENING) ** self.DT

        ### update state with new velocities ###

        state = dataclasses.replace(state,
            left_player_vel=left_player_nvel,
            right_player_vel=right_player_nvel,
            ball_vel=ball_nvel
        )

        # update positions
        state = self._physics_step(state)

        return state

    @functools.partial(jax.jit, static_argnames=('self'))
    def _physics_step(self, state: State) -> State:

        positions = jnp.array((
            FootballGame.GOALPOST_TL_POS, FootballGame.GOALPOST_BL_POS, # goalposts
                FootballGame.GOALPOST_TR_POS, FootballGame.GOALPOST_BR_POS,
            state.left_player_pos, state.right_player_pos, # players
            state.ball_pos # ball
        ), dtype=jnp.float32)

        velocities = jnp.array((
            FootballGame.ZERO_VECTOR, FootballGame.ZERO_VECTOR, # goalposts
                FootballGame.ZERO_VECTOR, FootballGame.ZERO_VECTOR,
            state.left_player_vel, state.right_player_vel, # players
            state.ball_vel # ball
        ), dtype=jnp.float32)

        radii = jnp.array((
            FootballGame.GOALPOST_RADIUS, FootballGame.GOALPOST_RADIUS, # goalposts
                FootballGame.GOALPOST_RADIUS, FootballGame.GOALPOST_RADIUS,
            FootballGame.PLAYER_RADIUS, FootballGame.PLAYER_RADIUS, # players
            FootballGame.BALL_RADIUS, # ball
        ), dtype=jnp.int32)

        masses = jnp.array((
            jnp.inf, jnp.inf, jnp.inf, jnp.inf, # goalposts
            FootballGame.PLAYER_MASS, FootballGame.PLAYER_MASS, # players
            FootballGame.BALL_MASS # ball
        ), dtype=jnp.float32)

        colliders = [ positions, velocities, radii, masses ]
        COLLIDER_IS = jnp.arange(len(colliders[0])) # static

        # calculate matrix of collision times between every collider

        coll_t_matrix = jax.vmap(
            lambda cur_i, colliders: jax.vmap(
                FootballGame._calc_coll_t, 
                in_axes=(0, None, None)
            )(COLLIDER_IS, cur_i, colliders),
            
            in_axes=(0, None)
        )(COLLIDER_IS, colliders)

        # iterate: 
        #   - find overall first collision, 
        #   - update positions of all objects to time of min(moment of collision, dt); update cur_t

        #   - if cur_t > DT, no more collisions in time interval -> finished, BREAK
 
        #      ELSE:
        #   - update velocity of both colliding objects using collision
        #   - update collision times of other objects with the 2 colliding objects 
        #       -> in case position/velocity update of cur colliding objects changes another collision
        #      REPEAT

        EPSILON = 0.0001 # stop clipping

        def find_first_coll_and_npos(cur_t, coll_t_matrix, colliders):
             # find the overall first (min time) collision; focus on this collsion in this iteration
            collider_is = jnp.unravel_index(jnp.argmin(coll_t_matrix), coll_t_matrix.shape)
            coll_t = coll_t_matrix[collider_is]

            # return updated position of all colliders to time of min(moment of collision, dt)
            return collider_is, coll_t, FootballGame._calc_new_collider_positions(
                colliders[0], # collider positions
                colliders[1], # collider velocities
                jnp.minimum(coll_t, self.DT) - cur_t # time to update by (dt of cur iteration)
                    - EPSILON # ensure that objects are not colliding after update; slightly apart
            )
        
        def handle_collision(collider_is, coll_t_matrix, colliders):
            # update the 2 collision objects' velocities
            n_vels = FootballGame._calc_collision_response_velocities(
                [ ele[collider_is[0]] for ele in colliders ], 
                [ ele[collider_is[1]] for ele in colliders ]
            )

            colliders[1] = colliders[1].at[collider_is, ].set(n_vels)

            # update collision times of other objects with the 2 colliding objects 
            for i in range(2):
                n_coll_ts = jax.vmap(
                    FootballGame._calc_coll_t, 
                    in_axes=(0, None, None)
                )(COLLIDER_IS, collider_is[i], colliders)

                coll_t_matrix = coll_t_matrix.at[collider_is[i], :].set(n_coll_ts)
                coll_t_matrix = coll_t_matrix.at[:, collider_is[i]].set(n_coll_ts)

            return coll_t_matrix, colliders 

        collider_is, coll_t, colliders[0] = find_first_coll_and_npos(0, coll_t_matrix, colliders)
            # cur_t starts at 0; nothing has been done yet

        # jax.debug.breakpoint()

        def collision_iteration(carry):
            collider_is, cur_t, coll_t_matrix, colliders = carry

            #jax.debug.print("goalpost={pos1} player={pos2}", pos1=colliders[0][0], pos2=colliders[0][4])

            coll_t_matrix, colliders = handle_collision(collider_is, coll_t_matrix, colliders)
            collider_is, coll_t, colliders[0] = find_first_coll_and_npos(cur_t, coll_t_matrix, colliders)

            return collider_is, coll_t, coll_t_matrix, colliders 

        _, _, _, colliders = jax.lax.while_loop(
            cond_fun=lambda carry: carry[1] < self.DT, # when cur_t >= DT, all objects updated to DT -> done
            body_fun=collision_iteration,
            init_val=(collider_is, coll_t, coll_t_matrix, colliders)
        )

        return State(
            left_player_pos=colliders[0][4],
            left_player_vel=colliders[1][4],

            right_player_pos=colliders[0][5],
            right_player_vel=colliders[1][5],

            ball_pos=colliders[0][6],
            ball_vel=colliders[1][6],
        )

    @staticmethod
    def _calc_coll_t(i, j, colliders):
        def f():
            collider1 = [ ele[i] for ele in colliders ]
            collider2 = [ ele[j] for ele in colliders ]

            pos1, vel1, radius1, _ = collider1
            pos2, vel2, radius2, _ = collider2

            return coll_time_moving_circle_circle(pos1, vel1, radius1, pos2, vel2, radius2)

        return jax.lax.cond(i == j, # make group system; matrix of bools for whether to check collision
            lambda: jnp.inf, f)

    @staticmethod
    @functools.partial(jax.vmap, in_axes=(0, 0, None))
    def _calc_new_collider_positions(position, velocity, dt):
        return position + velocity*dt

    @staticmethod
    def _calc_collision_response_velocities(collider1, collider2):
        pos1, vel1, _, mass1 = collider1
        pos2, vel2, _, mass2 = collider2

        dpos = pos1 - pos2
        dvel = vel1 - vel2

        dvel_proj_onto_dpos = jnp.dot(dpos, dvel) / jnp.dot(dpos, dpos) * dpos
        mass_sum = mass1 + mass2

        # infinity / infinity results in NaN; proper limit should be 1 in this case
        mass1_proportion = jnp.nan_to_num(mass1 / mass_sum, nan=1)
        mass2_proportion = jnp.nan_to_num(mass2 / mass_sum, nan=1)

        nvel1 = vel1 - 2 * mass2_proportion * dvel_proj_onto_dpos
        nvel2 = vel2 + 2 * mass1_proportion * dvel_proj_onto_dpos

        # jax.debug.print("collision response vels: {v1}, {v2}", v1=nvel1, v2=nvel2)

        return nvel1, nvel2

    ### RENDERING ###

    BACKGROUND_COLOR = jnp.array((91, 127, 101), dtype=jnp.uint8)
    BORDER_COLOR = jnp.array((255, 255, 255), dtype=jnp.uint8)
    BALL_COLOR = jnp.array((255, 255, 255), dtype=jnp.uint8)
    RIGHT_TEAM_COLOR = jnp.array((255, 0, 0), dtype=jnp.uint8)
    LEFT_TEAM_COLOR = jnp.array((0, 0, 255), dtype=jnp.uint8)
    
    @staticmethod
    @jax.jit
    def _generate_background_image():
        image = jnp.full(FootballGame.WINDOW_SHAPE, FootballGame.BACKGROUND_COLOR, dtype=jnp.uint8)

        # vertical border lines
        image = image.at[FootballGame.FIELD_BOUNDS[0][0]:FootballGame.FIELD_BOUNDS[1][0], FootballGame.FIELD_BOUNDS[0][1]].set(FootballGame.BORDER_COLOR) # left
        image = image.at[FootballGame.FIELD_BOUNDS[0][0]:FootballGame.FIELD_BOUNDS[1][0], FootballGame.FIELD_BOUNDS[1][1]].set(FootballGame.BORDER_COLOR) # right

        # horizontal border lines
        image = image.at[FootballGame.FIELD_BOUNDS[0][0], FootballGame.FIELD_BOUNDS[0][1]:FootballGame.FIELD_BOUNDS[1][1]].set(FootballGame.BORDER_COLOR) # top
        image = image.at[FootballGame.FIELD_BOUNDS[1][0], FootballGame.FIELD_BOUNDS[0][1]:FootballGame.FIELD_BOUNDS[1][1]].set(FootballGame.BORDER_COLOR) # bottom

        # goalposts
        image = draw_circle(image, *FootballGame.GOALPOST_TL_POS, FootballGame.BALL_RADIUS, FootballGame.LEFT_TEAM_COLOR)
        image = draw_circle(image, *FootballGame.GOALPOST_BL_POS, FootballGame.BALL_RADIUS, FootballGame.LEFT_TEAM_COLOR)
        image = draw_circle(image, *FootballGame.GOALPOST_TR_POS, FootballGame.BALL_RADIUS, FootballGame.RIGHT_TEAM_COLOR)
        image = draw_circle(image, *FootballGame.GOALPOST_BR_POS, FootballGame.BALL_RADIUS, FootballGame.RIGHT_TEAM_COLOR)

        return image

    BACKGROUND_IMAGE = None # initilized later, immediately after class declaration

    @functools.partial(jax.jit, static_argnames=('self'))
    def render(self, state: State) -> chex.Array:
        image = FootballGame.BACKGROUND_IMAGE
    
        # add white outline when a player is kicking?
            # size of kick range

        # players
        image = draw_circle(image, *jnp.rint(state.left_player_pos).astype(int), FootballGame.PLAYER_RADIUS, FootballGame.LEFT_TEAM_COLOR)
        image = draw_circle(image, *jnp.rint(state.right_player_pos).astype(int), FootballGame.PLAYER_RADIUS, FootballGame.RIGHT_TEAM_COLOR)

        # ball
        image = draw_circle(image, *jnp.rint(state.ball_pos).astype(int), FootballGame.BALL_RADIUS, FootballGame.BALL_COLOR)

        return image

FootballGame.BACKGROUND_IMAGE = FootballGame._generate_background_image()