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
    LEFT_PLAYER_START_POS = jnp.array((CENTER[0], FIELD_BOUNDS[0][1] + PLAYER_START_GOAL_DIST), dtype=jnp.float32)
    RIGHT_PLAYER_START_POS = jnp.array((CENTER[0], FIELD_BOUNDS[1][1] - PLAYER_START_GOAL_DIST), dtype=jnp.float32)

    BALL_START_POS = jnp.array(CENTER, dtype=jnp.float32)


    PLAYER_ACCEL = 30

    def __init__(self, dt=0.1):
        self.DT = dt

    @functools.partial(jax.jit, static_argnames=('self'))
    def reset(self) -> State:
        return State(
            left_player_pos=FootballGame.LEFT_PLAYER_START_POS,
            left_player_vel=FootballGame.ZERO_VECTOR,

            right_player_pos=FootballGame.RIGHT_PLAYER_START_POS,
            right_player_vel=FootballGame.ZERO_VECTOR,

            ball_pos=FootballGame.BALL_START_POS,
            ball_vel=FootballGame.ZERO_VECTOR,
        )

    @functools.partial(jax.jit, static_argnames=('self'))
    def step(self, state: State, left_player_action: Action, right_player_action: Action) -> State:

        # update velocities first, then positions (semi-implicit Euler method)
        state = dataclasses.replace(state,
            left_player_vel=state.left_player_vel + left_player_action.move*FootballGame.PLAYER_ACCEL*self.DT,
            right_player_vel=state.right_player_vel + right_player_action.move*FootballGame.PLAYER_ACCEL*self.DT
        )

        # update positions
        state = self._physics_step(state)

        return state

    @functools.partial(jax.jit, static_argnames=('self'))
    def _physics_step(self, state: State) -> State:

        positions = jnp.array((
            FootballGame.GOALPOST_TL_POS,
            FootballGame.GOALPOST_BL_POS,
            FootballGame.GOALPOST_TR_POS,
            FootballGame.GOALPOST_BR_POS,

            state.left_player_pos,
            state.right_player_pos,

            state.ball_pos
        ), dtype=jnp.float32)

        velocities = jnp.array((
            FootballGame.ZERO_VECTOR,
            FootballGame.ZERO_VECTOR,
            FootballGame.ZERO_VECTOR,
            FootballGame.ZERO_VECTOR,

            state.left_player_vel,
            state.right_player_vel,

            state.ball_vel
        ), dtype=jnp.float32)

        radii = jnp.array((
            FootballGame.GOALPOST_RADIUS,
            FootballGame.GOALPOST_RADIUS,
            FootballGame.GOALPOST_RADIUS,
            FootballGame.GOALPOST_RADIUS,

            FootballGame.PLAYER_RADIUS,
            FootballGame.PLAYER_RADIUS,

            FootballGame.BALL_RADIUS,
        ), dtype=jnp.int32)

        # masses = jnp.array((

        # ), dtype=jnp.int32)

        circle_colliders = ( positions, velocities, radii )

        # filter list of indexes (of circle_colliders) of colliders that are moving
        moving_is = jnp.arange(len(circle_colliders[0]))[velocities != FootballGame.ZERO_VECTOR]

        # generate indexes of pairs between all and moving colliders
        pairs_all_is, pairs_moving_is_is = jnp.triu_indices(n=len(circle_colliders[0]), m=len(moving_is))
        pairs_moving_is = moving_is[pairs_moving_is_is]

        pairs_all = [ ele[pairs_all_is] for ele in circle_colliders ]
        pairs_moving = [ ele[pairs_moving_is] for ele in circle_colliders ]

        coll_ts = FootballGame.get_coll_t(pairs_all, pairs_moving)

        min_t_pairs_i = jnp.argmin(coll_ts)
        min_t = coll_ts[min_t_pairs_i]
        min_t_i = pairs_moving_is[min_t_pairs_i]

        @functools.partial(jax.lax.while_loop, cond=lambda min_t, *_: min_t >= self.DT,
            init_val=(min_t, min_t_i, circle_colliders, moving_is))
        def f(min_t, min_t_i, circle_colliders, moving_is):
            min_t_collider = [ ele[min_t_i] for ele in circle_colliders ]

            coll_ts = FootballGame.get_coll_t(circle_colliders, pairs_moving)

        
        min_t >= self.DT # finished


        return state

    @staticmethod
    @jax.vmap
    def get_coll_t(circle1, circle2):
        pos1, vel1, radius1 = circle1
        pos2, vel2, radius2 = circle2

        return coll_time_moving_circle_circle(pos1, vel1, radius1, pos2, vel2, radius2)


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

        # players
        image = draw_circle(image, *jnp.rint(state.left_player_pos).astype(int), FootballGame.PLAYER_RADIUS, FootballGame.LEFT_TEAM_COLOR)
        image = draw_circle(image, *jnp.rint(state.right_player_pos).astype(int), FootballGame.PLAYER_RADIUS, FootballGame.RIGHT_TEAM_COLOR)

        # ball
        image = draw_circle(image, *jnp.rint(state.ball_pos).astype(int), FootballGame.BALL_RADIUS, FootballGame.BALL_COLOR)

        return image

FootballGame.BACKGROUND_IMAGE = FootballGame._generate_background_image()