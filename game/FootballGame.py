from chex import dataclass
import chex
import jax.numpy as jnp
import jax
import functools
import dataclasses

from utils import draw_circle, coll_time_moving_circle_circle, \
    coll_time_line_moving_circle, closest_point_on_line

@dataclass(frozen=True)
class State:
    ball_pos: chex.Array
    ball_vel: chex.Array

    # vectorized; axis 0 is the player index (for multiple players on the same team)
    left_player_pos: chex.Array
    left_player_vel: chex.Array
    right_player_pos: chex.Array
    right_player_vel: chex.Array

@dataclass(frozen=True)
class Action:
    # each field is vectorized; axis 0 is the player index (for multiple players on the same team)
    move: chex.Array # (y: int -1 to 1, x: int -1 to 1)
    kick: chex.Array # 0=no kick, 1=kick

@dataclass(frozen=True)
class Settings:
    field_size: tuple[int, int] = (500, 1000) # height, width
    field_padding: int = 100

    ball_radius: int = 10
    player_radius: int = 15

    goal_width: int = 200
    goalpost_radius: int = 10

    player_start_dist_from_center: int = 200

    player_accel: float = 450

    # percent velocity is reduced by per 1s; higher -> more friction
    player_friction_dampening: float = 0.97
    ball_friction_dampening: float = 0.3

    ball_mass: float = 1#5#1
    player_mass: float = 3

    kick_reach: float = 5
    kick_impulse_vel: float = 250

    # percent velocity is reduced by upon each collision; prevents infinite loops
    collision_dampening: float = 0.1

    players_per_team: int = 2

@dataclass(frozen=True)
class RenderSettings:
    background_color = jnp.array((91, 127, 101), dtype=jnp.uint8)
    border_color = jnp.array((255, 255, 255), dtype=jnp.uint8)
    ball_color = jnp.array((255, 255, 255), dtype=jnp.uint8)
    right_team_color = jnp.array((255, 0, 0), dtype=jnp.uint8)
    left_team_color = jnp.array((0, 0, 255), dtype=jnp.uint8)

class FootballGame:
    ZERO_VECTOR = jnp.zeros((2), dtype=jnp.float32)

    # matrix indicating whether to check collisions between members of the 2 groups 
        # should be diagonally symmetrical
    # groups: disabled, goalposts, players, ball, inner walls, outer walls
        # NOTE: group 0 does not check collisions with anything -> collider disabled
    _COLLIDER_GROUP_CHECK_COLLISIONS = jnp.array((
        ( False, False, False, False, False, False ),
        ( False, False, True,  True,  False, False ),
        ( False, True,  True,  True,  False, True  ),
        ( False, True,  True,  True,  True,  False ),
        ( False, False, False, True,  False, False ),
        ( False, False, True,  False, False, False )
    ), dtype=jnp.bool)

    def __init__(self, dt=0.1, settings=Settings(), render_settings=RenderSettings()):
        self.DT = dt
        self._render_settings = render_settings
        self.update_settings(settings)

    @dataclass
    class CachedConsts:
        window_size: tuple[int, int] = None
        window_shape: tuple[int, int, int] = None
        field_bounds: tuple[tuple[int, int], tuple[int, int]] = None
        center: list[int, int] = None
        goalpost_center_dist: int = None
        goalpost_y1: int = None
        goalpost_y2: int = None
        goalpost_tl_pos: chex.Array = None
        goalpost_bl_pos: chex.Array = None
        goalpost_tr_pos: chex.Array = None
        goalpost_br_pos: chex.Array = None
        left_player_start_pos: chex.Array = None
        right_player_start_pos: chex.Array = None
        ball_start_pos: chex.Array = None
        kick_total_range: float = None
        kick_total_range_with_ball: float = None

        line_colliders: tuple[chex.Array, chex.Array, chex.Array] = None

    def update_settings(self, settings: Settings):
        self._settings = settings

        # recompute cached consts
        self._cached_consts = self.CachedConsts()

        self._cached_consts.window_size = (
            self._settings.field_size[0] + 2*self._settings.field_padding, 
            self._settings.field_size[1] + 2*self._settings.field_padding
        )

        self._cached_consts.window_shape = (*self._cached_consts.window_size, 3)

        self._cached_consts.field_bounds = (
            (self._settings.field_padding, self._settings.field_padding), 
            (
                self._settings.field_padding + self._settings.field_size[0], 
                self._settings.field_padding + self._settings.field_size[1]
            )
        )

        self._cached_consts.center = [
            self._cached_consts.window_size[0] // 2, 
            self._cached_consts.window_size[1] // 2
        ]

        self._cached_consts.goalpost_center_dist = self._settings.goal_width//2 + self._settings.goalpost_radius
        self._cached_consts.goalpost_y1 = self._cached_consts.center[0] - self._cached_consts.goalpost_center_dist
        self._cached_consts.goalpost_y2 = self._cached_consts.center[0] + self._cached_consts.goalpost_center_dist

        self._cached_consts.goalpost_tl_pos = jnp.array(
            (self._cached_consts.goalpost_y1, self._cached_consts.field_bounds[0][1]), dtype=jnp.int32)
        self._cached_consts.goalpost_bl_pos = jnp.array(
            (self._cached_consts.goalpost_y2, self._cached_consts.field_bounds[0][1]), dtype=jnp.int32)
        self._cached_consts.goalpost_tr_pos = jnp.array(
            (self._cached_consts.goalpost_y1, self._cached_consts.field_bounds[1][1]), dtype=jnp.int32)
        self._cached_consts.goalpost_br_pos = jnp.array(
            (self._cached_consts.goalpost_y2, self._cached_consts.field_bounds[1][1]), dtype=jnp.int32)

        # previous temp fix: adding a very small y offset to starting positions
            # so y-coords don't line up perfectly and cause issues

        player_start_y_gap = self._settings.field_size[0] / (self._settings.players_per_team + 1)

        left_player_start_x = self._cached_consts.center[1] - self._settings.player_start_dist_from_center
        right_player_start_x = self._cached_consts.center[1] + self._settings.player_start_dist_from_center

        self._cached_consts.left_player_start_pos = jnp.array([(
            self._cached_consts.field_bounds[0][0] + (i + 1)*player_start_y_gap, 
            left_player_start_x
        ) for i in range(self._settings.players_per_team) ], dtype=jnp.float32)

        self._cached_consts.right_player_start_pos = jnp.array([(
            self._cached_consts.field_bounds[0][0] + (i + 1)*player_start_y_gap, 
            right_player_start_x
        ) for i in range(self._settings.players_per_team) ], dtype=jnp.float32)

        self._cached_consts.ball_start_pos = jnp.array(self._cached_consts.center, dtype=jnp.float32)

        self._cached_consts.kick_total_range = self._settings.kick_reach + self._settings.player_radius
        self._cached_consts.kick_total_range_with_ball = self._cached_consts.kick_total_range \
            + self._settings.ball_radius
        
        self._cached_consts.line_colliders = self._generate_line_colliders()

        self._recompute_render_cached_consts()

    def get_settings(self):
        return self._settings
    
    # def get_cached_consts(self):
    #     return self._cached_consts

    @functools.partial(jax.jit, static_argnames=('self'))
    def reset(self) -> State:
        return State(
            left_player_pos=self._cached_consts.left_player_start_pos,
            left_player_vel=jnp.zeros((self._settings.players_per_team, 2), dtype=jnp.float32),

            right_player_pos=self._cached_consts.right_player_start_pos,
            right_player_vel=jnp.zeros((self._settings.players_per_team, 2), dtype=jnp.float32),

            ball_pos=self._cached_consts.ball_start_pos,
            ball_vel=jnp.array((0,0))#FootballGame.ZERO_VECTOR,
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

        left_player_move = jnp.clip(left_player_action.move, -1, 1)
        right_player_move = jnp.clip(right_player_action.move, -1, 1)

        # prevent moving faster in diagonals
        # left_player_move = jnp.nan_to_num(left_player_action.move / jnp.linalg.norm(left_player_action.move))
        # right_player_move = jnp.nan_to_num(right_player_action.move / jnp.linalg.norm(right_player_action.move))

        # update player velocities based on movement actions
        left_player_nvel += left_player_move*self._settings.player_accel*self.DT
        right_player_nvel += right_player_move*self._settings.player_accel*self.DT

        # apply kick TODO: fix double-kicking?
        @functools.partial(jax.vmap, in_axes=(0, 0, None))
        def calc_kick_impulse_vel(kick, player_pos, ball_pos):
            player_to_ball = ball_pos - player_pos
            player_to_ball_dist = jnp.linalg.norm(player_to_ball)

            apply_kick = jnp.logical_and(kick, player_to_ball_dist <= self._cached_consts.kick_total_range_with_ball)
            kick_impulse_vel = self._settings.kick_impulse_vel * player_to_ball / player_to_ball_dist

            return apply_kick * kick_impulse_vel
        
        ball_nvel += jnp.sum(calc_kick_impulse_vel(left_player_action.kick, state.left_player_pos, state.ball_pos), axis=0)
        ball_nvel += jnp.sum(calc_kick_impulse_vel(right_player_action.kick, state.right_player_pos, state.ball_pos), axis=0)

        # apply friction, modelled as proportional to velocity (proportional to momentum for simplicity)
        left_player_nvel *= (1 - self._settings.player_friction_dampening) ** self.DT
        right_player_nvel *= (1 - self._settings.player_friction_dampening) ** self.DT

        ball_nvel *= (1 - self._settings.ball_friction_dampening) ** self.DT

        ### update state with new velocities ###

        state = dataclasses.replace(state,
            left_player_vel=left_player_nvel,
            right_player_vel=right_player_nvel,
            ball_vel=ball_nvel
        )

        ### update positions ###
        state = self._physics_step(state)


        # check for goal
        goal = 0 # -1=went in left goal; +1=went in right goal

        height_between_goalposts = jnp.logical_and(
            state.ball_pos[0] >= self._cached_consts.goalpost_y1,
            state.ball_pos[0] <= self._cached_consts.goalpost_y2
        )

        goal -= jnp.logical_and(height_between_goalposts,
            state.ball_pos[1] < self._cached_consts.field_bounds[0][1] - self._settings.ball_radius)
        goal += jnp.logical_and(height_between_goalposts,
            state.ball_pos[1] > self._cached_consts.field_bounds[1][1] + self._settings.ball_radius)

        return state, goal
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def _make_circle_colliders(self, state: State):
        goalpost_pos = jnp.vstack((self._cached_consts.goalpost_tl_pos,
            self._cached_consts.goalpost_bl_pos, self._cached_consts.goalpost_tr_pos, self._cached_consts.goalpost_br_pos))
            # static

        positions = jnp.vstack((goalpost_pos, state.left_player_pos, state.right_player_pos, state.ball_pos ))

        goalpost_vels = jnp.zeros((4, 2), dtype=jnp.float32) # static
        velocities = jnp.vstack(( goalpost_vels, state.left_player_vel, state.right_player_vel, state.ball_vel ))

        ### static elements ##

        masses = jnp.array((
            *([jnp.inf] * 4), # goalposts
            *([self._settings.player_mass] * (2*self._settings.players_per_team)), # players
            self._settings.ball_mass # ball
        ), dtype=jnp.float32)

        radii = jnp.array((
            *([self._settings.goalpost_radius] * 4), # goalposts
            *([self._settings.player_radius] * (2*self._settings.players_per_team)), # players
            self._settings.ball_radius # ball
        ), dtype=jnp.int32)

        groups = jnp.array(( # determines whether to check collisions between objects
            *([1] * 4), # goalposts
            *([2] * (2*self._settings.players_per_team)), # players
            3 # ball
        ), dtype=jnp.int8)

        circle_colliders = [ positions, velocities, masses, radii, groups ]

        return circle_colliders #[ ele[4:5] for ele in circle_colliders ]
    
    @functools.partial(jax.jit, static_argnames=('self'))
    def _generate_line_colliders(self):
        line_p1s = jnp.array((
            # field (inner, ball) walls
            (self._cached_consts.field_bounds[0][0], self._cached_consts.field_bounds[0][1]),
            (self._cached_consts.goalpost_y2, self._cached_consts.field_bounds[0][1]),

            (self._cached_consts.field_bounds[0][0], self._cached_consts.field_bounds[1][1]),
            (self._cached_consts.goalpost_y2, self._cached_consts.field_bounds[1][1]),

            (self._cached_consts.field_bounds[0][0], self._cached_consts.field_bounds[0][1]),
            (self._cached_consts.field_bounds[1][0], self._cached_consts.field_bounds[0][1]),

            # window (outer, player) walls
            (0, 0),
            (0, self._cached_consts.window_size[1]),
            (0, 0),
            (self._cached_consts.window_size[0], 0),
        ), dtype=jnp.int32)

        line_p2s = jnp.array((
            # field (inner, ball) walls
            (self._cached_consts.goalpost_y1, self._cached_consts.field_bounds[0][1]),
            (self._cached_consts.field_bounds[1][0], self._cached_consts.field_bounds[0][1]),

            (self._cached_consts.goalpost_y1, self._cached_consts.field_bounds[1][1]),
            (self._cached_consts.field_bounds[1][0], self._cached_consts.field_bounds[1][1]),

            (self._cached_consts.field_bounds[0][0], self._cached_consts.field_bounds[1][1]),
            (self._cached_consts.field_bounds[1][0], self._cached_consts.field_bounds[1][1]),

            # window (outer, player) walls
            (self._cached_consts.window_size[0], 0),
            (self._cached_consts.window_size[0], self._cached_consts.window_size[1]),
            (0, self._cached_consts.window_size[1]),
            (self._cached_consts.window_size[0], self._cached_consts.window_size[1]),
        ), dtype=jnp.int32)

        line_groups = jnp.array(( # determines whether to check collisions between objects
            4, 4, 4, 4, 4, 4, # field (inner, ball) walls
            5, 5, 5, 5 # window (outer, player) walls
        ), dtype=jnp.int8)

        line_colliders = ( line_p1s, line_p2s, line_groups )

        return line_colliders #[ ele[4:5] for ele in line_colliders ]

    @functools.partial(jax.jit, static_argnames=('self'))
    def _physics_step(self, state: State) -> State:
        circle_colliders = self._make_circle_colliders(state)
        CIRCLE_COLLIDER_IS = jnp.arange(len(circle_colliders[0])) # static

        # jax.debug.print("p2: pos={pos} vel={vel}", pos=circle_colliders[0][5], vel=circle_colliders[1][5], ordered=True)

        # calculate matrix of collision times between every circle collider and every collider
        coll_t_matrix = jax.vmap(FootballGame._calc_circle_coll_ts, in_axes=(0, None, None))(
            CIRCLE_COLLIDER_IS, circle_colliders, self._cached_consts.line_colliders)

        # iterate: 
        #   - find overall first collision, 
        #   - update positions of all objects to time of min(moment of collision, dt); update cur_t

        #   - if cur_t > DT, no more collisions in time interval -> finished, BREAK
 
        #      ELSE:
        #   - update velocity of both colliding objects using collision
        #   - update collision times of other objects with the 2 colliding objects 
        #       -> in case position/velocity update of cur colliding objects changes another collision
        #      REPEAT

        EPSILON = 0.00001 # stop clipping

        def find_first_coll_and_npos(cur_t, coll_t_matrix, circle_colliders):
            # find the overall first (min time) collision; focus on this collsion in this iteration
            collider_is = jnp.unravel_index(jnp.argmin(coll_t_matrix), coll_t_matrix.shape)
            coll_t = coll_t_matrix[collider_is]

            # put lower i (circle) 1st and higher i (possibly line) 2nd to allow separate processing of lines
            # collider_is = jnp.array(collider_is)
            # collider_is = jnp.array(( jnp.min(collider_is), jnp.max(collider_is) ))

            # return updated position of all circle colliders to time of min(moment of collision, dt)
            return collider_is, coll_t, FootballGame._calc_new_collider_positions(
                circle_colliders[0], # collider positions
                circle_colliders[1], # collider velocities
                jnp.maximum(0, jnp.minimum(coll_t - EPSILON, self.DT) - cur_t) # time to update by (dt of cur iteration)
                    # subtract epsilon to ensure that objects are not colliding after update -> slightly apart
            )
        
        def handle_collision(cur_t, collider_is, coll_t_matrix, circle_colliders):
            # colliders are either 2 circles, or circle and line

            # first collider is always a circle
            collider1 = [ ele[collider_is[0]] for ele in circle_colliders ]

            # second collider can be a circle or a line
            collider2 = jax.lax.cond(collider_is[1] < len(circle_colliders[0]),
                lambda: [ circle_colliders[i][collider_is[1]] for i in range(3) ], # circle
                lambda: [
                    closest_point_on_line(collider1[0], 
                        [ ele[collider_is[1] - len(circle_colliders[0])] for ele in self._cached_consts.line_colliders]), 
                    FootballGame.ZERO_VECTOR, 
                    jnp.inf
                ] # line; find collision point and create mock circle collider
            )

            # update the 2 collision objects' velocities
            n_vels = FootballGame._calc_collision_response_velocities(collider1, collider2)

            # damping of velocity; simulates energy lost & helps prevent infinite loops
            n_vels = (
                n_vels[0] * (1 - self._settings.collision_dampening), 
                n_vels[1] * (1 - self._settings.collision_dampening)
            )

            # jax.debug.print("collision: {collider_is} t={cur_t} n_vel1={n_vel1} n_vel2={n_vel2}", 
            #     collider_is=jnp.array(collider_is), cur_t=cur_t, n_vel1=n_vels[0], n_vel2=n_vels[1], ordered=True)

            circle_colliders[1] = circle_colliders[1].at[collider_is[0]].set(n_vels[0])
            circle_colliders[1] = jax.lax.cond(collider_is[1] < len(circle_colliders[0]),
                lambda: circle_colliders[1].at[collider_is[1]].set(n_vels[1]), lambda: circle_colliders[1])
                # update second collider's velocity if not a line

            # update collision times of other objects with the 2 colliding objects 
            def update_collision_times(collider_i, cur_t, coll_t_matrix):
                n_coll_ts = FootballGame._calc_circle_coll_ts(
                    collider_i, circle_colliders, self._cached_consts.line_colliders) + cur_t

                coll_t_matrix = coll_t_matrix.at[collider_i, :].set(n_coll_ts)
                coll_t_matrix = coll_t_matrix.at[:, collider_i].set(n_coll_ts[:len(circle_colliders[0])])

                return coll_t_matrix
            
            coll_t_matrix = update_collision_times(collider_is[0], cur_t, coll_t_matrix)
            coll_t_matrix = jax.lax.cond(collider_is[1] < len(circle_colliders[0]),
                lambda: update_collision_times(collider_is[1], cur_t, coll_t_matrix), lambda: coll_t_matrix)
                # update second collider's collisions if not a line 
                    # NOTE: vmapped cond executes both branches; this is fine since no errors are thrown

            return coll_t_matrix, circle_colliders 

        collider_is, coll_t, circle_colliders[0] = find_first_coll_and_npos(0, coll_t_matrix, circle_colliders)
            # cur_t starts at 0; nothing has been done yet

        # jax.debug.breakpoint()

        def collision_iteration(carry):
            collider_is, cur_t, coll_t_matrix, circle_colliders = carry

            #jax.debug.print("goalpost={pos1} player={pos2}", pos1=colliders[0][0], pos2=colliders[0][4])

            coll_t_matrix, circle_colliders = handle_collision(cur_t, collider_is, coll_t_matrix, circle_colliders)
            collider_is, coll_t, circle_colliders[0] = find_first_coll_and_npos(cur_t, coll_t_matrix, circle_colliders)

            return collider_is, coll_t, coll_t_matrix, circle_colliders 

        _, _, _, circle_colliders = jax.lax.while_loop(
            cond_fun=lambda carry: carry[1] < self.DT, # when cur_t >= DT, all objects updated to DT -> done
            body_fun=collision_iteration,
            init_val=(collider_is, coll_t, coll_t_matrix, circle_colliders)
        )

        # static
        left_player_start_i = 4
        right_player_start_i = left_player_start_i + self._settings.players_per_team
        players_end_i = right_player_start_i + self._settings.players_per_team


        return State(
            left_player_pos=circle_colliders[0][left_player_start_i:right_player_start_i],
            left_player_vel=circle_colliders[1][left_player_start_i:right_player_start_i],

            right_player_pos=circle_colliders[0][right_player_start_i:players_end_i],
            right_player_vel=circle_colliders[1][right_player_start_i:players_end_i],

            ball_pos=circle_colliders[0][players_end_i],
            ball_vel=circle_colliders[1][players_end_i],
        )

    @staticmethod
    def _calc_circle_coll_ts(i, circle_colliders, line_colliders):
        '''Find the time of collision for the circle collider at the given index 
        with all other colliders.'''

        CIRCLE_COLLIDER_IS = jnp.arange(len(circle_colliders[0])) # static
        LINE_COLLIDER_IS = jnp.arange(len(line_colliders[0])) # static

        circle_coll_ts = jax.vmap(
            FootballGame._calc_circle_circle_coll_t, 
            in_axes=(None, 0, None)
        )(i, CIRCLE_COLLIDER_IS, circle_colliders)

        line_coll_ts = jax.vmap(
            FootballGame._calc_circle_line_coll_t, 
            in_axes=(None, 0, None, None)
        )(i, LINE_COLLIDER_IS, circle_colliders, line_colliders)

        return jnp.concat((circle_coll_ts, line_coll_ts))

    @staticmethod
    def _calc_circle_circle_coll_t(i, j, circle_colliders):

        def f():
            collider1 = [ ele[i] for ele in circle_colliders ]
            pos1, vel1, _, radius1, *_ = collider1

            collider2 = [ ele[j] for ele in circle_colliders ]
            pos2, vel2, _, radius2, *_ = collider2

            return coll_time_moving_circle_circle(pos1, vel1, radius1, pos2, vel2, radius2)

        group1 = circle_colliders[4][i]
        group2 = circle_colliders[4][j]

        # only check collision if
        return jax.lax.cond(jnp.logical_and(i != j, # colliders are not the same
                # collider groups check collision between each other
                FootballGame._COLLIDER_GROUP_CHECK_COLLISIONS[group1][group2]), 
            f, lambda: jnp.inf) # NOTE: vmap in jax does not support cond; executes both branches
    
    @staticmethod
    def _calc_circle_line_coll_t(circle_i, line_i, circle_colliders, line_colliders):
        # NOTE: infinite loop bug when pushed into wall by another circle; fixed?

        def f():
            collider1 = [ ele[circle_i] for ele in circle_colliders ]
            pos1, vel1, _, radius1, *_ = collider1
            
            collider2 = [ ele[line_i] for ele in line_colliders ]
            return coll_time_line_moving_circle(collider2, pos1, vel1, radius1)

        circle_group = circle_colliders[4][circle_i]
        line_group = line_colliders[2][line_i]

        # only check collision if collider groups check collision between each other
        return jax.lax.cond(FootballGame._COLLIDER_GROUP_CHECK_COLLISIONS[circle_group][line_group], 
            f, lambda: jnp.inf) # NOTE: vmap in jax does not support cond; executes both branches

    @staticmethod
    @functools.partial(jax.vmap, in_axes=(0, 0, None))
    def _calc_new_collider_positions(position, velocity, dt):
        return position + velocity*dt

    @staticmethod
    def _calc_collision_response_velocities(collider1, collider2, epsilon=0.0001):
        pos1, vel1, mass1, *_ = collider1
        pos2, vel2, mass2, *_ = collider2

        dpos = pos1 - pos2
        dvel = vel1 - vel2

        dvel_proj_onto_dpos = jnp.dot(dpos, dvel) / jnp.dot(dpos, dpos) * dpos
        mass_sum = mass1 + mass2

        # infinity / infinity results in NaN; proper limit should be 1 in this case
        mass1_proportion = jnp.nan_to_num(mass1 / mass_sum, nan=1)
        mass2_proportion = jnp.nan_to_num(mass2 / mass_sum, nan=1)

        vel_change1 = 2 * mass2_proportion * dvel_proj_onto_dpos
        vel_change2 = 2 * mass1_proportion * dvel_proj_onto_dpos

        # nvel1 = vel1 - vel_change1
        # nvel2 = vel2 + vel_change2

        # apply velocity in direction going away from other collider; prevents getting stuck
        nvel1 = vel1 + jnp.sign(dpos)*jnp.abs(vel_change1)
        nvel2 = vel2 - jnp.sign(dpos)*jnp.abs(vel_change2)

        # round small values to zero to prevent infinite loops
        nvel1 = jnp.where(jnp.abs(nvel1) < epsilon, 0, nvel1)
        nvel2 = jnp.where(jnp.abs(nvel2) < epsilon, 0, nvel2)

        # jax.debug.print("collision response vels: {v1}, {v2}", v1=nvel1, v2=nvel2)

        return nvel1, nvel2

    ### RENDERING ###

    def update_render_settings(self, render_settings: RenderSettings):
        self._render_settings = render_settings
        self._recompute_render_cached_consts()

    def get_render_settings(self):
        return self._render_settings
        
    def _recompute_render_cached_consts(self):
        self._cached_background_image = self._generate_background_image()

    @functools.partial(jax.jit, static_argnames=('self'))
    def _generate_background_image(self):
        image = jnp.full(self._cached_consts.window_shape, self._render_settings.background_color, dtype=jnp.uint8)

        # vertical border lines
        image = image.at[self._cached_consts.field_bounds[0][0]:self._cached_consts.field_bounds[1][0], 
                         self._cached_consts.field_bounds[0][1]].set(self._render_settings.border_color) # left
        image = image.at[self._cached_consts.field_bounds[0][0]:self._cached_consts.field_bounds[1][0], 
                         self._cached_consts.field_bounds[1][1]].set(self._render_settings.border_color) # right

        # horizontal border lines
        image = image.at[self._cached_consts.field_bounds[0][0], 
                         self._cached_consts.field_bounds[0][1]:self._cached_consts.field_bounds[1][1]] \
            .set(self._render_settings.border_color) # top
        
        image = image.at[self._cached_consts.field_bounds[1][0], 
                         self._cached_consts.field_bounds[0][1]:self._cached_consts.field_bounds[1][1]] \
            .set(self._render_settings.border_color) # bottom

        # goalposts
        image = draw_circle(image, 
            *self._cached_consts.goalpost_tl_pos, self._settings.ball_radius, self._render_settings.left_team_color)
        image = draw_circle(image, 
            *self._cached_consts.goalpost_bl_pos, self._settings.ball_radius, self._render_settings.left_team_color)
        image = draw_circle(image, 
            *self._cached_consts.goalpost_tr_pos, self._settings.ball_radius, self._render_settings.right_team_color)
        image = draw_circle(image, 
            *self._cached_consts.goalpost_br_pos, self._settings.ball_radius, self._render_settings.right_team_color)

        return image

    @functools.partial(jax.jit, static_argnames=('self'))
    def render(self, state: State, left_player_kicking=None, right_player_kicking=None) -> chex.Array:
        if left_player_kicking is None: left_player_kicking = jnp.zeros(self._settings.players_per_team, dtype=jnp.int8)
        if right_player_kicking is None: right_player_kicking = jnp.zeros(self._settings.players_per_team, dtype=jnp.int8)

        image = self._cached_background_image

        for i in range(self._settings.players_per_team):
            # white outline when a player is kicking, size of kick range
            image = jax.lax.cond(left_player_kicking[i], 
                lambda: draw_circle(image, *jnp.rint(state.left_player_pos[i]).astype(int), 
                                    self._cached_consts.kick_total_range, self._render_settings.ball_color),
                lambda: image
            )
        
            image = jax.lax.cond(right_player_kicking[i], 
                lambda: draw_circle(image, *jnp.rint(state.right_player_pos[i]).astype(int), 
                                    self._cached_consts.kick_total_range, self._render_settings.ball_color),
                lambda: image
            )

            # players
            image = draw_circle(image, *jnp.rint(state.left_player_pos[i]).astype(int), 
                                self._settings.player_radius, self._render_settings.left_team_color)
            image = draw_circle(image, *jnp.rint(state.right_player_pos[i]).astype(int), 
                                self._settings.player_radius, self._render_settings.right_team_color)

        # ball
        image = draw_circle(image, *jnp.rint(state.ball_pos).astype(int), 
                            self._settings.ball_radius, self._render_settings.ball_color)

        return image