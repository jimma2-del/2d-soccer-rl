import jax
import jax.numpy as jnp
import functools

@functools.partial(jax.jit, static_argnames=["r"])
def draw_circle(image: jnp.ndarray, cy: int, cx: int, r: int, color: jnp.ndarray) -> jnp.ndarray:
    '''Draws a circle onto image. Circle must fully fit inside image.'''

    size = r*2 + 1
    y1 = cy - r
    x1 = cx - r

    # assert y1 >= 0 and y2 <= image.shape[0] and x1 >= 0 and x2 <= image.shape[1], \
    #    "Circle must fully fit inside image."

    # creates 2 grids where values are the y/x positions
    y_vals, x_vals = jnp.mgrid[0:size, 0:size]
    y_vals += y1
    x_vals += x1
    
    # bool circle mask
    circle_mask = ((x_vals - cx)**2 + (y_vals - cy)**2) <= r**2

    # apply circle onto rectanglar section of image
    cur_patch = jax.lax.dynamic_slice(image, (y1, x1, 0), (size, size, 3))
    new_patch = jnp.where(circle_mask[:, :, None], color[None, None, :], cur_patch)

    # apply rectanglar section with circle back onto the original image
    return jax.lax.dynamic_update_slice(image, new_patch, (y1, x1, 0))

@jax.jit
def coll_time_moving_circle_circle(pos1, vel1, radius1, pos2, vel2, radius2):
    '''Returns the earliest future collision time of two moving circles. 
    Returns jnp.inf if there are no valid collision times.'''

    coll_dist = radius1 + radius2

    dpos = pos1 - pos2
    dvel = vel1 - vel2
    dvel_squared = jnp.dot(dvel, dvel)

    discriminant = dvel_squared * coll_dist**2 - jnp.cross(dvel, dpos)**2

    def f():
        discr_sqrt = jnp.sqrt(discriminant)
        mid = -jnp.dot(dpos, dvel)

        t1 = (mid - discr_sqrt) / dvel_squared

        def g(): # if t1 invalid, check if t2 is valid and return if so, otherwise infinity
            t2 = (mid + discr_sqrt) / dvel_squared
            return jnp.where(t2 < 0, jnp.inf, t2)

        # if t1 >= 0, it is valid -> earliest collision time
        return jax.lax.cond(t1 > 0, lambda: t1, g)

    # no real solutions if discriminant < 0; return infinity as the time
        # <= is used instead of < to avoid situations when dvel=0, resulting in NaN
    return jax.lax.cond(discriminant <= 0, lambda: jnp.inf, f)