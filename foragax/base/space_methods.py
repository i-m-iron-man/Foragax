
from base.space_classes import *
import jax
import jax.numpy as jnp





def check_collision(walls: Wall, old_position:Point, new_position:Point):
    def check_one_wall(wall : Wall, old_position:Point, new_position:Point):

        def on_segment(p:Point, q:Point, r:Point):
            cond = jnp.logical_and(q.x <= jnp.maximum(p.x, r.x), 
                                   jnp.logical_and(q.x >= jnp.minimum(p.x, r.x), 
                                    jnp.logical_and(q.y <= jnp.maximum(p.y, r.y), q.y >= jnp.minimum(p.y, r.y))))
            return cond
    
        def orientation(p:Point, q:Point, r:Point):
            val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
            return jax.lax.select(val == 0, 0, jax.lax.select(val > 0, 1, 2))
        
        def do_intersect(wall : Wall, old_position:Point, new_position:Point):
            p1 = wall.p1
            q1 = wall.p2
            p2 = old_position
            q2 = new_position
            o1 = orientation(p1, q1, p2)
            o2 = orientation(p1, q1, q2)
            o3 = orientation(p2, q2, p1)
            o4 = orientation(p2, q2, q1)
            
            c1 = jnp.logical_and(o1 != o2, o3 != o4)
            c2 = jnp.logical_and(o1 == 0, on_segment(p1, p2, q1))
            c3 = jnp.logical_and(o2 == 0, on_segment(p1, q2, q1))
            c4 = jnp.logical_and(o3 == 0, on_segment(p2, p1, q2))
            c5 = jnp.logical_and(o4 == 0, on_segment(p2, q1, q2))
            return jnp.logical_or(jnp.logical_or(c1, c2), jnp.logical_or(c3, jnp.logical_or(c4, c5)))
        
        return do_intersect(wall, old_position, new_position)
    
    return jnp.any(jax.vmap(check_one_wall, in_axes=(0, None, None))(walls, old_position, new_position)) # any rertuns true if any wall intersects with the particle
jit_check_collision = jax.jit(check_collision)

def wall_generator(wall_array:jnp.array):# wall_array = [[[wall1_begin(x,y)], [wall1_end(x,y)]], [[wall2_begin(x,y)], [wall2_end(x,y)], ...]]])
    x_wall_begins = jnp.array([wall_array[i][0][0] for i in range(len(wall_array))])
    y_wall_begins = jnp.array([wall_array[i][0][1] for i in range(len(wall_array))])
    x_wall_ends = jnp.array([wall_array[i][1][0] for i in range(len(wall_array))])
    y_wall_ends = jnp.array([wall_array[i][1][1] for i in range(len(wall_array))])
    wall_points_begins = jax.vmap(Point.create_point)(x_wall_begins, y_wall_begins)
    wall_points_ends = jax.vmap(Point.create_point)(x_wall_ends, y_wall_ends)
    #walls = jax.vmap(Wall.create_wall)(wall_points_begins, wall_points_ends)
    return wall_points_begins, wall_points_ends

RESOLUTION = 9
def ray_generator(agent_pos, ray_span, ray_length):
    x,y,angle = agent_pos
    ray_angles = jnp.linspace(angle - ray_span, angle + ray_span, RESOLUTION)

    cos_ray_angles = jnp.cos(ray_angles)
    sin_ray_angles = jnp.sin(ray_angles)
    ray_directions = jax.vmap(Point.create_point)(cos_ray_angles, sin_ray_angles)

    ray_origin = Point.create_point(x, y)
    
    rays = jax.vmap(Ray.create_ray, in_axes=(None, 0, None))(ray_origin, ray_directions, ray_length)

    return rays
jit_ray_generator = jax.jit(ray_generator)

def ray_agent_sensor(ray, data):
    agent_x, agent_y, agent_rad, agent_active_status = data
    def ray_active_agent_sensing(ray, agent_x, agent_y, agent_rad):
        ray_origin = jnp.array([ray.ray_origin.x, ray.ray_origin.y])
        ray_origin = jnp.reshape(ray_origin, (2,))
        ray_direction = jnp.array([ray.ray_direction.x, ray.ray_direction.y])
        ray_direction = jnp.reshape(ray_direction, (2,))
        obj_center = jnp.array([agent_x, agent_y])
        obj_radius = agent_rad
        s = ray_origin - obj_center
        b = jnp.dot(s, ray_direction)
        c = jnp.dot(s, s) - obj_radius**2
        h = b**2 - c
        h = jax.lax.cond(h < 0, lambda _: -1.0, lambda _: jnp.sqrt(h), None)
        t = jax.lax.cond(h >= 0, lambda _: -b - h, lambda _: ray.ray_length, None)
        t = jax.lax.cond(t < 0, lambda _: ray.ray_length, lambda _: t, None)
        return jnp.minimum(t, ray.ray_length)

        
    def ray_inactive_agent_sensing():
        return ray.ray_length
    distance = jax.lax.cond(agent_active_status, lambda _: ray_active_agent_sensing(ray, agent_x, agent_y, agent_rad), lambda _: ray_inactive_agent_sensing(), None)
    return distance
jit_ray_agent_sensor = jax.jit(ray_agent_sensor)

def ray_wall_sensor(ray, data):
    wall_p1_x, wall_p1_y, wall_p2_x, wall_p2_y = data
    def line_intercept(p1:jnp.array, q1:jnp.array, p2:jnp.array, q2:jnp.array):
        d1 = jnp.cross(q1 - p1, p2 - p1)
        d2 = jnp.cross(q1 - p1, q2 - p1)
        r = d1 / ((d1 - d2)+1e-6)
        intersection = p2 + r * (q2 - p2)
        return jnp.linalg.norm(intersection - p1)
        
    def on_segment(p:jnp.array, q:jnp.array, r:jnp.array):
        cond = jnp.logical_and(q[0] <= jnp.maximum(p[0], r[0]), 
                                jnp.logical_and(q[0] >= jnp.minimum(p[0], r[0]), 
                                jnp.logical_and(q[1] <= jnp.maximum(p[1], r[1]), q[1] >= jnp.minimum(p[1], r[1])))
                            )
        return cond
    def orientation(p:jnp.array, q:jnp.array, r:jnp.array):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        return jax.lax.select(val == 0, 0, jax.lax.select(val > 0, 1, 2))
        
    def do_intersect(ray:Ray, wall_p1_x, wall_p1_y, wall_p2_x, wall_p2_y):
        p1 = jnp.array([ray.ray_origin.x, ray.ray_origin.y])
        p1 = jnp.reshape(p1, (2,))
        q1 = jnp.array([ray.ray_origin.x + ray.ray_direction.x * ray.ray_length, ray.ray_origin.y + ray.ray_direction.y * ray.ray_length])
        q1 = jnp.reshape(q1, (2,))
        p2 = jnp.array([wall_p1_x, wall_p1_y])
        q2 = jnp.array([wall_p2_x, wall_p2_y])
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)
            
        c1 = jnp.logical_and(o1 != o2, o3 != o4)
        d1 = jax.lax.cond(c1, lambda _: line_intercept(p1, q1, p2, q2), lambda _: ray.ray_length, None)
            
        c2 = jnp.logical_and(o1 == 0, on_segment(p1, p2, q1))
        d2 = jax.lax.cond(c2, lambda _: jnp.linalg.norm(p1 - p2), lambda _: ray.ray_length, None)
            
        c3 = jnp.logical_and(o2 == 0, on_segment(p1, q2, q1))
        d3 = jax.lax.cond(c3, lambda _: jnp.linalg.norm(p1 - q2), lambda _: ray.ray_length, None)
            
        c4 = jnp.logical_and(o3 == 0, on_segment(p2, p1, q2))
        d4 = jax.lax.cond(c4, lambda _: 0.0, lambda _: ray.ray_length, None)

        c5 = jnp.logical_and(o4 == 0, on_segment(p2, q1, q2))
        d5 = jax.lax.cond(c5, lambda _: jnp.linalg.norm(p1 - q1), lambda _: ray.ray_length, None)
        return jnp.minimum(d1, jnp.minimum(d2, jnp.minimum(d3, jnp.minimum(d4, d5))))
    return do_intersect(ray, wall_p1_x, wall_p1_y, wall_p2_x, wall_p2_y)
jit_ray_wall_sensor = jax.jit(ray_wall_sensor)


def create_space(x_min, x_max, y_min, y_max, torous, wall_array):
    x_min = x_min
    x_max = x_max
    y_min = y_min
    y_max = y_max
    torous = torous
    if wall_array is None:
        return Space(x_min, x_max, y_min, y_max, torous, None)
    wall_begin_points, wall_end_points = wall_generator(wall_array)
    walls = jax.vmap(Wall.create_wall)(wall_begin_points, wall_end_points)
    return Space(x_min, x_max, y_min, y_max, torous, walls)