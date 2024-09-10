from flax import struct 
import jax
import jax.numpy as jnp




@struct.dataclass
class Point:
    x: jnp.float32
    y: jnp.float32
    def create_point(x:float, y:float):
        return Point(x, y)

@struct.dataclass
class Wall:
    p1: Point
    p2: Point
    def create_wall(p1:Point, p2:Point):
        return Wall(p1, p2)

@struct.dataclass
class Ray:
    ray_origin: Point
    ray_direction: Point # cosines and sines of the angle
    ray_length: jnp.float32
    def create_ray(ray_origin:Point, ray_direction:Point, ray_length:float):
        return Ray(ray_origin, ray_direction, ray_length)



@struct.dataclass
class Space:
    x_min: jnp.float32
    x_max: jnp.float32
    y_min: jnp.float32
    y_max: jnp.float32
    torous: bool
    walls: Wall
    def create_space(x_min, x_max, y_min, y_max, torous, wall_begins:Point, wall_ends:Point):
        x_min = x_min
        x_max = x_max
        y_min = y_min
        y_max = y_max
        torous = torous
        walls = jax.vmap(Wall.create_wall)(wall_begins, wall_ends)
        return Space(x_min, x_max, y_min, y_max, torous, walls)
    




@struct.dataclass
class NetworkSpace:
    pass