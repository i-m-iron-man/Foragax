# in this version of the foraging world the agnent's angle is not an independent DOF.

import sys
sys.path.insert(1, '/home/siddhart/source/Foragax_v2')
from base.agent_classes import *
from base.agent_methods import *
from base.space_methods import *
from base.space_classes import *
from agent.forager_v3 import *
from agent.resource_v2 import *
import jax
import jax.numpy as jnp
from flax import struct
#from flax.training import checkpoints
from flax.training import orbax_utils
import orbax


@struct.dataclass
class forager_position:
    X: jnp.array
    X_vel: jnp.array
    
    Y: jnp.array
    Y_vel: jnp.array

    Ang: jnp.array

    id: jnp.int32

@struct.dataclass
class resource_position:
    X: jnp.array
    Y: jnp.array
    id: jnp.int32

def spawn_forager_pos(forager_set:Agent_Set, space:ContinuousSpace, key:jax.random.PRNGKey):
    '''Agent positions are spawned in the space'''
    foragers = forager_set.agents
    key, *spawning_keys = jax.random.split(key, foragers.active_state.shape[0] + 1)
    spawning_keys = jnp.array(spawning_keys)
    ids = jnp.arange(forager_set.num_total_agents)

    def spawn_one_forager_pos(spawning_key, active_state, id, space):
        
        def spawn_active_forager_pos(spawning_key, id, space):
            key, *spawning_keys = jax.random.split(spawning_key, 5)
            X = jax.random.uniform(spawning_keys[0], (1,), minval = space.x_min, maxval = space.x_max)
            X_vel = jax.random.uniform(spawning_keys[1], (1,), minval = -1.0, maxval = 1.0)
            Y = jax.random.uniform(spawning_keys[2], (1,), minval = space.y_min, maxval = space.y_max)
            Y_vel = jax.random.uniform(spawning_keys[3], (1,), minval = -1.0, maxval = 1.0)
            Ang = jnp.arctan2(Y_vel, X_vel)
            id = id
            return forager_position(X, X_vel, Y, Y_vel, Ang, id)
        
        def spawn_inactive_forager_pos(id):
            X = jnp.array([0.0])
            X_vel = jnp.array([0.0])
            Y = jnp.array([0.0])
            Y_vel = jnp.array([0.0])
            Ang = jnp.array([0.0])
            id = id
            return forager_position(X, X_vel, Y, Y_vel, Ang, id)
        
        forager_pos = jax.lax.cond(active_state, lambda _: spawn_active_forager_pos(spawning_key, id, space), lambda _: spawn_inactive_forager_pos(id), None)
        return forager_pos
    
    forager_positions = jax.vmap(spawn_one_forager_pos, in_axes=(0, 0, 0, None))(spawning_keys, foragers.active_state, ids, space)
    
    return forager_positions, key

jit_spawn_forager_pos = jax.jit(spawn_forager_pos)

def update_forager_pos(forager_poses:forager_position, foragers:Forager, space:ContinuousSpace):
    '''Update the positions of the agents in the space'''

    def update_one_forager_pos(forager_pos, forager_state, forager_params, active_state, walls):

        def update_active_forager_pos(forager_pos, forager_state, forager_params, walls):
            X = forager_pos.X + forager_params.content['dt']*forager_pos.X_vel
            Y = forager_pos.Y + forager_params.content['dt']*forager_pos.Y_vel
            
            X_vel = forager_pos.X_vel + forager_params.content['dt']*forager_state.content['X_acc']
            Y_vel = forager_pos.Y_vel + forager_params.content['dt']*forager_state.content['Y_acc']

            #check if the forager is hitting the wall
            old_point = Point(forager_pos.X[0], forager_pos.Y[0])
            new_point = Point(X[0], Y[0])
            collision = jit_check_collision(walls, old_point, new_point)
            #id = forager_pos.id
            (X, Y, X_vel, Y_vel) = jax.lax.cond(collision, lambda _: (forager_pos.X, forager_pos.Y, -forager_pos.X_vel, -forager_pos.Y_vel), lambda _: (X, Y, X_vel, Y_vel), None)
            
            Ang = jnp.arctan2(Y_vel, X_vel)
            return forager_pos.replace(X = X, Y = Y, X_vel = X_vel, Y_vel = Y_vel, Ang = Ang)
        
        def update_inactive_forager_pos(forager_pos):
            return forager_pos
        
        forager_pos = jax.lax.cond(active_state, lambda _: update_active_forager_pos(forager_pos, forager_state, forager_params, walls), lambda _: update_inactive_forager_pos(forager_pos), None)
        return forager_pos

    forager_poses = jax.vmap(update_one_forager_pos, in_axes=(0, 0, 0, 0, None))(forager_poses, foragers.state, foragers.params, foragers.active_state, space.walls)
    return forager_poses

jit_update_forager_pos = jax.jit(update_forager_pos)

def spawn_resource_pos(resource_set:Agent_Set, space:ContinuousSpace, key:jax.random.PRNGKey):
    '''Agent positions are spawned in the space'''
    resources = resource_set.agents
    key, *spawning_keys = jax.random.split(key, resources.active_state.shape[0] + 1)
    spawning_keys = jnp.array(spawning_keys)
    ids = jnp.arange(resource_set.num_total_agents)

    def spawn_one_resource_pos(spawning_key, active_state, id, space):
        
        def spawn_active_resource_pos(spawning_key, id, space):
            key, *spawning_keys = jax.random.split(spawning_key, 3)
            X = jax.random.uniform(spawning_keys[0], (1,), minval = space.x_min, maxval = space.x_max)
            Y = jax.random.uniform(spawning_keys[1], (1,), minval = space.y_min, maxval = space.y_max)
            id = id
            return resource_position(X, Y, id)
        
        def spawn_inactive_resource_pos(id):
            X = jnp.array([0.0])
            Y = jnp.array([0.0])
            id = id
            return resource_position(X, Y, id)
        
        resource_pos = jax.lax.cond(active_state, lambda _: spawn_active_resource_pos(spawning_key, id, space), lambda _: spawn_inactive_resource_pos(id), None)
        return resource_pos
    
    resource_positions = jax.vmap(spawn_one_resource_pos, in_axes=(0, 0, 0, None))(spawning_keys, resources.active_state, ids, space)
    
    return resource_positions, key

jit_spawn_resource_pos = jax.jit(spawn_resource_pos)

def remove_forager_pos(forager_poses, indexes_to_remove, num_to_remove):
    def remove_pos(idx, forager_poses__indexes_to_remove):
        forager_poses, indexes_to_remove = forager_poses__indexes_to_remove

        #could have used jax.tree_util.tree_map but don't want to change the id of the agent_poses

        X = forager_poses.X.at[indexes_to_remove[idx]].set(jnp.array([0.0]))
        X_vel = forager_poses.X_vel.at[indexes_to_remove[idx]].set(jnp.array([0.0]))
        Y = forager_poses.Y.at[indexes_to_remove[idx]].set(jnp.array([0.0]))
        Y_vel = forager_poses.Y_vel.at[indexes_to_remove[idx]].set(jnp.array([0.0]))
        Ang = forager_poses.Ang.at[indexes_to_remove[idx]].set(jnp.array([0.0]))
        
        new_forager_poses = forager_poses.replace(X = X, X_vel = X_vel, Y = Y, Y_vel = Y_vel, Ang = Ang)
        return new_forager_poses, indexes_to_remove
    
    forager_poses, _ = jax.lax.fori_loop(0, num_to_remove, remove_pos, (forager_poses, indexes_to_remove))
    return forager_poses
jit_remove_forager_pos = jax.jit(remove_forager_pos)

def remove_resource_pos(resource_poses, indexes_to_remove, num_to_remove):
    def remove_pos(idx, resource_poses__indexes_to_remove):
        resource_poses, indexes_to_remove = resource_poses__indexes_to_remove

        #could have used jax.tree_util.tree_map but don't want to change the id of the agent_poses

        X = resource_poses.X.at[indexes_to_remove[idx]].set(jnp.array([0.0]))
        Y = resource_poses.Y.at[indexes_to_remove[idx]].set(jnp.array([0.0]))

        new_resource_poses = resource_poses.replace(X = X, Y = Y)
        return new_resource_poses, indexes_to_remove
    
    resource_poses, _ = jax.lax.fori_loop(0, num_to_remove, remove_pos, (resource_poses, indexes_to_remove))
    return resource_poses
jit_remove_resource_pos = jax.jit(remove_resource_pos)

def sort_forager_pos(forager_poses, sorting_ids):
    '''Sort the forager_poses based on the id'''
    new_forager_poses = jax.tree_util.tree_map(lambda x: jnp.take(x, sorting_ids, axis = 0), forager_poses)
    return new_forager_poses
jit_sort_forager_pos = jax.jit(sort_forager_pos)

def sort_resource_pos(resource_poses, sorting_ids):
    '''Sort the resource_poses based on the id'''
    new_resource_poses = jax.tree_util.tree_map(lambda x: jnp.take(x, sorting_ids, axis = 0), resource_poses)
    return new_resource_poses
jit_sort_resource_pos = jax.jit(sort_resource_pos)


def add_forager_pos(forager_poses, num_agents_add, add_param, space, key):
    def add_one_pos(idx, forager_poses__add_param__space__key):
        forager_poses, add_param, space, key = forager_poses__add_param__space__key
        
        good_ids = add_param.content['good_ids']
        num_active_foragers = add_param.content['num_active_agents']
        forager_to_copy = jax.tree_util.tree_map(lambda x: x[good_ids[idx]], forager_poses)
        forager_to_add_idx = idx + num_active_foragers
        
        key, *noise_keys = jax.random.split(key, 5)
        
        X_to_set = jnp.minimum(forager_to_copy.X + jax.random.uniform(noise_keys[0], (1,), minval = -1.0, maxval = 1.0), space.x_max)
        X_to_set = jnp.maximum(X_to_set, space.x_min)
        X = forager_poses.X.at[forager_to_add_idx].set(X_to_set)#(forager_to_copy.X + jax.random.uniform(noise_keys[0], (1,), minval = -0.1, maxval = 0.1))
        
        #X_vel = forager_poses.X_vel.at[forager_to_add_idx].set(forager_to_copy.X_vel + jax.random.uniform(noise_keys[1], (1,), minval = -0.1, maxval = 0.1))
        X_vel = forager_poses.X_vel.at[forager_to_add_idx].set(jax.random.uniform(noise_keys[1], (1,), minval = -1.0, maxval = 1.0))

        Y_to_set = jnp.minimum(forager_to_copy.Y + jax.random.uniform(noise_keys[2], (1,), minval = -1.0, maxval = 1.0), space.y_max)
        Y_to_set = jnp.maximum(Y_to_set, space.y_min)
        Y = forager_poses.Y.at[forager_to_add_idx].set(Y_to_set)#(forager_to_copy.Y + jax.random.uniform(noise_keys[2], (1,), minval = -0.1, maxval = 0.1))
        

        #Y_vel = forager_poses.Y_vel.at[forager_to_add_idx].set(forager_to_copy.Y_vel + jax.random.uniform(noise_keys[3], (1,), minval = -0.1, maxval = 0.1))
        Y_vel = forager_poses.Y_vel.at[forager_to_add_idx].set(jax.random.uniform(noise_keys[3], (1,), minval = -1.0, maxval = 1.0))
        Ang = jnp.arctan2(Y_vel, X_vel)

        new_forager_poses = forager_poses.replace(X = X, X_vel = X_vel, Y = Y, Y_vel = Y_vel, Ang = Ang)

        return new_forager_poses, add_param, space, key

    
    forager_poses, add_params, space, key = jax.lax.fori_loop(0, num_agents_add, add_one_pos, (forager_poses, add_param, space, key))
    return forager_poses, key

jit_add_forager_pos = jax.jit(add_forager_pos)


def num_res_to_add(growth_rate:float, decay_rate:float, num_resources:int, num_resources_max:int):
    '''Number of resources to add to the environment'''
    num_resources_to_add = jax.lax.cond(num_resources < num_resources_max, lambda _: jnp.int32(growth_rate*num_resources - decay_rate*num_resources**2), lambda _: 0, None)
    return num_resources_to_add
jit_num_res_to_add = jax.jit(num_res_to_add)


def add_res_pos(res_poses, num_agents_add, add_param, space, key):
    def add_one_pos(idx, res_poses__add_param__space__key):
        res_poses, add_param, space, key = res_poses__add_param__space__key
        
        good_ids = add_param.content['good_ids']
        num_active_res = add_param.content['num_active_agents']
        res_to_copy = jax.tree_util.tree_map(lambda x: x[good_ids[idx]], res_poses)
        res_to_add_idx = idx + num_active_res
        
        def active_res_to_copy(res_to_copy, space, key):
            key, *noise_keys = jax.random.split(key, 3)
            X_to_set = jnp.minimum(res_to_copy.X + jax.random.uniform(noise_keys[0], (1,), minval = -2.0, maxval = 2.0), space.x_max)
            X_to_set = jnp.maximum(X_to_set, space.x_min)
            X = res_poses.X.at[res_to_add_idx].set(X_to_set)#(res_to_copy.X + jax.random.uniform(noise_keys[0], (1,), minval = -2.0, maxval = 2.0))
        
            Y_to_set = jnp.minimum(res_to_copy.Y + jax.random.uniform(noise_keys[1], (1,), minval = -2.0, maxval = 2.0), space.y_max)
            Y_to_set = jnp.maximum(Y_to_set, space.y_min)
            Y = res_poses.Y.at[res_to_add_idx].set(Y_to_set)#set(res_to_copy.Y + jax.random.uniform(noise_keys[1], (1,), minval = -2.0, maxval = 2.0))
            return X, Y, key
        
        def inactive_res_to_copy(space, key):
            key, *noise_keys = jax.random.split(key, 3)
            X = res_poses.X.at[res_to_add_idx].set(jax.random.uniform(noise_keys[0], (1,), minval = space.x_min, maxval = space.x_max))
            Y = res_poses.Y.at[res_to_add_idx].set(jax.random.uniform(noise_keys[1], (1,), minval = space.y_min, maxval = space.y_max))
            return X, Y, key
        
        #active_cond = jnp.logical_and(res_to_copy.X[0] > 0.0, res_to_copy.Y[0] > 0.0)
        X, Y, key = jax.lax.cond(idx < num_active_res, 
                                 lambda _: active_res_to_copy(res_to_copy, space, key), lambda _: inactive_res_to_copy(space, key), None)

        new_res_poses = res_poses.replace(X = X, Y = Y)

        return new_res_poses, add_param, space, key

    
    res_poses, add_params, space, key = jax.lax.fori_loop(0, num_agents_add, add_one_pos, (res_poses, add_param, space, key))
    return res_poses, key
jit_add_res_pos = jax.jit(add_res_pos)

def forager_resource_interaction(resources:Resource, forager_poses: forager_position, resource_poses: resource_position):
    
    def one_forager_all_res_dist(forager_x, forager_y, res_xs, res_ys):
        return jnp.linalg.norm(jnp.stack((res_xs - forager_x, res_ys - forager_y), axis = 1), axis = 1)
       
    foragers_res_dist_mat = jax.vmap(one_forager_all_res_dist, in_axes=(0, 0, None, None), out_axes=0)(forager_poses.X, forager_poses.Y, resource_poses.X, resource_poses.Y)

    '''
    since the values of all inactive resources are zeros, we don't have to to anything 
    in the following section regarding the inactive resources. similarly we cill not be adding energy to the inactive foragers
    even if we calculate the energy flow from the resources to them.
    '''
    
    def one_forager_all_res_energy(dist_mat_row, all_res_vals, all_res_rads):
        
        # hyperparam for neuroevolution
        def one_agent_one_res_energy(dist, val, rad):
            return jax.lax.cond(jnp.logical_and(dist[0] < rad[0], val[0] > 0.01), lambda _: jnp.minimum(10.0, val[0]/jnp.maximum(dist[0],0.1)), lambda _: 0.0, None)
        return jax.vmap(one_agent_one_res_energy, in_axes=(0, 0, 0), out_axes=0)(dist_mat_row, all_res_vals, all_res_rads)
    
    energy_matrix = jax.vmap(one_forager_all_res_energy, in_axes=(0, None, None), out_axes=0)(foragers_res_dist_mat, resources.state.content['value'], resources.state.content['radius'])
    energy_in = jnp.sum(energy_matrix, axis = 1)
    energy_out = jnp.sum(energy_matrix, axis = 0)
    
    
    def one_forager_all_forager_dist(forager_xs, forager_ys, forager_x, forager_y):
        return jnp.linalg.norm(jnp.stack((forager_xs - forager_x, forager_ys - forager_y), axis = 1), axis = 1)
    
    foragers_forager_dist_mat = jax.vmap(one_forager_all_forager_dist, in_axes=(None, None, 0, 0))(forager_poses.X, forager_poses.Y, forager_poses.X, forager_poses.Y)

    foragers_dist_mat = jnp.concatenate((foragers_forager_dist_mat, foragers_res_dist_mat), axis = 1)
        
    
    
    
    return foragers_dist_mat, energy_in, energy_out

jit_forager_resource_interaction = jax.jit(forager_resource_interaction)


def sensor_model(foragers:Forager, forager_poses: forager_position, resources:Resource, resource_poses: resource_position, walls:Wall, dist_mat: jnp.array):
    agent_xs = jnp.concatenate([forager_poses.X, resource_poses.X])
    agent_xs = jnp.reshape(agent_xs, (-1,))

    agent_ys = jnp.concatenate([forager_poses.Y, resource_poses.Y])
    agent_ys = jnp.reshape(agent_ys, (-1,))

    agent_rads = jnp.concatenate([foragers.state.content['radius'], resources.state.content['radius']])
    agent_rads = jnp.reshape(agent_rads, (-1,))

    agent_types = jnp.concatenate([foragers.agent_type, resources.agent_type])
    agent_types = jnp.reshape(agent_types, (-1,))

    agent_active_status = jnp.concatenate([foragers.active_state, resources.active_state])
    agent_active_status = agent_active_status.reshape(-1)

    data1 = jnp.concatenate([agent_xs, walls.p1.x])
    data2 = jnp.concatenate([agent_ys, walls.p1.y])
    data3 = jnp.concatenate([agent_rads, walls.p2.x])
    data4 = jnp.concatenate([agent_active_status, walls.p2.y])
    entity_data = (data1, data2, data3, data4)

    entity_types = jnp.concatenate([agent_types, jnp.zeros_like(walls.p1.x)])

    env_energy = jnp.concatenate([foragers.state.content['value'], resources.state.content['value']])
    env_energy = jnp.reshape(env_energy, (-1,))
    env_energy = jnp.concatenate([env_energy, jnp.zeros_like(walls.p1.x)])

    env_states = jnp.vstack([env_energy, entity_types])


    agent_wall_dist_mat_zero_pad = jnp.zeros((foragers.unique_id.shape[0], walls.p1.x.shape[0], 1))
    dist_mat = jnp.concatenate((dist_mat, agent_wall_dist_mat_zero_pad), axis = 1)

    foragers_poses = (forager_poses.X, forager_poses.Y, forager_poses.Ang)
    foragers_active_states = foragers.active_state

    def for_each_forager(foragers_active_state, forager_pos, dist_row, entity_data, entity_types, env_states):
        
        def for_each_active_forager(forager_pos, dist_row, entity_data, entity_types, env_states):
            
            ray_span = jnp.pi/8
            ray_max_length = 30.0
            rays = jit_ray_generator(forager_pos, ray_span, ray_max_length)

            def for_each_ray(ray, dist_row, entity_data, entity_types, env_states):
                
                def for_each_entity(entity_data_row, dist_element, entity_type, ray):
                    
                    def ray_agent_sensor_wrap(entity_data_row, dist_element, ray):
                        # radius of the entity_agent is at entity_data_row[2], 0-> agent x, 1-> agent y, 2-> agent radius, 3-> agent active status
                        def ray_agent_sensor_near(ray, entity_data_row):
                            return jit_ray_agent_sensor(ray, entity_data_row)
                        def ray_agent_sensor_far(ray):
                            return ray.ray_length
                        intercept = jax.lax.cond(dist_element[0] < ray.ray_length + entity_data_row[2], lambda _: ray_agent_sensor_near(ray, entity_data_row), lambda _: ray_agent_sensor_far(ray), None)
                        return intercept
                
                    def ray_wall_sensor_wrap(entity_data_row, ray):
                        return jit_ray_wall_sensor(ray, entity_data_row)
                
                    intercept_dist = jax.lax.cond(entity_type == 0, lambda _: ray_wall_sensor_wrap(entity_data_row, ray), lambda _: ray_agent_sensor_wrap(entity_data_row, dist_element, ray), None)
                    return intercept_dist
                    
                intercept_dists = jax.vmap(for_each_entity, in_axes=(0, 0, 0, None))(entity_data, dist_row, entity_types, ray)
                min_intercept_dist = jnp.min(intercept_dists)
                min_intercept_index = jnp.argmin(intercept_dists)
                min_intercept_index = jax.lax.cond(min_intercept_dist < ray.ray_length, lambda _: min_intercept_index, lambda _: -1, None)
                
                def get_sensor_data(env_states_row, min_intercept_index):
                    return jax.lax.cond(min_intercept_index < 0, lambda _:0.0, lambda _: env_states_row[min_intercept_index], None)
                sensor_data = jax.vmap(get_sensor_data, in_axes=(0, None))(env_states, min_intercept_index)
                #print(sensor_data.shape)
                obs_per_ray = jnp.concatenate((sensor_data, jnp.array([min_intercept_dist])))

                return obs_per_ray 

            active_forager_obs = jax.vmap(for_each_ray, in_axes=(0, None, None, None, None))(rays, dist_row, entity_data, entity_types, env_states)
            return active_forager_obs 
            
        def for_each_inactive_forager(env_states):
            return jnp.zeros((RESOLUTION,env_states.shape[0]+1)) # returns zeropadding for inactive foragers shape= num_rays * num states to sense +1, +1 for distance of intercept
        
        sensor_data_per_forager = jax.lax.cond(foragers_active_state == 1, lambda _: for_each_active_forager(forager_pos, dist_row, entity_data, entity_types, env_states), lambda _: for_each_inactive_forager(env_states), None)
        sensor_data_per_forager = jnp.reshape(sensor_data_per_forager, (-1,))
        return sensor_data_per_forager
    
    sensor_data = jax.vmap(for_each_forager, in_axes=(0, 0, 0, None, None, None))(foragers_active_states, foragers_poses, dist_mat, entity_data, entity_types, env_states)
    return sensor_data

jit_sensor_model = jax.jit(sensor_model)

def Neuroevolution(foragers:Forager, forager_poses: forager_position, resources:Resource, resource_poses: resource_position, space:ContinuousSpace, 
                    key:jax.random.PRNGKey, agent_growth_params):
    def select_energy(agents, select_params):
        value_arr = jnp.reshape(agents.state.content['value'], (-1,))
        cond = jnp.logical_and(agents.active_state == 1.0, value_arr < select_params.content['energy_threshold'])
        return cond
    
    select_params = Params(content={'energy_threshold': 1.0})
    num_foragers_remove, remove_indx = jit_select_agents(select_energy, select_params, foragers)

    def remove_foragers(foragers, forager_poses, remove_indx, num_agents_remove):
        remove_params = Params(content={'remove_ids': remove_indx})
        
        foragers = jit_remove_agents(Forager.remove_agent, foragers, num_agents_remove, remove_params)
        forager_poses = jit_remove_forager_pos(forager_poses, remove_indx, num_agents_remove)
        
        foragers, foragers_sorted_index = jit_sort_agents(foragers.active_state, False, foragers)
        forager_poses = jit_sort_forager_pos(forager_poses, foragers_sorted_index)
        
        return foragers, forager_poses
        
    foragers, forager_poses = jax.lax.cond(num_foragers_remove > 0, lambda _: remove_foragers(foragers, forager_poses, remove_indx, num_foragers_remove), 
                                           lambda _: (foragers, forager_poses), None)


    #step 2: remove resources whose value is less than 0.5
    select_params = Params(content={'energy_threshold': 0.5})
    num_res_remove, remove_indx = jit_select_agents(select_energy, select_params, resources)

    def remove_resources(resources, resource_poses, remove_indx, num_agents_remove):
        remove_params = Params(content={'remove_ids': remove_indx})
        
        resources = jit_remove_agents(Resource.remove_agent, resources, num_agents_remove, remove_params)
        resource_poses = jit_remove_resource_pos(resource_poses, remove_indx, num_agents_remove)
        
        resources, resources_sorted_index = jit_sort_agents(resources.active_state, False, resources)
        resource_poses = jit_sort_resource_pos(resource_poses, resources_sorted_index)
        
        return resources, resource_poses
    resources, resource_poses = jax.lax.cond(num_res_remove > 0, lambda _: remove_resources(resources, resource_poses, remove_indx, num_res_remove), 
                                             lambda _: (resources, resource_poses), None)
    
    #step 3: add foragers and resources

    growth_rate, decay_rate, max_resource, max_foragers = agent_growth_params

    select_params = Params(content={'time_threshold': 3.0})
    def select_time(foragers, select_params):
        time_arr = jnp.reshape(foragers.state.content['time_above_repr_thr'], (-1,))
        cond = jnp.logical_and(foragers.active_state == 1.0, time_arr > select_params.content['time_threshold'])
        return cond
    
    num_foragers_add, add_indx = jit_select_agents(select_time, select_params, foragers)
    num_active_foragers = jnp.sum(foragers.active_state, dtype=jnp.int32)
    num_foragers_add = jnp.minimum(num_foragers_add, max_foragers - num_active_foragers)
    
    def add_forager(foragers, forager_poses, add_indx, num_agents_add, num_active_agents, space, key):
        add_params = Params(content={'good_ids': add_indx, 'num_active_agents': num_active_agents})
        foragers, key = jit_add_agents(Forager.add_agent, foragers, num_agents_add, add_params, key)
        forager_poses, key = jit_add_forager_pos(forager_poses, num_agents_add, add_params, space, key)
        return foragers, forager_poses, key

    foragers, forager_poses, key = jax.lax.cond(num_foragers_add > 0, 
                                                       lambda _: add_forager(foragers, forager_poses, add_indx, num_foragers_add, num_active_foragers, space, key), 
                                                       lambda _: (foragers, forager_poses, key), None)
    
    num_active_res = jnp.sum(resources.active_state, dtype=jnp.int32)
    num_res_to_add = jit_num_res_to_add(growth_rate, decay_rate, num_active_res, max_resource)
    num_res_to_add = jnp.minimum(num_res_to_add, max_resource - num_active_res)

    resources, resources_sorted_index = jit_sort_agents(resources.state.content['value'], False, resources)
    resource_poses = jit_sort_resource_pos(resource_poses, resources_sorted_index)
    sorted_index = jnp.sort(resources_sorted_index) 
    
    def add_resource(resources, resource_poses, num_agents_add, num_active_agents, sorted_index, space, key):
        add_params = Params(content={'good_ids': sorted_index, 'num_active_agents': num_active_agents})
        resources, key = jit_add_agents(Resource.add_agent, resources, num_agents_add, add_params, key)
        resource_poses, key = jit_add_res_pos(resource_poses, num_agents_add, add_params, space, key)
        return resources, resource_poses, key
    
    resources, resource_poses, key = jax.lax.cond(num_res_to_add > 0, 
                                                  lambda _: add_resource(resources, resource_poses, num_res_to_add, num_active_res, sorted_index, space, key), 
                                                  lambda _: (resources, resource_poses, key), None)

    return foragers, forager_poses, resources, resource_poses, key

jit_neuroevolution = jax.jit(Neuroevolution)

@struct.dataclass
class Rec_Data:
    # res data
    res_xs: jnp.ndarray
    res_ys: jnp.ndarray
    res_rads: jnp.ndarray
    res_active_status: jnp.ndarray

    #forager data
    foragers_xs: jnp.ndarray
    foragers_ys: jnp.ndarray
    foragers_angs: jnp.ndarray
    foragers_rads: jnp.ndarray
    foragers_active_status: jnp.ndarray

def init_rec_data(frames:int, resources:Agent_Set, res_pos:resource_position, foragers:Agent_Set, foragers_pos:forager_position):
    res_xs = jnp.zeros((frames, resources.num_total_agents, 1))
    res_ys = jnp.zeros((frames, resources.num_total_agents, 1))
    res_rads = jnp.zeros((frames, resources.num_total_agents, 1))
    res_active_status = jnp.zeros((frames, resources.num_total_agents))

    foragers_xs = jnp.zeros((frames, foragers.num_total_agents, 1))
    foragers_ys = jnp.zeros((frames, foragers.num_total_agents, 1))
    foragers_angs = jnp.zeros((frames, foragers.num_total_agents, 1))
    foragers_rads = jnp.zeros((frames, foragers.num_total_agents, 1))
    foragers_active_status = jnp.zeros((frames, foragers.num_total_agents))

    res_xs = res_xs.at[0].set(res_pos.X)
    res_ys = res_ys.at[0].set(res_pos.Y)
    res_rads = res_rads.at[0].set(resources.agents.state.content['radius'])
    res_active_status = res_active_status.at[0].set(resources.agents.active_state)

    foragers_xs = foragers_xs.at[0].set(foragers_pos.X)
    foragers_ys = foragers_ys.at[0].set(foragers_pos.Y)
    foragers_angs = foragers_angs.at[0].set(foragers_pos.Ang)
    foragers_rads = foragers_rads.at[0].set(foragers.agents.state.content['radius'])
    foragers_active_status = foragers_active_status.at[0].set(foragers.agents.active_state)

    rec_data = Rec_Data(res_xs, res_ys, res_rads, res_active_status, 
                        foragers_xs, foragers_ys, foragers_angs, foragers_rads, foragers_active_status)
    return rec_data
jit_init_rec_data = jax.jit(init_rec_data)

def record_render_data(res:Resource, res_pos:resource_position, foragers:Forager, forager_pos:forager_position, step:int, rec_data:Rec_Data):
    # record resource data
    res_xs = rec_data.res_xs.at[step].set(res_pos.X)
    res_ys = rec_data.res_ys.at[step].set(res_pos.Y)
    res_rads = rec_data.res_rads.at[step].set(res.state.content['radius'])
    res_active_status = rec_data.res_active_status.at[step].set(res.active_state)
    
    
    foragers_xs = rec_data.foragers_xs.at[step].set(forager_pos.X)
    foragers_ys = rec_data.foragers_ys.at[step].set(forager_pos.Y)
    foragers_angs = rec_data.foragers_angs.at[step].set(forager_pos.Ang)
    foragers_rads = rec_data.foragers_rads.at[step].set(foragers.state.content['radius'])
    foragers_active_status = rec_data.foragers_active_status.at[step].set(foragers.active_state)

    rec_data = Rec_Data(res_xs, res_ys, res_rads, res_active_status, foragers_xs, foragers_ys, foragers_angs, foragers_rads, foragers_active_status)
    
    return rec_data
jit_record_render_data = jax.jit(record_render_data)

class Foraging_World:

    Foragers: Agent_Set
    Forager_poses: forager_position
    Resources: Agent_Set
    Resource_poses: resource_position
    Space: ContinuousSpace
    Sim_steps: jnp.int32
    key: jax.random.PRNGKey
    max_record_steps: jnp.int16

    def __init__(self, foragers_params:dict, resources_params:dict, space_params:dict, sim_steps: jnp.int32, max_record_steps: jnp.int16, key=jax.random.PRNGKey(2)):
        self.Sim_steps = sim_steps
        key, subkey = jax.random.split(key)

        self.Foragers = Agent_Set(agent = Forager, num_total_agents=foragers_params['total_num'],
                                  num_active_agents=foragers_params['active_num'], agent_type = foragers_params['type'])
        
        foragers_create_params = Params(content={'policy_params':
                                                 {
                                                    'num_neurons': foragers_params['num_neurons'],
                                                    'num_obs': foragers_params['num_obs'],
                                                    'num_actions': foragers_params['num_actions'],
                                                    'dt': foragers_params['dt'],
                                                    'action_scale': foragers_params['action_scale'],
                                                    'deterministic': foragers_params['deterministic']
                                                 },
                                                 'dt': foragers_params['dt'],
                                                 'repr_thresh': foragers_params['repr_thresh'],
                                                 'cooling_period': foragers_params['cooling_period']
                                                 })
        self.Foragers.agents = create_agents(self.Foragers, foragers_create_params, subkey)

        self.Resources = Agent_Set(agent = Resource, num_total_agents=resources_params['total_num'],
                                  num_active_agents=resources_params['active_num'], agent_type = resources_params['type'])
        
        resources_create_params = Params(content={'dt': resources_params['dt']
                                                 })
        key, subkey = jax.random.split(key)
        self.Resources.agents = create_agents(self.Resources, resources_create_params, subkey)

        self.Space = create_cont_space( x_min = space_params['x_min'], 
                                        x_max = space_params['x_max'],
                                        y_min = space_params['y_min'],
                                        y_max = space_params['y_max'],
                                        torous = space_params['torous'],
                                        wall_array = space_params['wall_array']
                                        )
        key, subkey = jax.random.split(key)
        self.Forager_poses, key = spawn_forager_pos(self.Foragers, self.Space, subkey)
        self.key, subkey = jax.random.split(key)
        self.Resource_poses, key = spawn_resource_pos(self.Resources, self.Space, subkey)

        self.max_resource_population = resources_params['total_num'] - 10
        print("max_resource_population", self.max_resource_population)
        self.resource_population_growth_rate = self.max_resource_population/10000.0
        print("resource_population_growth_rate", self.resource_population_growth_rate)
        self.resource_population_death_rate = self.resource_population_growth_rate/self.max_resource_population
        print("resource_population_death_rate", self.resource_population_death_rate)
        self.max_record_steps = max_record_steps
        
        self.rec_data = init_rec_data(self.max_record_steps, self.Resources, self.Resource_poses, self.Foragers, self.Forager_poses)

        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        rec_data_dict = {'rec_data': self.rec_data}
        self.save_args = orbax_utils.save_args_from_target(rec_data_dict)
        options = orbax.checkpoint.CheckpointManagerOptions(max_to_keep=100, create=True)
        self.checkpoint_manager = orbax.checkpoint.CheckpointManager('/Users/siddarth.chaturvedi/Desktop/source/foragax/Foragax_v4/src/worlds/orbax_data',
                                                                     self.orbax_checkpointer, options)

    
    def step(self):
        dist_mat, energy_in, energy_out = jit_forager_resource_interaction(self.Resources.agents, self.Forager_poses, self.Resource_poses)
        obs = jit_sensor_model(self.Foragers.agents, self.Forager_poses, self.Resources.agents, self.Resource_poses, self.Space.walls, dist_mat)
        
        #update internal states of the agents
        forager_input = Signal(content = {'obs': obs, 'energy_in': energy_in})
        resource_input = Signal(content = {'energy_out': energy_out})
        self.Foragers.agents = step_agents(self.Foragers, forager_input)
        self.Resources.agents = step_agents(self.Resources, resource_input)
        self.Forager_poses = jit_update_forager_pos(self.Forager_poses, self.Foragers.agents, self.Space)

        ## Neuroevolution
        res_growth_dynams = (self.resource_population_growth_rate, self.resource_population_death_rate, self.max_resource_population, self.Foragers.num_total_agents)
        self.Foragers.agents, self.Forager_poses, self.Resources.agents, self.Resource_poses, self.key = jit_neuroevolution(self.Foragers.agents, self.Forager_poses, 
                                                                                                                         self.Resources.agents, self.Resource_poses, 
                                                                                                                         self.Space, self.key, res_growth_dynams)
    def run(self, print_freq, record_freq, record_start_step):
        
        sim_step = 0
        record_flag = False
        record_frame = 0

        while sim_step < self.Sim_steps:
            self.step()
            sim_step += 1
            
            if sim_step % print_freq == 0 and sim_step > 0:
                print("sim_step", sim_step)
                print("num_active_foragers", jnp.sum(self.Foragers.agents.active_state, dtype=jnp.int32))
                print("average age:", jnp.sum(self.Foragers.agents.age)/(jnp.sum(self.Foragers.agents.active_state+1)))
                print("max age:", jnp.max(self.Foragers.agents.age))

            if sim_step % record_freq == 0 and sim_step > record_start_step and not record_flag:
                print("start Recording data")
                record_flag = True
                self.rec_data = init_rec_data(self.max_record_steps, self.Resources, self.Resource_poses, self.Foragers, self.Forager_poses)
                record_frame = 0
            
            if record_flag:
                #print("Recording data")
                self.rec_data = record_render_data(self.Resources.agents, self.Resource_poses, self.Foragers.agents, self.Forager_poses, record_frame, self.rec_data)
                record_frame += 1
                if record_frame >= self.max_record_steps:
                    record_flag = False
                    print("Recording done, saving data")
                    rec_data = {'rec_data': self.rec_data}
                    self.checkpoint_manager.save(sim_step, rec_data, save_kwargs={'save_args': self.save_args})
                    

foragers_params = {'total_num': 1000, 'active_num': 330, 'type': 1, 'num_neurons': 3, 'num_obs': 27, 
                   'num_actions': 2, 'dt': 0.1, 'repr_thresh':4.0, 'action_scale': 1, 'deterministic': False, 'cooling_period': 3.3}
resources_params = {'total_num': 1000, 'active_num': 500, 'type': 2, 'dt': 0.1}
space_params = {'x_min': 0.0, 'x_max': 1000.0, 'y_min': 0.0, 'y_max': 1000.0, 'torous': False, 'wall_array': jnp.array([[[0.0, 0.0], [1000.0, 0.0]], [[1000.0, 0.0], [1000.0,1000.0]],[[0.0,0.0], [0.0, 1000.0]], [[0.0, 1000.0], [1000.0, 1000.0]]]) }

f_world = Foraging_World(foragers_params, resources_params, space_params, sim_steps=100000, max_record_steps=1000, key=jax.random.PRNGKey(2))

f_world.run(print_freq=100, record_freq=5000, record_start_step=10)