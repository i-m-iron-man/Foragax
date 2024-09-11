import jax
import jax.numpy as jnp
from flax import struct
from base.agent_classes import *
from base.space_classes import *



def create_agents(params:Params, agent_set:Agent_Set, key:jax.random.PRNGKey):
    uniq_ids = jnp.arange(agent_set.num_total_agents)
    key, *create_keys = jax.random.split(key, agent_set.num_total_agents+1)
    create_keys = jnp.array(create_keys)

    active_states = jnp.concatenate((jnp.ones(agent_set.num_active_agents), jnp.zeros(agent_set.num_inactive_agents)))
    agent_types = jnp.tile(agent_set.agent_type, agent_set.num_total_agents)

    return agent_set.create_agents(params, uniq_ids, active_states, agent_types, create_keys)


def step_agents(params:Params, input:Signal, agent_set:Agent_Set):
    try:
        return agent_set.step_agents(params, input, agent_set.agents)
    except ValueError:
        print("Error in step_agents")


def add_agents(add_func:callable, num_agents_add:int, add_params:Params, agents:Agent, key:jax.random.PRNGKey):
    id_last_active = jnp.sum(agents.active_state, dtype=jnp.int32)

    def set_data(idx, agents__add_params__key):
        agents, add_params, key = agents__add_params__key
        new_agent,key = jax.jit(add_func)(add_params, agents, idx, key)
        new_agents = jax.tree_util.tree_map(lambda x,y:x.at[idx].set(y), agents, new_agent)
        return new_agents, add_params, key
        
    new_agents, add_params, key = jax.lax.fori_loop(id_last_active, id_last_active+num_agents_add, set_data, (agents, add_params, key))
    return new_agents, key

jit_add_agents = jax.jit(add_agents, static_argnums=(0,))


def remove_agents(remove_func:callable, num_agents_remove:int, remove_params:Params, agents:Agent):
    
    def remove_data(idx, agents__remove_params):
        agents, remove_params = agents__remove_params
        remove_ids = remove_params.content['remove_ids']
        new_agent = jax.jit(remove_func)(remove_params, agents, remove_ids[idx])
        new_agents = jax.tree_util.tree_map(lambda x,y:x.at[remove_ids[idx]].set(y), agents, new_agent)
        return new_agents, remove_params
    new_agents, remove_params = jax.lax.fori_loop(0, num_agents_remove, remove_data, (agents, remove_params))
    return new_agents

jit_remove_agents = jax.jit(remove_agents, static_argnums=(0,))

def set_agents(set_func:callable, num_agents_set:int, set_params:Params, agents:Agent, key:jax.random.PRNGKey):
    def set_data(idx, agents__set_params__key):
        agents, set_params, key = agents__set_params__key
        set_ids = set_params.content['set_ids']
        new_agent, key = jax.jit(set_func)(set_params, agents, set_ids[idx], key)
        new_agents = jax.tree_util.tree_map(lambda x,y:x.at[set_ids[idx]].set(y), agents, new_agent)
        return new_agents, set_params, key
    new_agents, set_params, key = jax.lax.fori_loop(0, num_agents_set, set_data, (agents, set_params, key))
    return new_agents, key
jit_set_agents = jax.jit(set_agents, static_argnums=(0,))

#returns a sorted list of indexes of agents (1 for ids that are selected, 0 for ids that are not selected) + the length of selected ids 
def select_agents(select_func:bool, select_params:Params, agents:Agent):
    selected_ids = jnp.where(jax.jit(select_func)(agents, select_params), 1.0, 0.0)
    selected_ids = jnp.reshape(selected_ids,(-1,))
    sort_selected_ids = jnp.argsort(-1*selected_ids)    
    selected_ids_len = jnp.sum(selected_ids, dtype=jnp.int32)
    return selected_ids_len, sort_selected_ids
jit_select_agents = jax.jit(select_agents, static_argnums=(0,))

def sort_agents(quantity:jnp.array, ascend:bool, agents:Agent):
    quantity = jnp.reshape(quantity, (-1,))
    sorted_ids = jax.lax.cond(ascend, lambda _: jnp.argsort(quantity), lambda _: jnp.argsort(-quantity), None)
    new_agents = jax.tree_util.tree_map(lambda x: jnp.take(x, sorted_ids, axis=0), agents)
    return new_agents, sorted_ids
jit_sort_agents = jax.jit(sort_agents)


