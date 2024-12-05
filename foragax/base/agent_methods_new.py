import jax
import jax.numpy as jnp
from flax import struct
from agent_class_new import *



def create_agents(agent:Agent, params:Params, num_agents:jnp.int32, num_active_agents:jnp.int32, agent_type:jnp.int16, key:jax.random.PRNGKey):
    ids = jnp.arange(num_agents)
    key, *subkeys = jax.random.split(key, num_agents+1)
    subkeys = jnp.array(subkeys)
    active_states = jnp.ones(num_agents)
    active_states = jnp.where(ids<num_active_agents, 1, 0)
    return jax.vmap(agent.create_agent, in_axes=(None, 0, 0, 0, 0))(agent_type, params, ids, active_states, subkeys)

def create_agent_sets(agent_set_params:Params, agent_params:Params, agent_set:Agent_Set, agent:Agent, agent_type:jnp.int16,
                     num_agent_set:jnp.int32, num_agents:jnp.int32, num_active_agents:jnp.int32, key:jax.random.PRNGKey):
    
    ids = jnp.arange(num_agent_set)
    key, *agent_subkeys = jax.random.split(key, num_agent_set+1)
    agent_subkeys = jnp.array(agent_subkeys)
    key, *agent_set_subkeys = jax.random.split(key, num_agent_set+1)
    agent_set_subkeys = jnp.array(agent_set_subkeys)
    agents = jax.vmap(create_agents, in_axes=(None, 0, None, 0, None, 0))(agent, agent_params, num_agents, num_active_agents, agent_type, agent_subkeys)
    
    return jax.vmap(agent_set.create_agent_set, in_axes=(None, 0, 0, 0, 0, 0))(num_agents, num_active_agents, agents, agent_set_params, ids, agent_set_subkeys)


def step_agents(step_params:Params, input:Signal, agent:Agent):
    return agent.step_agent(step_params, input, agent)
jit_step_agents = jax.jit(jax.vmap(step_agents, in_axes=(None, 0, 0)))


def step_agents_2(step_func:callable, step_params:Params, input:Signal, agents:Agent):
    return jax.vmap(step_func, in_axes=(None, 0, 0))(step_params, input, agents)
jit_step_agents_2 = jax.jit(step_agents_2, static_argnums=(0,))


def step_agent_sets(step_params:Params, input:Signal, agent_set:Agent_Set):
    return agent_set.step_agent_set(step_params, input, agent_set)
jit_step_agent_sets = jax.jit(jax.vmap(step_agent_sets, in_axes=(None, 0, 0)))


def step_agent_sets_2(step_func:callable, step_params:Params, input:Signal, agent_set:Agent_Set):
    return jax.vmap(step_func, in_axes=(None, 0, 0))(step_params, input, agent_set)
jit_step_agent_sets_2 = jax.jit(step_agent_sets_2, static_argnums=(0,))


def add_agents(add_func:callable, num_agents_add:jnp.int32, add_params:Params, agent_set:Agent_Set):
    def _add_agents(num_agents_add, add_params, agent_set):
        id_last_active = agent_set.num_active_agents
        max_agents_add = agent_set.num_agents - id_last_active
        num_agents_add = jnp.minimum(num_agents_add, max_agents_add)

        def add_data(idx, agents):
            new_agent = jax.jit(add_func)(add_params, agents, idx)
            new_agents = jax.tree_util.tree_map(lambda x,y:x.at[idx].set(y), agents, new_agent)
            return new_agents
        
        new_agents= jax.lax.fori_loop(id_last_active, id_last_active+num_agents_add, add_data, agent_set.agents)
    
        new_agent_set = agent_set.replace(agents=new_agents, num_active_agents=agent_set.num_active_agents+num_agents_add)
        return new_agent_set
    return jax.vmap(_add_agents)(num_agents_add, add_params, agent_set)
jit_add_agents = jax.jit(add_agents, static_argnums=(0,))


def remove_agents(remove_func:callable, num_agents_remove:jnp.int32, remove_params:Params, agent_set:Agent_Set):
    def _remove_agents(num_agents_remove, remove_params, agent_set):
        num_agents_remove = jnp.minimum(num_agents_remove, agent_set.num_active_agents)
        def remove_data(idx, agents):
            remove_ids = remove_params.content['remove_ids']
            new_agent = jax.jit(remove_func)(remove_params, agents, remove_ids[idx])
            new_agents = jax.tree_util.tree_map(lambda x,y:x.at[remove_ids[idx]].set(y), agents, new_agent)
            return new_agents
        
        new_agents = jax.lax.fori_loop(0, num_agents_remove, remove_data, agent_set.agents)
        new_agents, sorted_ids = jit_sort_agents(-1*new_agents.active_state, new_agents)
        new_agent_set = agent_set.replace(agents=new_agents, num_active_agents=agent_set.num_active_agents-num_agents_remove)
        
        return new_agent_set
    
    return jax.vmap(_remove_agents)(num_agents_remove, remove_params, agent_set)
jit_remove_agents = jax.jit(remove_agents, static_argnums=(0,))


def set_agents(set_func:callable, num_agents_set:jnp.int32, set_params:Params, agent_set:Agent_Set):
    def _set_agents(num_agents_set, set_params, agent_set):
        def set_data(idx, agents):
            set_ids = set_params.content['set_ids']
            new_agent = jax.jit(set_func)(set_params, agents, set_ids[idx])
            new_agents = jax.tree_util.tree_map(lambda x,y:x.at[set_ids[idx]].set(y), agents, new_agent)
            return new_agents
        
        new_agents = jax.lax.fori_loop(0, num_agents_set, set_data, agent_set.agents)
        new_agent_set = agent_set.replace(agents=new_agents)
        return new_agent_set
    return jax.vmap(_set_agents)(num_agents_set, set_params, agent_set)
jit_set_agents = jax.jit(set_agents, static_argnums=(0,))

def sort_agents(quantity:jnp.array, agents:Agent):
    quantity = jnp.reshape(quantity, (-1,))
    sorted_ids = jnp.argsort(quantity)
    new_agents = jax.tree_util.tree_map(lambda x: jnp.take(x, sorted_ids, axis=0), agents)
    return new_agents, sorted_ids
jit_sort_agents = jax.jit(sort_agents)

def sort_agent_sets(quantity:jnp.array, agent_sets:Agent_Set):
    quantity = jnp.reshape(quantity, (-1,))
    sorted_ids = jnp.argsort(quantity)
    new_agent_sets = jax.tree_util.tree_map(lambda x: jnp.take(x, sorted_ids, axis=0), agent_sets)
    return new_agent_sets, sorted_ids
jit_sort_agent_sets = jax.jit(sort_agent_sets)



#returns a sorted list of indexes of agents (1 for ids that are selected, 0 for ids that are not selected) + the length of selected ids 
def select_agents(select_func:bool, select_params:Params, agents:Agent):
    def _select_agents(select_params, agents):
        selected_ids = jnp.where(jax.jit(select_func)(agents, select_params), 1.0, 0.0)
        selected_ids = jnp.reshape(selected_ids,(-1,))
        sort_selected_ids = jnp.argsort(-1*selected_ids)    
        selected_ids_len = jnp.sum(selected_ids, dtype=jnp.int32)
        return selected_ids_len, sort_selected_ids
    return jax.vmap(_select_agents)(select_params, agents)
jit_select_agents = jax.jit(select_agents, static_argnums=(0,))

def select_agent_sets(select_func:bool, select_params:Params, agent_sets:Agent_Set):
    def _select_agent_sets(select_params, agent_sets):
        selected_ids = jnp.where(jax.jit(select_func)(agent_sets, select_params), 1.0, 0.0)
        selected_ids = jnp.reshape(selected_ids,(-1,))
        sort_selected_ids = jnp.argsort(-1*selected_ids)    
        selected_ids_len = jnp.sum(selected_ids, dtype=jnp.int32)
        return selected_ids_len, sort_selected_ids
    return jax.vmap(_select_agent_sets)(select_params, agent_sets)


'''
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

'''
