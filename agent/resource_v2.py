import sys

from jax._src.random import PRNGKey as PRNGKey

from base.agent_classes import Params

sys.path.insert(1, '/home/siddhart/source/Foragax_v2')
from base.agent_classes import *
from policy.wilson_cowan import WilsonCowan
import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class Resource(Agent):

    @staticmethod
    def create_agent(params: Params, unique_id: int, active_state: int, 
                     agent_type: int, key: jax.random.PRNGKey):
        
        policy = None

        key, param_key = jax.random.split(key)
        growth_rate = jax.random.uniform(param_key, (1,), minval=0.0, maxval=1.0)
        decay_rate = 0.1*growth_rate
        params_content = {'growth_rate': growth_rate, 'decay_rate': decay_rate, 'dt': params.content['dt']}
        agent_params = Params(content=params_content)

        def create_active_agent(key):
            key, *state_keys = jax.random.split(key, 2)
            val = jax.random.uniform(state_keys[0], (1,), minval=5.0, maxval=10.0)
            rad = val

            state_content = {'value': val, 'radius': rad}
            state = State(content=state_content)

            return state, key
        
        def create_inactive_agent(key):
            state_content = {'value': jnp.array([0.0]), 'radius': jnp.array([0.0])}
            state = State(content=state_content)

            return state, key
        
        agent_state, key = jax.lax.cond(active_state, lambda key: create_active_agent(key), lambda key: create_inactive_agent(key),
                                        key)
        return Resource(state=agent_state, policy=policy, 
                        params=agent_params, unique_id=unique_id, 
                        agent_type=agent_type, active_state=active_state, age=0.0)
    
    @staticmethod
    def step_agent(input:Signal, res:Agent):
        def do_step(input, res):
            dt = res.params.content['dt']
            state = res.state
            old_val = state.content['value']
            gr = res.params.content['growth_rate']
            dr = res.params.content['decay_rate']
            eaten = input.content['energy_out']
            new_val = old_val + dt*(gr * old_val - dr * (old_val**2) - eaten)
            state.content['value'] = new_val
            state.content['radius'] = new_val
            res.replace(state=state, age = res.age + dt)

            return res
        def dont_step():
            return res
        
        res = jax.lax.cond(res.active_state, lambda res: do_step(input, res), lambda res: dont_step(), res)
        return res
    
    @staticmethod
    def remove_agent(remove_params: Params, Agents: Agent, idx):
        agent_to_remove = jax.tree_util.tree_map(lambda x: x[idx], Agents)
        state_content = {'value': jnp.array([0.0]), 'radius': jnp.array([0.0])}
        state = State(content=state_content)
        age = 0.0
        active_state = False
        new_agent = agent_to_remove.replace(state=state, active_state=active_state, age=age)
        return new_agent

    @staticmethod
    def add_agent(add_params:Params, Resources: Agent, idx, key:jax.random.PRNGKey):
        good_ids = add_params.content['good_ids']
        num_active_res = add_params.content['num_active_agents']
        res_to_add = jax.tree_util.tree_map(lambda x: x[idx], Resources)
        res_to_copy = jax.tree_util.tree_map(lambda x: x[good_ids[idx - num_active_res]], Resources)

        def res_to_copy_active(res_to_copy, res_to_add, key):
            key, *keys = jax.random.split(key, 3)
            val = res_to_copy.state.content['value'] + jax.random.uniform(keys[0], (1,), minval=-0.01, maxval=0.01)
            rad = val
            growth_rate = res_to_copy.params.content['growth_rate'] + jax.random.uniform(keys[1], (1,), minval=-0.01, maxval=0.01)
            decay_rate = 0.1*growth_rate

            state = State(content={'value': val, 'radius': rad})

            params = Params(content={'growth_rate': growth_rate, 'decay_rate': decay_rate, 'dt': res_to_add.params.content['dt']})

            return res_to_add.replace(state=state, params=params, active_state=True, age=0.0), key

        def res_to_copy_inactive(res_to_add, key):
            key, *keys = jax.random.split(key, 3)
            val = jax.random.uniform(keys[0], (1,), minval=5.0, maxval=10.0)
            rad = val
            growth_rate = jax.random.uniform(keys[1], (1,), minval=0.0, maxval=1.0)
            decay_rate = 0.1*growth_rate
            state = State(content={'value': val, 'radius': rad})
            params = Params(content={'growth_rate': growth_rate, 'decay_rate': decay_rate, 'dt': res_to_add.params.content['dt']})
            return res_to_add.replace(state=state, params=params, active_state=True, age=0.0), key
        
        new_res, key = jax.lax.cond(res_to_copy.active_state,lambda _: res_to_copy_active(res_to_copy, res_to_add, key), lambda _: res_to_copy_inactive(res_to_add, key), None)
        return new_res, key