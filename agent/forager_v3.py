#in this version the forager's angle does not change is not independent dof
'''
In this version of forager, the positions model is exported
to the foraging world because the position needs to be
updated by checking wall collisions. The forager's position
should reside in the foraging world, not in the agent itself.
'''

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
class Forager(Agent):
    
    @staticmethod
    def create_agent(params: Params, unique_id: int, active_state: int, 
                     agent_type: int, key: jax.random.PRNGKey):
        
        key, policy_key = jax.random.split(key)
        policy_params_content = params.content['policy_params']
        policy_params = Params(content=policy_params_content)
        policy = WilsonCowan.create_policy(policy_params, policy_key)
        
        params_content = {'dt': params.content['dt'], 'repr_thresh': params.content['repr_thresh'], 
                          'cooling_period': params.content['cooling_period']}
        agent_params = Params(content=params_content)
        
        def create_active_agent(key):
            
            key, *keys = jax.random.split(key, 4)
            X_acc = jax.random.uniform(keys[0], (1,), minval=-1.0, maxval=1.0)
            Y_acc = jax.random.uniform(keys[1], (1,), minval=-1.0, maxval=1.0)
            val = jax.random.uniform(keys[2], (1,), minval=5.0, maxval=10.0)
            rad = val
            time_above_repr_thr = jnp.zeros(1)
            time_below_death_thr = jnp.zeros(1)

            state_content = {'X_acc':X_acc, 'Y_acc':Y_acc, 
                             'value': val, 'radius': rad, 'time_above_repr_thr': 
                             time_above_repr_thr, 'time_below_death_thr': time_below_death_thr}
            state = State(content=state_content)

            return state
        
        def create_inactive_agent():
            state_content = {'X_acc': jnp.array([0.0]), 'Y_acc': jnp.array([0.0]),
                            'value': jnp.array([0.0]), 'radius': jnp.array([0.0]), 
                            'time_above_repr_thr': jnp.array([0.0]), 'time_below_death_thr': jnp.array([0.0])}
            state = State(content=state_content)

            return state
        
        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(key), lambda _: create_inactive_agent(), 
                                                None)
        return Forager(state=agent_state, policy=policy, params=agent_params, unique_id=unique_id, agent_type=agent_type, active_state=active_state, age=0.0)
    

    @staticmethod
    def step_agent(input:Signal, forager:Agent):
        def do_step(input, forager):
            obs = Signal(content={'obs':input.content['obs']})
            #update the policy state and get the actions
            actions, new_policy = WilsonCowan.step_policy(obs, forager.policy)

            #update the acceleration of the forager
            X_acc = jnp.array([actions.content['actions'][0]])
            Y_acc = jnp.array([actions.content['actions'][1]])

            dt = forager.params.content['dt']
            

            #update the energy of the forager
            val = forager.state.content['value']
            new_val = val + dt * (input.content['energy_in'] - jnp.sum(jnp.abs(actions.content['actions'])))

            #hyperparameter for neuroevolution
            new_val = jnp.minimum(new_val, 15.0)
            new_rad = new_val

            repr_thresh = forager.params.content['repr_thresh']
            cooling_period = forager.params.content['cooling_period']
            time_above_repr_thr = jax.lax.cond(new_val[0] > repr_thresh, 
                                               lambda _: forager.state.content['time_above_repr_thr'] + dt, 
                                               lambda _: jnp.zeros(1), None)
            time_above_repr_thr = jax.lax.cond(time_above_repr_thr[0] > cooling_period,
                                                  lambda _: jnp.zeros(1),
                                                  lambda _: time_above_repr_thr, None) 
            
            new_forager_state_content = { 'X_acc': X_acc, 'Y_acc': Y_acc, 
                                          'value': new_val, 'radius': new_rad, 
                                          'time_above_repr_thr': time_above_repr_thr, 
                                          'time_below_death_thr': forager.state.content['time_below_death_thr']}
            new_forager_state = State(content=new_forager_state_content)
            new_forager = forager.replace(state=new_forager_state, policy=new_policy, age = forager.age + dt)
            return new_forager
        
        def dont_step():
            return forager
        
        new_forager = jax.lax.cond(forager.active_state, lambda _: do_step(input, forager), lambda _: dont_step(), None)
        return new_forager
    
    @staticmethod
    def remove_agent(remove_params:Params, Foragers:Agent, idx):
        forager_to_remove = jax.tree_util.tree_map(lambda x: x[idx], Foragers)
        state_content = {'X_acc': jnp.array([0.0]), 'Y_acc': jnp.array([0.0]),
                            'value': jnp.array([0.0]), 'radius': jnp.array([0.0]), 
                            'time_above_repr_thr': jnp.array([0.0]), 'time_below_death_thr': jnp.array([0.0])}
        state = State(content=state_content)

        policy_state_content = {'Z': jnp.zeros_like(forager_to_remove.policy.state.content['Z'])}
        policy_state = State(content=policy_state_content)
        policy = forager_to_remove.policy.replace(state=policy_state)

        age = 0.0
        active_state = False
        new_agent = forager_to_remove.replace(state=state, policy = policy, active_state=active_state, age=age)
        return new_agent
    
    
    @staticmethod
    def add_agent(add_params:Params, Foragers:Agent, idx, key:jax.random.PRNGKey):
        good_ids = add_params.content['good_ids'] # contains ids of all the agents, but the ones in the beginning are the ones that will be copies, idx takes care of the number of agents to copy
        
        '''
        logic: value of idx is iterated from num_active_agents to (number of agents to add + num active agents) by the external for loop in agents_methods
        thus idx-num_active_agents will give the index of the agent that is to be copied
        '''
        
        num_active_agents = add_params.content['num_active_agents']
        agent_to_add = jax.tree_util.tree_map(lambda x: x[idx], Foragers)
        agent_to_copy = jax.tree_util.tree_map(lambda x: x[good_ids[idx - num_active_agents]], Foragers)

        key, *keys = jax.random.split(key, 4)
        X_acc = jax.random.uniform(keys[0], (1,), minval=-1.0, maxval=1.0)
        Y_acc = jax.random.uniform(keys[1], (1,), minval=-1.0, maxval=1.0)
        val = jax.random.uniform(keys[2], (1,), minval=5.0, maxval=10.0)
        rad = val
        time_above_repr_thr = jnp.zeros(1)
        time_below_death_thr = jnp.zeros(1)

        state_content = {'X_acc':X_acc, 'Y_acc':Y_acc, 
                        'value': val, 'radius': rad, 'time_above_repr_thr': 
                        time_above_repr_thr, 'time_below_death_thr': time_below_death_thr}
        state = State(content=state_content)

        key, *noise_keys = jax.random.split(key, 6)
        J = agent_to_copy.policy.params.content['J']
        J_noise = jax.random.normal(noise_keys[0], J.shape)
        new_J = J + 0.1 * J_noise

        tau = agent_to_copy.policy.params.content['tau']
        tau_noise = jax.random.normal(noise_keys[1], tau.shape)
        new_tau = tau + 0.1 * tau_noise

        E = agent_to_copy.policy.params.content['E']
        E_noise = jax.random.normal(noise_keys[2], E.shape)
        new_E = E + 0.1*E_noise

        B = agent_to_copy.policy.params.content['B']
        B_noise = jax.random.normal(noise_keys[3], B.shape)
        new_B = B + 0.1*B_noise

        D = agent_to_copy.policy.params.content['D']
        D_noise = jax.random.normal(noise_keys[4], D.shape)
        new_D = D + 0.1*D_noise


        policy_params_content = {'J': new_J, 'tau': new_tau, 'E': new_E, 'B': new_B, 'D': new_D, 
                                 'dt': agent_to_add.policy.params.content['dt'], 'action_scale' : agent_to_add.policy.params.content['action_scale']}
        policy_params = Params(content=policy_params_content)

        policy = agent_to_add.policy.replace(params=policy_params)

        active_state = True

        new_agent = agent_to_add.replace(state=state, policy=policy, active_state=active_state)
        return new_agent, key

    