import sys
sys.path.insert(1, '/home/siddhart/source/Foragax_v2')
from base.agent_classes import Policy, Signal, Params, State

import jax
import jax.numpy as jnp

class WilsonCowan(Policy):
    @staticmethod
    def create_policy(params:Params, key:jax.random.PRNGKey):
        num_neurons = params.content['num_neurons']
        num_obs = params.content['num_obs']
        num_actions = params.content['num_actions']
        deterministic = params.content['deterministic']

        Z = jnp.zeros((num_neurons,))
        state = State(content={'Z':Z})
       
        key, *init_keys = jax.random.split(key, 6)
        init_keys = jnp.array(init_keys)

        J = jax.random.uniform(init_keys[0], (num_neurons,num_neurons), minval=-1, maxval=1)
        tau = jax.random.uniform(init_keys[1], (num_neurons,), minval=-1, maxval=1)
        E = jax.random.uniform(init_keys[2], (num_neurons,num_obs), minval=-1, maxval=1)
        B = jax.random.uniform(init_keys[3], (num_neurons,), minval=-1, maxval=1)
        D = jax.random.uniform(init_keys[4], (num_actions,num_neurons), minval=-1, maxval=1)
        dt = params.content['dt']
        action_scale = params.content['action_scale']
        
        params = Params(content={'J':J, 'tau':tau, 'E':E, 'B':B, 'D':D, 'dt':dt, 'action_scale':action_scale})
        return Policy(state=state, params=params, deterministic=deterministic)

    @staticmethod
    def step_policy(input:Signal, policy:Policy):
        J = policy.params.content['J']
        tau = policy.params.content['tau']
        E = policy.params.content['E']
        B = policy.params.content['B']
        D = policy.params.content['D']
        dt = policy.params.content['dt']
        action_scale = policy.params.content['action_scale']

        Z = policy.state.content['Z']
        obs = input.content['obs']

        dz = jnp.matmul(J, jnp.tanh(Z)) + jnp.matmul(E, obs) + B - Z
        dz = jnp.multiply(dz, 10*jax.nn.sigmoid(tau))
        new_Z = Z + dt*dz # maybe remove dt to make the steps bigger
        new_policy = policy.replace(state=State(content={'Z':new_Z}))
        actions = action_scale * jnp.tanh(jnp.matmul(D, jnp.tanh(new_Z)))
        return Signal(content={'actions':actions}), new_policy

    @staticmethod
    def reset_policy(key:jax.random.PRNGKey, policy:Policy):
        key, *init_keys = jax.random.split(key, 6)
        init_keys = jnp.array(init_keys)

        J = jax.random.uniform(init_keys[0], policy.params.content['J'].shape, minval=-1, maxval=1)
        tau = jax.random.uniform(init_keys[1], policy.params.content['tau'].shape, minval=-1, maxval=1)
        E = jax.random.uniform(init_keys[2], policy.params.content['E'].shape, minval=-1, maxval=1)
        B = jax.random.uniform(init_keys[3], policy.params.content['B'].shape, minval=-1, maxval=1)
        D = jax.random.uniform(init_keys[4], policy.params.content['D'].shape, minval=-1, maxval=1)

        new_params = Params(content={'J':J, 'tau':tau, 'E':E, 'B':B, 'D':D, 'dt':policy.params.content['dt'], 'action_scale':policy.params.content['action_scale']})
        new_policy = policy.replace(params=new_params)
        return new_policy

        