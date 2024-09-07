#this policy class is used just to transfer the keys and show the structure of the policy class


from flax.struct import dataclass
from base.agent_classes import Policy, Signal, Params, State
import jax
import jax.numpy as jnp

class Random_Policy(Policy):

    @staticmethod
    def create_policy(params:Params, key:jax.random.PRNGKey):
        state = State(content={'key':key})
        return Policy(state=state, params=params, deterministic=False)
    
    @staticmethod
    def step_policy(input:Signal, policy:Policy):
        key = policy.state.content['key']
        key, subkey = jax.random.split(key)
        action = jax.random.uniform(key, shape=(1,))
        state = State(content={'key':subkey})
        new_policy = policy.replace(state=state)
        return Signal(content={'action':action}), new_policy
    
    @staticmethod
    def reset_policy(key:jax.random.PRNGKey, policy:Policy):
        state = State(content={'key':key})
        return policy.replace(state=state)
