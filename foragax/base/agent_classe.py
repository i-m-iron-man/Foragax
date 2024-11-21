from flax import struct
import jax
import jax.numpy as jnp

@struct.dataclass
class Signal:
    content:dict

@struct.dataclass
class Params:
    content:dict

@struct.dataclass
class State:
    content:dict

@struct.dataclass
class Policy:
    state:State
    params:Params
    deterministic:bool

    @staticmethod
    def create_policy(params:Params, active_state:bool, key:jax.random.PRNGKey):
        pass
    @staticmethod
    def step_policy(input:Signal, policy:struct.dataclass, key:jax.random.PRNGKey):
        pass
    @staticmethod
    def reset_policy(input:Signal, policy:struct.dataclass, key:jax.random.PRNGKey):
        pass

@struct.dataclass
class Agent:
    params:Params
    state:State
    unique_id:jnp.int32
    agent_type:jnp.int32
    active_state:bool
    age:jnp.float32
    policy: Policy

    @staticmethod
    def create_agent(create_params:Params, unique_id:int, active_state:int, agent_type:int, key: jax.random.PRNGKey):
        pass
    
    @staticmethod
    def step_agent(step_params:Params, input:Signal, agents:struct.dataclass, key:jax.random.PRNGKey):
        pass

    @staticmethod
    def add_agent(add_params:Params, agents:struct.dataclass, idx:int, key:jax.random.PRNGKey):
        pass

    @staticmethod
    def set_agent(set_params:Params, agents:struct.dataclass, idx:int, key:jax.random.PRNGKey):
        pass

    @staticmethod
    def remove_agent(remove_params:Params, Agents:struct.dataclass, idx:int, key:jax.random.PRNGKey):
        pass



class Agent_Set:
    agents:Agent
    def __init__(self, agent:Agent, num_total_agents:jnp.int32, num_active_agents:jnp.int32, agent_type:jnp.int32):

        self.create_agents = jax.vmap(agent.create_agent, in_axes=(None,0,0,0,0))

        self.step_agents = jax.jit(jax.vmap(agent.step_agent, in_axes=(None,0,0,0)))

        self.num_total_agents = num_total_agents
        self.num_active_agents = num_active_agents
        self.num_inactive_agents = num_total_agents - num_active_agents
        self.agent_type = agent_type
        print("AgentSet initialized")