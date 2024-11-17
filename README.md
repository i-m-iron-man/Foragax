<div align="center">
    <img src="https://github.com/i-m-iron-man/Foragax/blob/main/docs/assets/small_foragaing.gif" width="250"/>
    <img src="https://github.com/i-m-iron-man/Foragax/blob/main/docs/assets/foragax_logo.webp" width="250"/>
    <img src="https://github.com/i-m-iron-man/Foragax/blob/main/docs/assets/sheep_wolf.gif" width="250"/>
</div>



Foragax is an Agent-Based Modelling (ABM) package based on JAX. It provides scalable and efficient ABM simulations by leveraging JAX's automatic vectorization and just-in-time compilation capabilities. The main features of Foragax include:

 - Agent manipulation (adding, removing, updating, selecting, and sorting agents) with just-in-time compilation.
 - Vectorized ray-casting and wall-detection for simulating agents moving in a continuous 2D environment with custom boundaries and obstacles.
 - Tutorials and examples to help users get started with ABM using JAX.
 - Familiar ABM interface for creating and manipulating agents.

## Installation
```
pip install foragax
```
Requires Python 3.10+, [JAX 0.4.13+](https://jax.readthedocs.io/en/latest/quickstart.html), and [flax 0.7.4+](https://flax.readthedocs.io/en/latest/quick_start.html)

## Hello World : Rolling Dice
This is an example of agents that roll a dice.
For each agent that draws a 6, a new agent is activated.
Each agent that draws a 1 is deactivated.
The note-book implementing this can be found [here](https://github.com/i-m-iron-man/Foragax/blob/main/examples/basic/hello_world/hello_world.ipynb).

```python
import foragax.base.agent_classes as fgx_classes
import foragax.base.agent_methods as fgx_methods
import jax
import jax.numpy as jnp
from flax import struct

@struct.dataclass
class Dice(fgx_classes.Agent):
    @staticmethod
    def create_agent(params: fgx_classes.Params, unique_id: int, active_state: int, agent_type: int, key:jax.random.PRNGKey):
        key, subkey = jax.random.split(key)
        
        def create_active_agent(key):
            draw = jax.random.randint(key, (1,), 1, 7)
            state_content = {'draw': draw, 'key': key}
            return fgx_classes.State(content=state_content)
        
        def create_inactive_agent(key):
            state_content = {'draw': jnp.array([0]), 'key': key}
            return fgx_classes.State(content=state_content)
        agent_state = jax.lax.cond(active_state, lambda _: create_active_agent(subkey), lambda _: create_inactive_agent(subkey), None)
        
        return Dice(params = params, unique_id = unique_id, agent_type = agent_type, 
                    active_state = active_state, state = agent_state, policy = None, age = 0.0)
    
    @staticmethod
    def step_agent(params: fgx_classes.Params, input: fgx_classes.Signal, dice_agent: fgx_classes.Agent):
        
        def step_active_agent(dice_agent):
            old_state = dice_agent.state.content
            key, subkey = jax.random.split(old_state['key'])
            
            draw = jax.random.randint(subkey, (1,), 1, 7)
            state_content = {'draw': draw, 'key': subkey}
            new_state = fgx_classes.State(content = state_content)
            return dice_agent.replace(state = new_state, age = dice_agent.age + 1.0)
        
        def step_inactive_agent(dice_agent):
            return dice_agent
        
        new_dice_agent = jax.lax.cond(dice_agent.active_state, lambda _: step_active_agent(dice_agent), lambda _: step_inactive_agent(dice_agent), None)
        return new_dice_agent
    
    @staticmethod
    def add_agent(params: fgx_classes.Params, dice_agents: fgx_classes.Agent, idx, key: jax.random.PRNGKey):
        inactive_dice_agent = jax.tree_util.tree_map(lambda x:x[idx], dice_agents)
        useless_key, subkey = jax.random.split(inactive_dice_agent.state.content['key'])
        draw = jax.random.randint(subkey, (1,), 1, 7)
        state_content = {'draw': draw, 'key': subkey}
        new_state = fgx_classes.State(content=state_content)
        active_dice_agent = inactive_dice_agent.replace(active_state = True, state = new_state)
        return active_dice_agent, key
    
    @staticmethod
    def remove_agent(params: fgx_classes.Params, dice_agents:fgx_classes.Agent, idx):
        active_dice_agent = jax.tree_util.tree_map(lambda x:x[idx], dice_agents)
        draw = jnp.array([0])
        state_content = {'draw': draw, 'key': active_dice_agent.state.content['key']}
        state = fgx_classes.State(content=state_content)
        inactive_dice_agent = active_dice_agent.replace(active_state = False, state = state)
        return inactive_dice_agent



Dice_set = fgx_classes.Agent_Set(agent = Dice, num_total_agents = 10, num_active_agents = 5, agent_type = 0)

Dice_set.agents = fgx_methods.create_agents(params = None, agent_set = Dice_set, key = jax.random.PRNGKey(0))

Dice_set.agents = fgx_methods.step_agents(params = None, agent_set = Dice_set, input=None)

# remove all agents who have drawn a 1
# first, select all agents who have drawn a 1
def is_one(dice_agent: fgx_classes.Agent, select_params: fgx_classes.Params):
    draw = jnp.reshape(dice_agent.state.content['draw'], (-1))
    return draw == 1
num_agents_dead, remove_indices = fgx_methods.jit_select_agents(select_func = is_one, select_params = None, agents = Dice_set.agents)

# now, remove the agents
dice_remove_params_content = {'remove_ids': remove_indices}
dice_remove_params = fgx_classes.Params(content = dice_remove_params_content)
Dice_set.agents = fgx_methods.jit_remove_agents(remove_func = Dice.remove_agent, num_agents_remove = num_agents_dead, 
                                                remove_params = dice_remove_params, agents = Dice_set.agents)

# sort agents by active state, as new agents are ALWAYS added at the END of the set
Dice_set.agents, sorted_indices = fgx_methods.jit_sort_agents(quantity = Dice_set.agents.active_state, ascend = False, agents = Dice_set.agents)


# add a new agent for every agent that has drawn a 6
# first, select all agents who have drawn a 6
def is_six(dice_agent: fgx_classes.Agent, select_params: fgx_classes.Params):
    draw = jnp.reshape(dice_agent.state.content['draw'], (-1))
    return draw == 6
num_agents_add, add_indices = fgx_methods.jit_select_agents(select_func = is_six, select_params = None, agents = Dice_set.agents)

# clip the number of agents to add to the number of inactive agents
num_active_agents = jnp.sum(Dice_set.agents.active_state, dtype = jnp.int32)
num_agents_add = jnp.minimum(num_agents_add, Dice_set.num_total_agents - num_active_agents)

# now, add the agents
Dice_set.agents, key = fgx_methods.jit_add_agents(add_func = Dice.add_agent, num_agents_add = num_agents_add, 
                                                  add_params = None, agents = Dice_set.agents, key = None)


```
## Version v0.0.5
The framework is still under active development. Feel free to open an issue if you find any bugs or have any suggestions.

## Citation

If this framework was useful in your work, please consider starring and cite: [(arXiv link)](https://www.arxiv.org/abs/2409.06345v2)

```bibtex
@misc{chaturvedi2024foragaxagentbasedmodelling,
      title={Foragax: An Agent Based Modelling framework based on JAX}, 
      author={Siddharth Chaturvedi and Ahmed El-Gazzar and Marcel van Gerven},
      year={2024},
      eprint={2409.06345},
      archivePrefix={arXiv},
      primaryClass={cs.MA},
      url={https://arxiv.org/abs/2409.06345}, 
}
```

