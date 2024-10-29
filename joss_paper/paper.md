---
title: 'Foragax: A Python package for Agent-Based Modelling using JAX'
tags:
    - Python
    - JAX
    - Agent-Based Modelling

authors:
    - name: Siddharth S. Chaturvedi
      orcid: 0009-0000-4965-1718
      correspondence: true
      affiliation: 1
    - name: Ahmed El-Gazzar
      affiliation: 1
    - name: Marcel van Gerven
      affiliation: 1

affiliations:
    - name: Department of Machine Learning and Neural Computation, Donders Institute for Brain, Cognition and Behaviour, Radboud University, Nijmegen, The Netherlands
      index: 1

date: 25 November 2024
bibliography: paper.bib
---
# Summary
Agent-based modelling (ABM) is a useful tool to simulate models of complex systems. However, ABM can be computationally expensive, especially when the number of agents is large. `Foragax` is a Python package that uses `JAX` to accelerate ABM simulations. `JAX` is a library that provides automatic vectorization and just-in-time compilation, which can be used to speed up computations. It also provides automatic differentiation, which can be useful for training agents using gradient-based methods. `Foragax` provides these features to users to design efficient ABM simulations. Although `Foragax` can be used to formulate a general ABM, it is particularly tailored for foraging scenarios by providing utilities to simulate a ray-casting sensor model for agents to detect walls and other agents in a continuous 2D environment in a vectorized manner.


# Statement of need

Many-agent scenarios are prevalent in many fields, including neuroscience, biology, social sciences, and economics [@eck2023decision]. Chief examples include simulating the behavior of neurons in the brain, the movement of animals in a habitat, the spread of diseases in a population, the dynamics of financial markets, and gauging the effects of policy changes on society. `Foragax` is an ABM framework that provides a simple and efficient way to simulate such scenarios using `JAX` [@jax2018github]. In the front end, `Foragax` provides a simple API to define agents, their properties, and their interactions similar to other ABM frameworks like `Mesa` [@kazil2020utilizing]. 
The main feature of `Foragax` is that it can handle agent-manipulation operations without giving up the just-in-time compilation functionality provided by `JAX`. This includes adding and removing a traced number of agents, updating properties of selected agents to traced values, and sorting them based on custom trends. This feature combined with automatic vectorization helps scale ABM developed in `Foragax` to many-agent scenarios efficiently. `Foragax` will also provide a series of tutorials and examples to help users get started with ABM using `JAX`.

The package was inspired by the need to simulate continuous non-episodic evolution in many-agent patch foraging environments [@hamon2023eco]. Thus, it also provides utilities for simulating a ray-casting sensor model for agents to detect walls and other agents in a continuous 2D environment in a vectorized manner. This feature is particularly useful for simulating scenarios where agents need to navigate through a maze or avoid obstacles.

# An example.
An example highlighting a collection of dice-rolling agents can be found [here](https://github.com/i-m-iron-man/Foragax/blob/main/examples/hello_world/hello_world.ipynb). In this example an agent updates its state based on the outcome of a dice roll. If the agent draws a six, a deactivated agent is activated. If the agent draws a one, the agent itself is deactivated.

# Limitations

Since `Foragax` uses `JAX` at its backend, it is limited to the features provided by JAX as summarized [here](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html). For example, certain functionalities like automatic-vectorization doe not support Python's built-in data structures like lists and dictionaries readily. Instead, it uses `NumPy` arrays and `JAX's` own data structures. This means that users may need to adapt their code to work with these data structures. Additionally, `JAX's` just-in-time compilation and automatic vectorization can only be applied to pure functions, which in turn need the function arguments to be of fixed shapes during runtime. In `Foragax`, we circumvent this by simulating in-active agents along with active agents as blanks such that their collective sum remains constant. This in turn needs the user to define a maximum number of agents beforehand.


# Acknowledgements
This software is part of the project Dutch Brain Interface Initiative (DBI2) with project number 024.005.022 of the research programme Gravitation which is (partly) financed by the Dutch Research Council (NWO).

# References


    