---
title: 'Foragax: A Python package for Agent Based Modelling using JAX'
tags:
    - Python
    - JAX
    - Agent Based Modelling

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

# Sunmmary

Agent Based Modelling (ABM) is a useful tool to simulate models of complex systems. However, ABM can be computationally expensive, especially when the number of agents is large. Foragax is a Python package that uses JAX to accelerate ABM simulations. JAX is a library that provides automatic vectorization and just-in-time compilation, which can be used to speed up computations. It also provides automatic differentiation, which can be useful for training agents using gradient-based methods. Foragax provides a simple and familiar ABM interface for creating and manipulating agents. Although Foragax is geared towards general-purpose ABM, it provides certain predefined functionalities like vectorized ray-casting and wall-detection for simulating agents moving in a continuous 2D environment with custom boundaries and obstacles.

# Statement of need

Many-agents scenarios are prevalent in many fields, including neuroscience, biology, social sciences, and economics. Chief examples include simulating the behavior of neurons in the brain, the movement of animals in a habitat, the spread of diseases in a population, the dynamics of financial markets, and gauging the effects of policy changes on a society. Foragax will provide a simple and efficient way to simulate these scenarios using JAX.

The main feature of Foragax is that it can handle agent-manipulation operations without giving up the just-in-time compilation functionality provided by JAX. This includes adding and removing a traced-number of agents, updating properties of selected agents to traced values, and sorting them based on custom trends. This feature combined with automatic vectorization helps scale ABM developed in Foragax to many-agent scenarios efficiently. Foragax will also provide a series of tutorials and examples to help users get started with ABM using JAX.

The package was inspired by the need to simulate continuous non-episodic evolution in many-agent patch foraging environments. Thus, it also provides utilities for simulating a ray-casting sensor model for agents to detect walls and other agents in a continuous 2D environment in a vectorized manner. This feature is particularly useful for simulating scenarios where agents need to navigate through a maze or avoid obstacles. A dedicated tutorial will be provided to demonstrate how to use this feature.


# Acknowledgements
This software is part of the project Dutch Brain Interface Initiative (DBI2) with project number 024.005.022 of the research programme Gravitation which is (partly) financed by the Dutch Research Council (NWO).







    