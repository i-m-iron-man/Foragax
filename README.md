<div align="center">
    <img src="https://github.com/i-m-iron-man/Foragax/blob/main/Docs/assets/foragax_logo.webp" width="250">
</div>


Foragax is an Agent Based Modelling (ABM) package based on JAX. It provides scalable and efficient ABM simulations by leveraging JAX's automatic vectorization and just-in-time compilation capabilities. The main features of Foragax include:

 - Agent manipulation (adding, removing, updating, selecting, and sorting agents) with just-in-time compilation.
 - Vectorized ray-casting and wall-detection for simulating agents moving in a continuous 2D environment with custom boundaries and obstacles.
 - Tutorials and examples to help users get started with ABM using JAX.
 - Familiar ABM interface for creating and manipulating agents.

## Installation
```
pip install foragax
```
Requires Python 3.10+, [JAX 0.4.13+](https://jax.readthedocs.io/en/latest/quickstart.html), and [flax 0.7.4+](https://flax.readthedocs.io/en/latest/quick_start.html)


## Citation

If this framework was useful in your work, please consider starring and cite: [(arXiv link)](https://arxiv.org/abs/2409.06345v1)

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

