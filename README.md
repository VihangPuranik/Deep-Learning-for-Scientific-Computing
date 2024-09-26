# Deep Learning for Scientific Computing

Uses a new class of neural networks, called Physics Informed Neural Networks to compute the solutions of the Partial Differential Equations that govern the physical phenomena.

This is based on the course work of 401-4656-21L at ETH Zurich.

1. FunctionApproximation.ipynb:
   Demonstrates the basic behaviour of these neural networks, covering the function spaces that would represent the governing equations, upon a suitable choice.
3. Pinns.ipynb
   The notebook demonstrates the role of integrating the governing PDEs within the loss functions to show that the real life phenomena can be simulated by using the Physics Informed Neural Networks class from scratch.
5. NeuralOperator.ipynb
   This notebook demonstrates the development of Operators as they occur in PDEs and implements a deep learning way of using them. The deep-O nets are implemented that calculate these neural operators for evaluation.

The following image shows the evaluation of Heat equation PDE as simulated with classical techniques on the left, and with using Physics Informed Neural Networks on the right.

![output](https://github.com/user-attachments/assets/3437d8fd-8998-4290-bc1b-bbdf7935e777)

The main inspiration comes from the works:
- Raissi, Maziar, Paris Perdikaris, and George E. Karniadakis. "[Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Journal of Computational Physics 378 (2019): 686-707.

- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).

- Raissi, Maziar, Paris Perdikaris, and George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).
