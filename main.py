import torch
import numpy as np
from rde_solver import rde_solver
from neural_ode_solver import neural_ode_solver
import matplotlib.pyplot as plt

# Set random seed
torch.manual_seed(0)

# Set hyperparameters
L = 1.0
alpha1 = lambda t : np.sin(10 * t)
alpha2 = lambda t : t ** 2

# Solve the RDE
t_rde_sin, u_rde_sin = rde_solver(L, alpha1)
t_rde_sq, u_rde_sq = rde_solver(L, alpha2)

# Solve the Neural ODE
# t_neural_sin, u_neural_sin, u_neural_sin_init = neural_ode_solver(L, alpha1, num_iters=1000, rde_init=True, return_init=True)
# t_neural_sq, u_neural_sq, u_neural_sq_init = neural_ode_solver(L, alpha2, num_iters=1000, rde_init=True, return_init=True)
t_neural_sin, u_neural_sin = neural_ode_solver(L, alpha1, num_iters=1000, rde_init=False)
t_neural_sq, u_neural_sq = neural_ode_solver(L, alpha2, num_iters=1000, rde_init=False)

# Plot the control in two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(t_rde_sin, u_rde_sin, color='r', label='RDE Solution')
# ax1.plot(t_neural_sin, u_neural_sin_init, color='dodgerblue', label='Neural ODE Initialization')
ax1.plot(t_neural_sin, u_neural_sin, color='b', label='Neural ODE Solution')
ax1.set_xlabel('Time')
ax1.set_ylabel('Control')
ax1.set_title(r'$\alpha(t) = \sin(10t)$')

ax2.plot(t_rde_sq, u_rde_sq, color='r', label='RDE Solution')
# ax2.plot(t_neural_sq, u_neural_sq_init, color='dodgerblue', label='Neural ODE Initialization')
ax2.plot(t_neural_sq, u_neural_sq, color='b', label='Neural ODE Solution')
ax2.set_xlabel('Time')
ax2.set_ylabel('Control')
ax2.set_title(r'$\alpha(t) = t^2$')

plt.suptitle('RDE vs Neural ODE Solutions')
legend, labels = ax1.get_legend_handles_labels()
fig.legend(legend, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3)

plt.show()