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
t_neural_sin, u_neural_sin, u_neural_sin_init = neural_ode_solver(L, alpha1, num_iters=1000, rde_init=True, return_init=True)
t_neural_sq, u_neural_sq, u_neural_sq_init = neural_ode_solver(L, alpha2, num_iters=1000, rde_init=True, return_init=True)
# t_neural_sin, u_neural_sin = neural_ode_solver(L, alpha1, num_iters=1000, rde_init=False)
# t_neural_sq, u_neural_sq = neural_ode_solver(L, alpha2, num_iters=1000, rde_init=False)

# Plot the control in two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(t_rde_sin, u_rde_sin, color='r', label='RDE Solution')
ax1.plot(t_neural_sin, u_neural_sin_init, color='dodgerblue', label='Neural ODE Initialization')
ax1.plot(t_neural_sin, u_neural_sin, color='b', label='Neural ODE Solution')
ax1.set_xlabel('Time')
ax1.set_ylabel('Control')
ax1.set_title(r'$\alpha(t) = \sin(10t)$')
ax1.set_ylim(-0.9, 0.1)

ax2.plot(t_rde_sq, u_rde_sq, color='r', label='RDE Solution')
ax2.plot(t_neural_sq, u_neural_sq_init, color='dodgerblue', label='Neural ODE Initialization')
ax2.plot(t_neural_sq, u_neural_sq, color='b', label='Neural ODE Solution')
ax2.set_xlabel('Time')
ax2.set_ylabel('Control')
ax2.set_title(r'$\alpha(t) = t^2$')
ax2.set_ylim(-0.9, 0.1)

plt.suptitle('RDE vs Neural ODE Solutions')
legend, labels = ax1.get_legend_handles_labels()
fig.legend(legend, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=3)

plt.show()

# Plot the position trajectories
@torch.no_grad()
def state_eqn(t, z, u, alpha):
    A = torch.tensor([[0., 1.], [0., -alpha(t)]]).float()
    B = torch.tensor([[0.], [1.]]).float()
    f = A @ z + B @ u
    return f

@torch.no_grad()
def get_pos_traj(t, u, alpha):
    z = torch.tensor([1., 0.]).float()
    h = (t[1] - t[0]).item()
    pos_traj = [z[0].numpy()]
    for i in range(len(t) - 1):
        u_i = u[i]
        z = z + h * state_eqn(t[i], z, u_i, alpha)
        pos_traj.append(z[0].numpy())
    return np.array(pos_traj)

u_rde_sin = torch.tensor(u_rde_sin).float().unsqueeze(-1)
u_rde_sq = torch.tensor(u_rde_sq).float().unsqueeze(-1)
u_neural_sin = torch.tensor(u_neural_sin).float()
u_neural_sq = torch.tensor(u_neural_sq).float()

pos_traj_rde_sin = get_pos_traj(t_rde_sin, u_rde_sin, alpha1)
pos_traj_rde_sq = get_pos_traj(t_rde_sq, u_rde_sq, alpha2)
pos_traj_neural_sin = get_pos_traj(t_neural_sin, u_neural_sin, alpha1)
pos_traj_neural_sq = get_pos_traj(t_neural_sq, u_neural_sq, alpha2)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

ax1.plot(t_rde_sin, pos_traj_rde_sin, color='r', label='RDE Solution')
ax1.plot(t_neural_sin, pos_traj_neural_sin, color='b', label='Neural ODE Solution')
ax1.set_xlabel('Time')
ax1.set_ylabel('Position')
ax1.set_title(r'$\alpha(t) = \sin(10t)$')

ax2.plot(t_rde_sq, pos_traj_rde_sq, color='r', label='RDE Solution')
ax2.plot(t_neural_sq, pos_traj_neural_sq, color='b', label='Neural ODE Solution')
ax2.set_xlabel('Time')
ax2.set_ylabel('Position')
ax2.set_title(r'$\alpha(t) = t^2$')

plt.suptitle('RDE vs Neural ODE Solutions')
legend, labels = ax1.get_legend_handles_labels()
fig.legend(legend, labels, loc='lower center', bbox_to_anchor=(0.5, -0.01), ncol=2)

plt.show()

# Also plot them together for RDE but with different alpha
fig, ax = plt.subplots(figsize=(8, 6))

ax.plot(t_rde_sin, pos_traj_rde_sin, color='r', label=r'$\alpha(t) = \sin(10t)$')
ax.plot(t_rde_sq, pos_traj_rde_sq, color='b', label=r'$\alpha(t) = t^2$')
ax.set_xlabel('Time')
ax.set_ylabel('Position')

plt.title('RDE Solutions with Different Resistance')

plt.legend()

plt.show()

