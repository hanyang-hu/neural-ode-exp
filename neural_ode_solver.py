import numpy as np
from scipy.integrate import solve_ivp
import jax.numpy as jnp
from jax import grad
from tqdm import tqdm

def flatten_parameters(theta):
    return jnp.concatenate([param.flatten() for param in theta])

def unflatten_parameters(flatten_theta):
    shapes = [(64, 1), (64,), (1, 64), (1,)]
    params = []
    idx = 0
    for shape in shapes:
        size = np.prod(shape)
        params.append(flatten_theta[idx:idx + size].reshape(shape))
        idx += size
    return params

def mlp(t, flatten_theta):
    theta = unflatten_parameters(flatten_theta)
    W1, b1, W2, b2 = theta
    x = jnp.array([t])
    h = jnp.tanh(W1 @ x + b1)
    y = W2 @ h + b2
    return y

def dynamics_func(t, y, *args):
    alpha, theta = args
    A = jnp.array([[0, 1], [0, -alpha(t)]])
    B = jnp.array([[0], [1]])
    u = mlp(t, theta)
    return A @ y + B @ u

def hamiltonian(t, z, p, flatten_theta, *args):
    L, alpha = args
    f = dynamics_func(t, z, alpha, flatten_theta)
    u = mlp(t, flatten_theta)
    return (p @ f - L * u**2).reshape(())

def backward_augmented_ode(t, y, *args):
    z, p, _ = jnp.split(y, [2, 4])
    L, alpha, flatten_theta = args

    # Compute gradients
    grad_H_z = grad(hamiltonian, argnums=1)(t, z, p, flatten_theta, L, alpha)
    grad_H_p = grad(hamiltonian, argnums=2)(t, z, p, flatten_theta, L, alpha)
    grad_H_theta = grad(hamiltonian, argnums=3)(t, z, p, flatten_theta, L, alpha)

    dz_dt = grad_H_p
    dp_dt = -grad_H_z
    dw_dt = grad_H_theta

    return jnp.concatenate([dz_dt, dp_dt, dw_dt])

def neural_ode_solver(L, alpha, iterations=10, lr=0.01):
    # initialize theta parameter
    input_dim, hidden_dim, output_dim = 1, 64, 1
    theta = [
        jnp.zeros((hidden_dim, input_dim)),
        jnp.zeros(hidden_dim),
        jnp.zeros((output_dim, hidden_dim)),
        jnp.zeros(output_dim)
    ]
    flatten_theta = flatten_parameters(theta)

    progress_bar = tqdm(range(iterations))
    for _ in progress_bar:
        # forward pass to compute z(T)
        t_span = (0.0, 1.0)
        z_0 = jnp.array([1.0, 0.0])
        params = (alpha, flatten_theta)
        solution = solve_ivp(
            fun=dynamics_func,
            t_span=t_span,
            y0=z_0,
            args=params,
            method='RK45',
            dense_output=False,
            t_eval=[1.0]
        )
        z_T = solution.y.flatten()

        # backward pass to solve the augmented ODE
        t_span = (1.0, 0.0)
        augmented_z_T = jnp.concatenate([z_T, jnp.zeros_like(z_T), jnp.zeros_like(flatten_theta)])
        params = (L, alpha, flatten_theta)
        backward_solution = solve_ivp(
            fun=backward_augmented_ode,
            t_span=t_span,
            y0=augmented_z_T,
            args=params,
            method='RK45',
            dense_output=False,
            t_eval=[0.0]
        )
        y_0 = backward_solution.y.flatten()

        flatten_w_0 = y_0[4:]

        # update theta
        flatten_theta = flatten_theta + lr * flatten_w_0
        theta = unflatten_parameters(flatten_theta)

        progress_bar.set_description(f"Grad Norm: {jnp.linalg.norm(flatten_w_0):.6f}")

    # evaluate the optimal control
    t_eval = np.linspace(t_span[0], t_span[1], 1000)
    u = np.array([mlp(t, theta) for t in t_eval])

    return t_eval, u


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Solve the RDE
    t, u = neural_ode_solver(100, lambda t : 0)
    
    # Plot the solution
    plt.plot(t, u)
    plt.xlabel("Time")
    plt.ylabel("Control")
    plt.title("Neural ODE Solution")
    plt.show()





    