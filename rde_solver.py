import numpy as np
from scipy.integrate import solve_ivp

def rde_solver(L, alpha):
    """
    Compute the solution of the LQR problem using the RDE.
    Args:
        L: Parameter lambda for the action cost
        alpha: Time-varying resistance function
    Returns:
        t: Time points
        y: Solution of the optimal control u^\ast(t)
    """
    def rde_func(t, y, *args):
        L, alpha = args
        dydt = np.zeros_like(y)
        dydt[0] = 1/L * y[1]**2
        dydt[1] = -y[0] + alpha(t) * y[1] + 1/L * y[1] * y[2]
        dydt[2] = -2 * y[1] + 2 * alpha(t) * y[2] + 1/L * y[2]**2
        return dydt
    
    def dynamics_func(t, y, *args):
        # y' = A(t)y + B(t)u
        # A(t) = [[0, 1], [0, -alpha(t)]]
        # B(t) = [[0], [1]]
        # u(t) = -1/L * B^T(t)P(t)y(t)
        # augmented state vector z = [z1, z2, p1, p2, p3]
        L, alpha = args
        A = np.array([[0, 1], [0, -alpha(t)]])
        B = np.array([[0], [1]])
        P = np.array([[y[2], y[3]], [y[3], y[4]]])
        z = np.array([y[0], y[1]])
        u = -1/L * B.T @ P @ z
        dzdt = A @ z + B @ u
        dydt = np.zeros_like(y)
        dydt[0:2] = dzdt
        dydt[2:5] = np.zeros(3)
        dydt[2] = 1/L * y[3]**2
        dydt[3] = -y[2] + alpha(t) * y[3] + 1/L * y[3] * y[4]
        dydt[4] = -2 * y[3] + 2 * alpha(t) * y[4] + 1/L * y[4]**2
        return dydt

    # Parameters and initial conditions
    params = (L, alpha)
    P_T = [1.0, 0.0, 0.0]  # Initial state vector
    t_span = (1.0, 0.0)  # Solve the ODE backward in time

    # Solve the ODE
    backward_solution = solve_ivp(
        fun=rde_func,
        t_span=t_span,
        y0=P_T,
        args=params,
        method='RK45',  # Explicit Runge-Kutta method of order 5(4)
        dense_output=False,
        t_eval=[0.0]  # Only evaluate at t=0
    )

    # Extract results and re-arrange
    P_0 = backward_solution.y.flatten()
    z_0 = np.array([1.0, 0.0])
    augmented_z_0 = np.concatenate((z_0, P_0), axis=0)
    t_span = (0.0, 1.0)
    t_eval = np.linspace(t_span[0], t_span[1], 1000)

    # Solve the ODE forward in time
    forward_solution = solve_ivp(
        fun=dynamics_func,
        t_span=t_span,
        y0=augmented_z_0,
        args=params,
        method='RK45',
        dense_output=True,
        t_eval=t_eval
    )

    # Extract results
    t = forward_solution.t
    augmented_z = forward_solution.y

    # Compute the optimal control u(t) = -1/L * B^T(t)P(t)z(t)
    B = np.array([[0], [1]])
    u = np.zeros(len(t))
    for i in range(len(t)):
        P_i = np.array([[augmented_z[2, i], augmented_z[3, i]], [augmented_z[3, i], augmented_z[4, i]]])
        z_i = augmented_z[0:2, i]
        u[i] = -1/L * B.T @ P_i @ z_i

    return (t, u)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Solve the RDE
    t, u = rde_solver(1/3, lambda t : np.sin(t))
    
    # Plot the solution
    plt.plot(t, u)
    plt.xlabel("Time")
    plt.ylabel("Control")
    plt.title("RDE Solution")
    plt.show()
