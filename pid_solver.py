import torch
import numpy as np

@torch.no_grad()
def state_eqn(t, z, u, alpha):
    A = torch.tensor([[0., 1.], [0., -alpha(t)]])
    B = torch.tensor([[0.], [1.]])
    f = A @ z + B @ u
    return f


class PD():
    def __init__(self, Kp=2.0, Kd=0.1):
        self.Kp = Kp
        self.Kd = Kd

    def __call__(self, error, error_dot):
        control = self.Kp * error + self.Kd * error_dot
        return control
    

def pid_solver(L, alpha):
    # Use PID controller as initial guess, collect control trajectory after simulation
    t_0, T, h = 0.0, 1.0, 0.01
    z = torch.tensor([1.0, 0.0])
    pid_controller = PD()
    data_t, data_u = [], []
    for t in torch.arange(t_0, T, h):
        error = -z[0].item() # goal is to minimize x^2
        error_dot = -z[1].item()
        u = pid_controller(error, error_dot)
        z = z + h * state_eqn(t, z, torch.tensor([u]), alpha)
        data_t.append(t)
        data_u.append(u)

    return data_t, data_u


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    t, u = pid_solver(1/3, lambda t : 10)
    plt.plot(t, u)
    plt.xlabel('Time')
    plt.ylabel('Control')
    plt.title('PID Solution')
    plt.show()

    