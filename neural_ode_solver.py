import torch
import numpy as np
from tqdm import tqdm

class MLP(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, zero_init=False):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_size, hidden_size)
        self.fc2 = torch.nn.Linear(hidden_size, output_size)

        # Initialize weights
        if zero_init:
            torch.nn.init.zeros_(self.fc1.weight)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.zeros_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc2.bias)
        else:
            torch.nn.init.xavier_uniform_(self.fc1.weight)
            torch.nn.init.zeros_(self.fc1.bias)
            torch.nn.init.xavier_uniform_(self.fc2.weight)
            torch.nn.init.zeros_(self.fc2.bias)

        self.act_fn = torch.nn.Tanh()

    def forward(self, x):
        x = self.act_fn(self.fc1(x))
        x = self.fc2(x)
        return x

@torch.no_grad()
def state_eqn(t, z, u, alpha):
    A = torch.tensor([[0., 1.], [0., -alpha(t)]])
    B = torch.tensor([[0.], [1.]])
    f = A @ z + B @ u
    return f

@torch.no_grad()
def costate_eqn(t, p, alpha):
    A = torch.tensor([[0., 1.], [0., -alpha(t)]])
    return -A.T @ p

def hamiltonian(t, z, p, u, *args):
    L, alpha = args
    A = torch.tensor([[0., 1.], [0., -alpha(t)]])
    B = torch.tensor([[0.], [1.]])
    f = A @ z + B @ u
    H = p @ f - L * u ** 2
    return H

def update_w(w, t, z, p, model, L, alpha, h):
    # clear the gradients
    model.zero_grad(set_to_none=True)

    # Compute Hamiltonian and backpropagate
    u = model(torch.tensor([t]))
    H = -hamiltonian(t, z, p, u, L, alpha) # the goal is to compute -\nabla_\theta H
    H.backward()

    # Update w
    with torch.no_grad():
        w.fc1.weight -= h * model.fc1.weight.grad
        w.fc1.bias -= h * model.fc1.bias.grad
        w.fc2.weight -= h * model.fc2.weight.grad
        w.fc2.bias -= h * model.fc2.bias.grad

    return w


class PD():
    def __init__(self, Kp=2.0, Kd=0.1):
        self.Kp = Kp
        self.Kd = Kd

    def __call__(self, error, error_dot):
        control = self.Kp * error + self.Kd * error_dot
        return control
    

def neural_ode_solver(L, alpha, num_iters=500, lr=0.05, decay_rate=0.999, time_iterval=0.05, pd_init=False, rde_init=True, pre_train_iters=5000):
    hidden_dim = 10
    model = MLP(1, hidden_dim, 1)

    if pd_init:
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

        # Fit a neural network to the control trajectory
        progress_bar = tqdm(range(pd_train_iters), desc="Fitting PID Controller")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        t, u = torch.tensor(data_t).reshape(-1, 1), torch.tensor(data_u).reshape(-1, 1)
        for _ in progress_bar:
            optimizer.zero_grad()

            loss = torch.mean((model(t) - u) ** 2)

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'Loss': loss.item()})

    elif rde_init:
        # Use RDE solution as initial guess
        from rde_solver import rde_solver
        t_rde, u_rde = rde_solver(L, alpha)
        t, u = torch.tensor(t_rde).reshape(-1, 1).float(), torch.tensor(u_rde).reshape(-1, 1).float()

        # Fit a neural network to the control trajectory
        progress_bar = tqdm(range(pre_train_iters), desc="Fitting RDE Solution")
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        for _ in progress_bar:
            optimizer.zero_grad()

            loss = torch.mean((model(t) - u) ** 2)

            loss.backward()
            optimizer.step()

            progress_bar.set_postfix({'Loss': loss.item()})

    # add random noise to the model parameters
    with torch.no_grad():
        sigma = 0.05
        model.fc1.weight += sigma * torch.randn_like(model.fc1.weight)
        model.fc1.bias += sigma * torch.randn_like(model.fc1.bias)
        model.fc2.weight += sigma * torch.randn_like(model.fc2.weight)
        model.fc2.bias += sigma * torch.randn_like(model.fc2.bias)

    progress_bar = tqdm(range(num_iters), desc="Training Neural ODE")
    for _ in progress_bar:
        # forward pass to compute z_T
        with torch.no_grad():
            t_0, T, h = 0.0, 1.0, time_iterval
            z = torch.tensor([1.0, 0.0])
            for t in torch.arange(t_0, T, h):
                u = model(torch.tensor([t]))
                z = z + h * state_eqn(t, z, u, alpha)
        z_T = z.detach()

        # backward pass to compute w_0
        M = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        augmented_z = {
            'z': z_T,
            'p': -2 * M @ z_T,
            'w': MLP(1, hidden_dim, 1, zero_init=True)
        }
        for t in torch.arange(T, t_0, -h):
            new_z = augmented_z['z'] - h * state_eqn(t, augmented_z['z'], model(torch.tensor([t])).detach(), alpha)
            new_p = augmented_z['p'] - h * costate_eqn(t, augmented_z['p'], alpha)
            new_w = update_w(augmented_z['w'], t, augmented_z['z'], augmented_z['p'], model, L, alpha, h)
            augmented_z.update({
                'z': new_z,
                'p': new_p,
                'w': new_w
            })

        # extract gradient and update model
        w_0 = augmented_z['w']
        with torch.no_grad():
            model.fc1.weight += lr * w_0.fc1.weight
            model.fc1.bias += lr * w_0.fc1.bias
            model.fc2.weight += lr * w_0.fc2.weight
            model.fc2.bias += lr * w_0.fc2.bias

            grad_norm = torch.norm(w_0.fc1.weight) + torch.norm(w_0.fc1.bias) + torch.norm(w_0.fc2.weight) + torch.norm(w_0.fc2.bias)

        lr *= decay_rate # learning rate decay

        progress_bar.set_postfix({'Grad Norm': grad_norm.item()})

        # # early stopping
        # if grad_norm < 1e-3:
        #     lr = 0.1
        # else:
        #     lr = 0.05

    # Return t, u
    t = torch.linspace(0, 1, 100).reshape(-1, 1)
    u = model(t).detach().numpy()
    return t.numpy(), u


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # Set random seed
    torch.manual_seed(0)

    # Solve the Neural ODE
    t, u = neural_ode_solver(1/3, lambda t : np.sin(t), pd_init=False)
    
    # Plot the solution
    plt.plot(t, u)
    plt.xlabel("Time")
    plt.ylabel("Control")
    plt.title("Neural ODE Solution")
    plt.show()