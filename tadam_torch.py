import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer

import random
import numpy as np
import matplotlib.pyplot as plt

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class Tadam(Optimizer):
    """
    TAdamOptimizer is a variant of the Adam optimizer that includes an adaptive trust region 
    and utilizes the Fisher information matrix to adaptively adjust the learning rate, 
    helping the model achieve more stable and efficient convergence.

    Attributes:
        total_steps (int): Total training steps for which the optimizer will run.
        lr (float): Learning rate for parameter updates.
        betas (tuple): Coefficients used for computing running averages of gradient and its square.
        gamma (float): Trust region decay parameter.
        eps (float): Small value to prevent division by zero.
        weight_decay (float): Weight decay (L2 penalty) coefficient.
    """

    def __init__(self, params, total_steps, lr=1e-3, betas=(0.9, 0.999), gamma=0.25, eps=1e-8, weight_decay=0):
        """
        Initialize the TAdamOptimizer with the specified parameters.

        Parameters:
            params (iterable): Parameters to be optimized.
            total_steps (int): Total number of training steps.
            lr (float, optional): Initial learning rate. Default is 1e-3.
            betas (tuple, optional): Coefficients for computing running averages (default is (0.9, 0.999)).
            gamma (float, optional): Trust region decay rate. Default is 0.25.
            eps (float, optional): Small constant to prevent division by zero. Default is 1e-8.
            weight_decay (float, optional): Weight decay coefficient. Default is 0.
        """
        defaults = dict(lr=lr, betas=betas, gamma=gamma, eps=eps, total_steps=total_steps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        """
        Perform a single optimization step.

        Parameters:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.

        Returns:
            loss (float, optional): The loss value, if the closure is provided.
        """
        loss = None
        if closure is not None:
            loss = closure()

        # Loop through each parameter group in the optimizer
        for group in self.param_groups:
            grad_lst = []
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.detach()  # Access the gradient data
                grad_lst.append(grad.view(-1))
            grad_flat = torch.cat(grad_lst)
            
            # Initialize state variables if this is the first update
            state = self.state[p]
            if len(state) == 0:
                state['step'] = 0
                state['m'] = torch.zeros_like(grad_flat)
                state['m_hat'] = torch.zeros_like(grad_flat)
                state['v'] = torch.full_like(grad_flat, group['eps'])
                state['s'] = torch.zeros_like(grad_flat)
                state['dt'] = 1.0
                state['ls_h'] = 0.0
                state['loss_avg'] = 0.0
                state['pr'] = 0.0

            # Increment step count
            state['step'] += 1

            # Extract parameters
            beta1, beta2, gamma, eps = group['betas'][0], group['betas'][1], group['gamma'], group['eps']
            m, m_hat, v, s = state['m'], state['m_hat'], state['v'], state['s']
            
            bias1 = (1 - beta1**state['step'])
            bias2 = (1 - beta2**state['step'])

            # Update moving averages
            v.mul_(beta2).addcmul_(grad_flat - m_hat, (grad_flat - m_hat)*(beta2 - beta2**state['step'])/bias2, value=(1 - beta2))
            v_hat = v / bias2

            m.mul_(beta1).add_(grad_flat, alpha=(1 - beta1))
            m_hat = m / bias1
            
            s.mul_(beta2).addcmul_(grad_flat, grad_flat, value=(1 - beta2))
            s_hat = s / bias2

            # Trust region adjustment using Fisher information approximation
            v.mul_(beta2).addcmul_(grad_flat - m_hat, grad_flat - m_hat, value=(1 - beta2))
            v_hat = v / bias2
            
            fisher_information = (1.0 + (m_hat.square() / (v_hat + eps)).sum()) * v_hat
            trust_region_scale = torch.max(state['dt'] * fisher_information, s_hat.sqrt())
            adjusted_gradient = m_hat * state['dt'] / (trust_region_scale + eps)

            gradients_idx = 0
            # restore gradient shape and update
            for p in group['params']:
                if p.grad is not None:
                    param_size = p.grad.numel()
                    restored_grad = adjusted_gradient[gradients_idx:gradients_idx + param_size].view_as(p.grad)
                    gradients_idx += param_size

                    # Apply weight decay if applicable
                    if group['weight_decay'] != 0:
                        p.data.add_(p.data, alpha=-group['weight_decay'])
            
                    # Update parameters using adjusted gradient
                    p.data.add_(restored_grad, alpha=-group['lr'])

            # Optionally update the trust region using the loss function
            if loss is not None:
                # Adjust trust region based on predicted reduction
                rho = torch.tensor((state['ls_h'] - loss.item()) / max(state['pr'], eps), dtype=torch.float)
                dt_min = torch.tensor((1.0 - gamma) ** ((state['step']) / group['total_steps']), dtype=torch.float)
                dt_max = torch.tensor(1.0 + gamma ** ((state['step']) / group['total_steps']), dtype=torch.float)
                r = torch.where(rho < gamma, dt_min, torch.where(rho> 1-gamma, dt_max, torch.tensor(1.0)))
                state['dt'] = min(max(r * state['dt'], dt_min), dt_max)
                state['pr'] = (m_hat * adjusted_gradient - 0.5 * v_hat * adjusted_gradient.square()).sum().item() * group['lr']
                # Update the moving average of the loss function
                state['loss_avg'] = beta1 * state['loss_avg'] + (1 - beta1) * loss.item()
                state['ls_h'] = state['loss_avg'] / bias1
        return loss

class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(16, 16)
        self.fc2 = nn.Linear(16, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 2)
        self.fc5 = nn.Linear(2, 2)
        self.fc6 = nn.Linear(2, 4)
        self.fc7 = nn.Linear(4, 8)
        self.fc8 = nn.Linear(8, 16)
        self.fc9 = nn.Linear(16, 16)
        self.fc10 = nn.Linear(16, 16)


    def forward(self, x):
        x = F.selu(self.fc1(x))
        x = F.selu(self.fc2(x))
        x = F.selu(self.fc3(x))
        x = F.selu(self.fc4(x))
        x = self.fc5(x)
        x = F.selu(self.fc6(x))
        x = F.selu(self.fc7(x))
        x = F.selu(self.fc8(x))
        x = F.selu(self.fc9(x))
        x = self.fc10(x) 
        return x

if __name__ == "__main__":
    seed = 100
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    deterministic  = True
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Example usage of TAdamOptimizer
    # Define a simple model
    model = SimpleModel()

    # Initialize the Tadam optimizer with betas tuple
    dt = 0
    # optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = Tadam(model.parameters(), total_steps=1000, lr=0.001, betas=(0.9, 0.999), gamma=0.25, eps=1e-8)

    # Define a simple loss function and data
    criterion = nn.MSELoss()
    # input_data = torch.randn(128, 16)  # Batch of input data
    # target_data = torch.randn(128, 2)  # Corresponding target data
    val_data = torch.randn(128, 16)

    # Training loop example
    loss_lst = []
    val_lst = []
    for epoch in range(1000):  # Training for 100 epochs
        input_data = torch.randn(128, 16)
        output = model(input_data)  # Forward pass
        loss = criterion(output, input_data)  # Compute loss
        optimizer.zero_grad()  # Zero the gradients
        loss.backward()  # Backward pass (compute gradients)
        def closure():
            return loss
        loss = optimizer.step(closure)  # Update parameters using Tadam optimizer
        loss_lst.append(loss.item())

        model.eval()
        with torch.no_grad():
            # val_data = torch.randn(128, 16)
            output = model(val_data)
            val_loss = nn.MSELoss()(output, val_data)
            val_lst.append(val_loss.item())

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}, v_loss{val_loss.item()})

    plt.figure()
    plt.plot(loss_lst)
    plt.plot(val_lst)
    plt.show()

