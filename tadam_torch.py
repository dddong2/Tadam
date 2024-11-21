import torch
from torch.optim.optimizer import Optimizer

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
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data  # Access the gradient data

                # Initialize state variables if this is the first update
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['m_hat'] = torch.zeros_like(p.data)
                    state['v'] = torch.full_like(p.data, group['eps'])
                    state['s'] = torch.zeros_like(p.data)
                    state['dt'] = 1.0
                    state['ls_h'] = 0.0
                    state['loss_avg'] = 0.0
                    state['pr'] = 0.0

                # Increment step count
                state['step'] += 1

                # Extract parameters
                beta1, beta2, gamma, eps = group['betas'][0], group['betas'][1], group['gamma'], group['eps']
                m, m_hat, v, s = state['m'], state['m_hat'], state['v'], state['s']

                # Update moving averages
                v.mul_(beta2).addcmul_(grad - m_hat, (grad - m_hat)*(beta2 - beta2**state['step'])/(1 - beta2**state['step']), value=(1 - beta2))
                v_hat = v / (1 - beta2**state['step'])

                m.mul_(beta1).add_(grad, alpha=(1 - beta1))
                m_hat = m / (1 - beta1**state['step'])
                
                s.mul_(beta2).addcmul_(grad, grad, value=(1 - beta2))
                s_hat = s / (1 - beta2**state['step'])

                # Trust region adjustment using Fisher information approximation
                v.mul_(beta2).addcmul_(grad - m_hat, grad - m_hat, value=(1 - beta2))
                v_hat = v / (1 - beta2**state['step'])
                
                fisher_information = (1.0 + torch.sum(torch.square(m_hat) / (v_hat + eps))) * v_hat
                trust_region_scale = torch.max(state['dt'] * fisher_information, torch.sqrt(s_hat))
                adjusted_gradient = m_hat * state['dt'] / (trust_region_scale + eps)

                # Apply weight decay if applicable
                if group['weight_decay'] != 0:
                    p.data.add_(p.data, alpha=-group['weight_decay'])
                
                # Update parameters using adjusted gradient
                p.data.add_(adjusted_gradient, alpha=-group['lr'])

                # Optionally update the trust region using the loss function
                if loss is not None:
                    
                    # Adjust trust region based on predicted reduction
                    rho = (state['ls_h'] - loss.item()) / max(state['pr'], eps)
                    dt_min = (1.0 - gamma) ** ((state['step'] - 1) / group['total_steps'])
                    dt_max = 1.0 + gamma ** ((state['step'] - 1) / group['total_steps'])
                    state['dt'] = min(max(rho * state['dt'], dt_min), dt_max)
                    state['pr'] = ((m_hat * adjusted_gradient).sum() - 0.5 * (v_hat * adjusted_gradient ** 2).sum()).item() * group['lr']
                    
                    # Update the moving average of the loss function
                    state['loss_avg'] = beta1 * state['loss_avg'] + (1 - beta1) * loss.item()
                    state['ls_h'] = state['loss_avg'] / (1 - beta1**state['step'])
        return loss


if __name__ == "__main__":
    # Example usage of TAdamOptimizer
    # Define a simple model
    model = torch.nn.Linear(10, 1)  # Example model

    # Initialize the Tadam optimizer with betas tuple
    optimizer = Tadam(model.parameters(), total_steps=1000, lr=0.001, betas=(0.9, 0.999), gamma=0.25, eps=1e-8)

    # Define a simple loss function and data
    criterion = torch.nn.MSELoss()
    input_data = torch.randn(32, 10)  # Batch of input data
    target_data = torch.randn(32, 1)  # Corresponding target data

    # Training loop example
    for epoch in range(100):  # Training for 100 epochs
        optimizer.zero_grad()  # Zero the gradients
        output = model(input_data)  # Forward pass
        loss = criterion(output, target_data)  # Compute loss
        loss.backward()  # Backward pass (compute gradients)
        optimizer.step()  # Update parameters using Tadam optimizer

        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

