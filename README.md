# TADAM(Trust region ADAptive Moment estimation)


## ℹ️Summary:

- ### Tadam approximates the loss up to the second order using the Fisher.

- ### Tadam approximates the Fisher and reduces the computational burdens to the O(N) level.

- ### Tadam employs an adaptive trust region scheme to reduce approximate errors and guarantee stability. 

- ### Tadam evaluates how well it minimizes the loss function and uses this information to adjust the trust region dynamically.

## Experiment

- ### We use Tadam to train the deep auto-encodermodel of MNIST, Fashion-MNIST, CIFAR-10, and celebA of 32 × 32 × 1 images after cropping and resizing with the batch size of 128. We train the model ten times and record the validation loss’s mean and standard deviations. we do the same experiments using Adam, Amsgrad, Radam, and Nadam which have the O(N) space and time complexity as Tadam.

## Validation loss per epoch

![L2 loss per epoch](/images/loss_mse_step.png)

- ### Tadam’s red curves stay below other curves, meaning Tadam converges better than
the benchmarks at each epoch.

## Validation loss by varying γ 

![L2 loss per epoch](/images/loss_mse_gamma_up.png)

- ### We evaluate the effect of γ, the hyper-parameter for the trust region boundary on the algorithm using γ = 0.1, 0.2, and 0.25 with fixed η = 0.001, respectively. We observe that Tadam consistently maintains a relatively stable validation loss across the different γ values, suggesting that Tadam’s performance is relatively insensitive to the specific choices of γ.
