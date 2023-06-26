# TADAM(Trust region ADAptive Moment estimation)


## ℹ️Summary:

- ### Tadam approximates the loss up to the second order using the Fisher.

- ### Tadam approximates the Fisher and reduces the computational burdens to the O(N) level.

- ### Tadam employs an adaptive trust region scheme to reduce approximate errors and guarantee stability. 

- ### Tadam evaluates how well it minimizes the loss function and uses this information to adjust the trust region dynamically.

<br><br>

## Experiment
-  We use our Tadam to train the deep auto-encoder. The training data sets are MNIST, Fashion-MNIST, CIFAR-10, and celebA. We train each auto-encoder ten times and record the loss's mean and standard deviations. Tadam exhibits a space and time complexity of $O(N)$, placing it on par with other widely used optimizers such as Adam, AMSGrad, Radam, and Nadam.



### Validation loss per epoch

![L2 loss per epoch](/images/loss_mse_step.png)

- Tadam converges faster than the benchmarks.

<br>

### Validation loss by varying $\gamma$

![L2 loss per epoch](/images/loss_mse_gamma_up.png)

- We use the hyper-parameter $\gamma \in (0, 0.25]$ to measure Tadam's training performance and update the $\delta_n$, which controls the trust region size. 
- We evaluate the effect of $\gamma$, we use $\gamma$ values of $0.1$, $0.2$, and $0.25$ while maintaining a fixed learning rate $\eta$ of $0.001$, respectively. We observe that Tadam consistently maintains a relatively stable validation loss across the different $\gamma$ values, suggesting that Tadam's performance is relatively insensitive to the specific choices of $\gamma$.
