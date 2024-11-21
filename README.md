# TADAM(Trust region ADAptive Moment estimation)


## ℹ️ Summary:

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
- We evaluate the impact of $\gamma$, we use $\gamma$ values of $0.1$, $0.2$, and $0.25$ while maintaining a fixed learning rate $\eta$ of $0.001$, respectively. We observe that Tadam consistently maintains a relatively stable validation loss across the different $\gamma$ values, suggesting that Tadam's performance is relatively insensitive to the specific choices of $\gamma$.

# Q&A

### Q. I don't quite understand the update equation for v_n in your Algorithm 1. Why is the expression MA(g_n - gbar_n-1)(g_n - gbar_n)? The gbar_n-1 term is a little surprising to me.

A. Initially, we searched for references on how others handle the moving average of the second moment, and we found both MA(g_n - gbar_n)(g_n - gbar_n) and MA(g_n - gbar_n-1)(g_n - gbar_n). We experimented using both representations; the second performed better than the first, and we reported only the second in the paper. g_n is the current gradient, gbar_n is the moving average containing the current, and  gbar_n-1 is without the current. So, MA(g_n - gbar_n-1)(g_n - gbar_n) is a mixture of (backward) Nesterov momentum moving average and more traditional moving average.

### Q. The "for n = 1 to N" loop in Algorithm 1, does n represent the n-th sample, n-th mini-batch, or n-th epoch? 

A. One likely uses adam in the code when one trains a model. To use tadam instead of adam, just add t in front of adam, i.e., change adam to tadam. That is the original intention of our algorithm. One can interpret the for loop in Algorithm 1 in this respect. For our experiment setting, however, to quickly observe the difference between the adam and tadam, we update the model parameters for each mini-batch.

# Paper
https://www.sciencedirect.com/science/article/abs/pii/S089360802300504X
