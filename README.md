# TADAM(Trust region ADAptive Moment estimation)


ℹ️Summary:

-Tadam approximates the loss up to the second order using the Fisher.

-Tadam approximates the Fisher and reduces the computational burdens to the O(N) level.

-Tadam employs an adaptive trust region scheme to reduce approximate errors and guarantee stability. 

-Tadam evaluates how well it minimizes the loss function and uses this information to adjust the trust region dynamically.

## L2 loss per epoch

![L2 loss per epoch](/images/loss_mse_step.png)
