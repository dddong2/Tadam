# TADAM(Trust region ADAptive Moment estimation)


## ℹ️Summary:

-Tadam approximates the loss up to the second order using the Fisher.

-Tadam approximates the Fisher and reduces the computational burdens to the O(N) level.

-Tadam employs an adaptive trust region scheme to reduce approximate errors and guarantee stability. 

-Tadam evaluates how well it minimizes the loss function and uses this information to adjust the trust region dynamically.

## We use Tadam to train the deep auto-encoder.  we do the same experiments using Adam, Amsgrad, Radam, and Nadam which have the O(N) space and time complexity as Tadam.

## L2 loss per epoch

![L2 loss per epoch](/images/loss_mse_step.png)

## L2 loss per time

![L2 loss per time](/images/loss_mse_time.png)

## Huber loss per epoch

![huber loss per epoch](/images/loss_huber_step.png)

## Huber loss per time

![huber loss per time](/images/loss_huber_time.png)
