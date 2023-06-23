# Tadam

We introduce Tadam (Trust region ADAptive Moment estimation), a new optimizer based on the trust region of the second-order approximation of the loss using the Fisher information matrix.
Second-order approximations pose a significant computational burden and necessitate large batch sizes.
We aim to develop a second-order approximation algorithm that minimizes computational and memory demands.
Tadam approximates the loss up to the second order using the Fisher. 
Since estimating the Fisher is expensive in both memory and time, 
Tadam approximates the Fisher and reduces the computational burdens to the $O(N)$ level. Furthermore, Tadam employs an adaptive trust region scheme to reduce approximate errors and guarantee stability.
Tadam evaluates how well it minimizes the loss function and uses this information to adjust the trust region dynamically.
In addition, Tadam adjusts the learning rate internally, even if we provide the learning rate as a fixed constant. 
We run several experiments to measure Tadam's performance against Adam, AMSGrad, Radam, and Nadam, which have the same space and time complexity as Tadam.
The test results show that Tadam outperforms the benchmarks and finds reasonable solutions fast and stably.
