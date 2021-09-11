# sgld-nrg
Implements Stochastic gradient Langevin dynamics for energy-based models, as per https://arxiv.org/pdf/1912.03263.pdf

This paper discovers that classifier networks contain an extra degree of freedom in the logits of their predictions. This extra degree of freedom allows us to model the data distribution p(X). This allows researchers to realize the long-promised benefits of generative neural networks: out-of-distribution detection, model calibration, and adversarial robustness, while the classifier itself remains very effective.

This repository implements SGLD and energy-based training for classifier networks on common toy datasets. One side-effect of SGLD training is that the classifier network itself can also be used as part of a generative MCMC mechanism. In this sense, classifier networks are also generative models.

These are 64 samples from young MCMC chains at the start of training the energy model. They're blurry noisy because they are initialized from uniform random noise, and gradually refined based on the gradient of its energy wrt the image (i.e. SGLD).
![plot](./results/beginning.png)
![plot](./results/half-epoch.png)
