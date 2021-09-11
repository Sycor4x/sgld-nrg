# sgld-nrg
Implements Stochastic gradient Langevin dynamics for energy-based models, as per https://arxiv.org/pdf/1912.03263.pdf

This paper discovers that classifier networks contain an extra degree of freedom in the logits of their predictions. This extra degree of freedom allows us to model the data distribution p(X). This allows researchers to realize the long-promised benefits of generative neural networks: out-of-distribution detection, model calibration, and adversarial robustness, while the classifier itself remains very effective.

This repository implements SGLD and energy-based training for classifier networks on common toy datasets. One side-effect of SGLD training is that the classifier network itself can also be used as part of a generative MCMC mechanism. In this sense, classifier networks are also generative models.

These are 64 samples from young MCMC chains at the start of training the energy model. There are some vaguely-digit-shaped blobs among the random noise. They're blurry noisy because they are initialized from uniform random noise, and only gradually refined based on the gradient of its energy wrt the image (i.e. SGLD). Further evolution and computing gradients from model trained to optimize energy will improve them.
![plot](./results/beginning.png)

These are 64 samples from older MCMC chains. After using energy-based training for about half an epoch and evolving the chains during that time, they are recognizable as digits for the most part. Some of the images are fuzzier than others due to re-initialization to random noise.
![plot](./results/half-epoch.png)

As of this commit, the repository is solely a proof-of-concept implementation based on the details in the paper. My TODO list includes
- creating a package with suitable objects for general use
- implementing mini-batching for SGLD
- improving the SGLD implementation itself for reproducibility and speed
- improving the argparse interface to allow more fine-grained control of the model estimation
- testing whether the models still work if we jettison the classifier-only pre-training and buffer initialization (these were implemented as a way to validate that the JEM model could actually do what we needed in the _best possible circumstances_, i.e. having a pretty good classifier off the bat to start JEM training)
- apply JEM to problems that are harder than MNIST digits (MNIST digits is know to be an easy task for CNNs, so it's a nice place to start for proof-of-concept code).
