# Variance-Aware Adaptive Weighting for Diffusion Model Training
📍Offical Pytorch Implementation

📖 Paper: Variance-Aware Adaptive Weighting for Diffusion Model Training
--------------------------------------------------------------

&nbsp;&nbsp;&nbsp;&nbsp; Authors: Nanlong Sun, Lei Shi
  
&nbsp;&nbsp;&nbsp;&nbsp; Submitted to Neurocomputing
  
📝link: https://arxiv.org/abs/

🌎 Abstract
--------------------------------------------------------------

Diffusion models have recently achieved remarkable success in generative modeling, yet their 
training dynamics across different noise levels remain highly imbalanced, which can lead to 
inefficient optimization and unstable learning behavior. In this work, we investigate this 
imbalance from the perspective of loss variance across log-SNR levels and propose a variance-aware adaptive weighting strategy to address it. The proposed approach dynamically adjusts 
training weights based on the observed variance distribution, encouraging a more balanced 
optimization process across noise levels. Extensive experiments on CIFAR-10 and CIFAR-100 
demonstrate that the proposed method consistently improves generative performance over 
standard training schemes, achieving lower Fréchet Inception Distance (FID) while also 
reducing performance variance across random seeds. Additional analysis, including loss-log-SNR visualization, variance heatmaps, and ablation studies, further reveal that the adaptive 
weighting effectively stabilizes training dynamics. These results highlight the potential of 
variance-aware training strategies for improving diffusion model optimization.

🗺️ Framework
--------------------------------------------------------------

<p align='center'>
  <img src='pipeline.jpg' width='600'>
</p>

🎯 Method
--------------------------------------------------------------

### Overview

Diffusion models are commonly trained using a noise prediction objective across different noise levels. However, the training process can be highly **imbalanced across log-SNR regions**, leading to large variance in per-sample losses.

To address this issue, we introduce a **variance-aware adaptive reweighting strategy** that dynamically adjusts the contribution of each training sample.

---

### Diffusion Training Objective

The standard diffusion training objective predicts the added noise:

$$
\mathcal{L} = \mathbb{E}_{x,\epsilon,t}
\left[
\|\epsilon - \epsilon_\theta(x_t,t)\|^2
\right]
$$

where

- $x$ is the clean data sample  
- $\epsilon \sim \mathcal{N}(0,I)$ is Gaussian noise  
- $\epsilon_\theta$ is the noise prediction network  

The noisy sample is constructed as

$$
x_t = \alpha_t x + \sigma_t \epsilon
$$

Many diffusion frameworks parameterize the noise level using **log-SNR**:

$$
\lambda = \log\left(\frac{\alpha_t^2}{\sigma_t^2}\right)
$$

---

### Adaptive Loss Reweighting

Let the per-sample loss be

$$
\ell_i = \|\epsilon_i - \epsilon_\theta(x_{t_i},t_i)\|^2
$$

We compute a normalized adaptive weight:

$$
w_i = 1 + \alpha \frac{\ell_i - \mu}{\sigma}
$$

where $\mu$ and $\sigma$ are the batch mean and standard deviation.

The final training objective becomes
  
$$
\mathcal{L}_{adaptive} = \mathbb{E}(w_i \cdot \ell_i)
$$

This mechanism increases the influence of harder samples and encourages **more balanced optimization across log-SNR levels**.
