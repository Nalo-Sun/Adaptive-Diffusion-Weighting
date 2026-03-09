# Variance-Aware Adaptive Weighting for Diffusion Model Training
📍Offical Pytorch Implementation

📖Paper: Variance-Aware Adaptive Weighting for Diffusion Model Training
--------------------------------------------------------------

&nbsp;&nbsp;&nbsp;&nbsp; Authors: Nanlong Sun, Lei Shi
  
&nbsp;&nbsp;&nbsp;&nbsp; Submitted to Neurocomputing
  
📝link: https://arxiv.org/abs/

🌎Abstract
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

🗺️Framework
--------------------------------------------------------------
