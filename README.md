# MFA-FSL

# License / 许可证

This project is released under a custom non-commerical license, prohibiting its use for any commerical purposes.

本项目基于自定义非商业许可证发布，禁止用于任何形式的商业用途

# Channel-Robust Few-Shot Specific Emitter Identification Using Meta-Feature Augmentation

The rapid increase in wireless devices has raised significant security and privacy concerns, positioning Specific Emitter Identification (SEI) as a crucial physical-layer security technology. While Deep Learning (DL) has been widely applied to SEI due to its powerful end-to-end nonlinear mapping capabilities, it often requires large amounts of high-quality signal examples, which are laborious and expensive. Moreover, the DL-enabled SEI models have difficulties extracting features from the signal examples in the testing process that are consistent with those from the signal examples in the training process due to the wireless channel perturbation, further resulting in a significant reduction in identification performance. To address these challenges, we propose a channel-robust Few-Shot SEI (FS-SEI) method based on Meta-Feature Augmentation (MFA). Our approach utilizes datasets from base emitters to implement a meta-feature embedding function that extracts generalizable features from a few signal examples of target emitters. We then calculate and calibrate the statistics of these extracted features to describe the feature distribution of target emitters. A Multi-Layer Perceptron (MLP) is subsequently trained on both original and augmented features derived from this distribution, achieving a robust FS-SEI model. Experiments conducted on a Wi-Fi dataset comprising 16 emitter categories—10 as base emitters and 6 as target emitters—demonstrate that our method achieves 93.75% identification accuracy with only 5 signal examples per target emitter, maintaining 92.56% accuracy even under varying wireless channel conditions.

X. Fu et al., "Channel-Robust Few-Shot Specific Emitter Identification Using Meta-Feature Augmentation," 2025 IEEE Wireless Communications and Networking Conference (WCNC), Milan, Italy, 2025, pp. 1-6, doi: 10.1109/WCNC61545.2025.10978187.

# Requirement
pytorch 1.10.2
python 3.6.13

# Dataset
The dataset can be download from the link: 

# E-mail
If you have any question, please feel free to contact us by e-mail (1020010415@njupt.edu.cn, 745003219@qq.com, focusfuxue@gmail.com).
