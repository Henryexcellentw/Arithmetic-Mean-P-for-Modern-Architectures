# Arithmetic-Mean-P-for-Modern-Architectures
Choosing an appropriate learning rate remains a key challenge in scaling
depth of modern deep networks. The classical maximal update parameteri-
zation (μP) enforces a fixed per-layer update magnitude, which is well suited
to homogeneous multilayer perceptrons (MLPs) but becomes ill-posed in
heterogeneous architectures where residual accumulation and convolutions
introduce imbalance across layers. We introduce Arithmetic-Mean μP
(AM-μP), which constrains not each individual layer but the network-wide
average one-step pre-activation second moment to a constant scale. Com-
bined with a residual-aware He fan-in initialization—scaling residual-branch
weights by the number of blocks (Var[W ] = c/(K · fan-in))—AM-μP yields
width-robust depth laws that transfer consistently across depths. We prove
that, for one- and two-dimensional convolutional networks, the maximal-
update learning rate satisfies η⋆(L) ∝ L−3/2; with zero padding, boundary
effects are constant-level as N ≫ k. For standard residual networks with
general conv+MLP blocks, we establish η⋆(L) = Θ(L−3/2), with L the min-
imal depth. Empirical results across a range of depths confirm the −3/2
scaling law and enable zero-shot learning-rate transfer, providing a unified
and practical LR principle for convolutional and deep residual networks
without additional tuning overhead.
