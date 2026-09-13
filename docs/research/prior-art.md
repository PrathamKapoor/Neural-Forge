# Prior-art boundary

NeuroForge is not claiming conditional computation, learned routing, or adaptive depth as novel. Sparse MoE uses a learned gate to activate a sparse expert combination ([Shazeer et al., 2017](https://arxiv.org/abs/1701.06538)). Adaptive Computation Time learns variable recurrent steps ([Graves, 2016](https://arxiv.org/abs/1603.08983)). BlockDrop learns image-conditioned residual-block execution paths and reports both computation and measured inference results ([Wu et al., 2018](https://arxiv.org/abs/1711.08393)).

This repository's initial contribution is a reproducible *experimental harness*: a known-oracle, per-sample mixed-structure task; explicit representation adapters for non-identical MLP/graph/attention inputs; and side-by-side reporting of soft-routing cost proxies and hard-dispatch latency. Whether that framing is scientifically useful remains under investigation; it is not a novelty claim.
