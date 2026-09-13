# Controlled Routing Diagnosis Design

## Root-cause correction

The foundation dataset assigned oracle routes by row index while sampling all feature tensors from the same distribution. The router received only mean-pooled encoded features. Thus route identity was statistically unidentifiable from its input; its near-chance agreement cannot diagnose optimization or architecture. The foundation artifacts and conclusions remain valid for that benchmark and will not be overwritten.

## Controlled benchmark

Each sample receives an oracle family at generation. Family A has independently distributed positions, Family B has a local correlation motif, and Family C has an endpoint correlation motif. These are observable, non-label structural signatures. Labels retain distinct feature, local-relation, and endpoint-context relations. The dataset returns:

- tensor sequence for predictive modules;
- binary prediction label;
- generator-defined oracle route;
- **natural descriptor**: variance, adjacent correlation, endpoint correlation computed from the sequence;
- **explicit metadata**: one-hot family descriptor, used only in a diagnostic condition.

The explicit descriptor is not a realistic deployment signal; it is an identifiability control. It tells us whether router optimization/capacity can learn when the relevant information is supplied.

## Experiments

1. Random and static routers establish route-accuracy references.
2. Oracle router establishes upper-bound route accuracy and conditional-expert prediction.
3. Supervised soft and hard routers train cross-entropy directly on generator labels, using natural and explicit descriptors.
4. End-to-end router trains only task loss through a soft mixture of three frozen, family-specialist score functions. This isolates whether route selection emerges without route labels while keeping the oracle fixed.
5. Small/large router, temperature, metadata, supervision, and compute-penalty conditions are run across three seeds.

The specialist score functions are transparent benchmark instruments, not claims that a generic MLP/GNN/attention has been proven uniquely necessary. Existing trainable modules are also benchmarked per family and the report explicitly distinguishes this suitability check from the router-identifiability probe.

## Interpretation contract

Supervised routing success answers *can the router learn route labels?* End-to-end failure answers *does the current task objective make that route emerge?* Neither proves general adaptive-compute utility. Hard routing latency includes the selection/dispatch overhead; soft routing’s all-expert activation is not reported as a physical compute saving.
