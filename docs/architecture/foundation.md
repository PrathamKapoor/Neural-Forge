# Architecture: adaptive heterogeneous computation foundation

## Data flow

`[B,S,D] input → shared linear encoder → learned router → selected block(s) → shared sequence state → pooled classifier`.

The MLP route mean-pools a sequence, transforms its feature vector, then broadcasts it back. This is economical but loses ordering. The graph route explicitly creates a ring adjacency and performs local message passing; it retains node features but adds a topology assumption. The attention route lets all positions interact and has quadratic sequence cost. All return the same `[B,S,H]` shared state only after their representation-specific computation.

## Router

The router mean-pools the shared state and uses a two-layer MLP to produce three logits. Softmax temperature produces module weights. In training, a weighted mixture makes the route differentiable but computes every block. In hard evaluation, the top-1 route dispatches only its selected sample subset to each block. These modes are deliberately reported separately: soft-mode estimated sparsity is not a latency claim.

## Cost and failure behavior

Each adapter and block returns a transparent analytical proxy for compute; it is not a hardware FLOP profiler. Wall-clock batch latency is separately measured after warmup. Diagnostics flag >98% average utilization by one route, near-uniform/all-module-like routing, and top-1 low diversity. The current model has one routing stage; dynamic multi-step depth and early exits are not yet implemented.
