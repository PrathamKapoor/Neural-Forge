# Phase 3 — Capacity-matched expert specialization

## Results

| Architecture | Feature | Relational | Contextual |
|---|---:|---:|---:|
| mlp | 1.000 | 0.608 | 0.468 |
| graph | 1.000 | 0.714 | 0.469 |
| attention | 0.750 | 0.518 | 0.485 |

## Capacity and controls

Depths: {'mlp': 3, 'graph': 2, 'attention': 1}; parameters: {'mlp': 3914, 'graph': 3866, 'attention': 3314}; maximum relative gap: 15.3%.

GraphBlock constructs its ring internally, so literal external edge randomization was not implemented. The relational control is deterministic node-feature permutation relative to fixed ring topology: it retains values, shape, degree, parameters, and block implementation while disrupting feature/topology correspondence.

## Interpretation

These serialized data are the authoritative evidence. Rankings and controls must be assessed together; no claim of learned routing is made in Phase 3.