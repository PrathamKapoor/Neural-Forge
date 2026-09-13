# Foundation CPU report — 2026-09-04

## Environment

Windows 11, Python 3.13.14, PyTorch 2.13.0 CPU; seed 42. Full machine-readable evidence is in `results/metrics/foundation_cpu/`.

## Result

The initial hypothesis is **not supported by this run**. The best fixed model was graph-only (accuracy 0.641, estimated compute 1740), while adaptive soft routing reached 0.635 at 1571 proxy units. Hard dispatch reached 0.615 at 1446 proxy units, lower accuracy and slower CPU batch latency (2.95 ms) than graph-only (1.64 ms).

## Routing analysis

Soft weights strongly suppressed attention (mean 0.004). Hard routing used MLP 66.1% and graph 33.9%, never attention. Oracle agreement was 30.7%, below the one-third chance reference for three balanced construction labels. This is evidence against meaningful route-to-construction alignment, not evidence for it. No single route exceeded the 98% collapse threshold, but one module was effectively unused.

## Interpretation and next action

The synthetic task/optimization setup does not currently force or reward the desired heterogeneous specialization. It would be misleading to call the observed proxy-cost reduction an adaptive win: the hard path lost accuracy and latency. The next iteration should first strengthen task separability and add a load-balancing/route-supervision ablation, while retaining a no-supervision baseline. Dynamic depth and external-domain claims remain NOT VERIFIED.
