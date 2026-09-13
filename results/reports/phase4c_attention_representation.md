# Phase 4C — Representation probing and causal localization

## Method

For each of three independently trained depth-3 query-readout models, wrapper-level extraction captured R0 (encoder state) and R1/R2/R3 after each unchanged AttentionBlock. The block was frozen; probes were linear `24 -> 2` classifiers trained only on extracted tensors. Query/match/distractor indices were reconstructed from generator-visible query marker and analysis-only oracle match metadata; no role metadata entered the model.

## Frozen label/value probes

All target labels equal the signed associated value. Mean and query linear-probe test accuracy stayed near chance at every representation depth: R0 mean/query 47.9%/49.0%, R1 46.8%/47.6%, R2 47.4%/48.6%, R3 47.2%/48.6% (three-seed means). In contrast, the matching-token probe was 100% at R0 and stayed 100% through R3: the value is stored locally at its source token, not transferred.

Distractor probes were 54.7%, 54.2%, 52.9%, and 54.7% at R0–R3; this weak, variable elevation is not targeted retrieval. Random untrained-attention query probes were also near chance. Label-shuffled probe training generalized at chance, validating the probe path.

## Controls and localization

At R3, valid, broken-association, value-shuffled, and permuted query probes all remained near chance (approximately 49%, 52%, 48%, and 49%). Therefore controls cannot demonstrate a lost retrieval effect: there was no decodable query transfer to remove. Token permutation preserves query-probe results. The query/key match itself is already identifiable at R0 by exact normalized-key similarity; attention does not need to create that identity relation. The unresolved operation is value transfer, not key match detection.

## Verdict

Post-attention task information: **NOT SUPPORTED** at query/pooled locations. Query-key matching representation: **SUPPORTED only as input-level identity at R0**, not as learned attention evidence. Query-value transfer: **NOT SUPPORTED**. Readout bottleneck: **NOT SUPPORTED**; alternative readouts cannot expose information absent from the query representation. Depth-2 effect: **INCONCLUSIVE/likely variance**, as R2 probes show no transition. Existing AttentionBlock contextual capability and contextual specialization remain **NOT SUPPORTED**. Recommended next branch: separately investigate a redesigned value-transfer-capable attention primitive or task-compatible interface; do not start routing.
