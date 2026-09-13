# Experiment protocol

## Question

Can a learned router allocate MLP, graph, and attention computation by sample more effectively than fixed models under comparable stated cost?

## Controlled task

Samples are random sequences. Three equally represented construction regimes define labels: pooled feature relation (MLP-biased), local adjacent feature products (graph-biased), and first/last-position comparison (attention-biased). The construction identity is held out of model features and retained only for oracle-route evaluation.

## Comparisons

The runner trains MLP-only, graph-only, attention-only, fixed-hybrid, and adaptive-hybrid models with matching dimensions and epoch budget. It reports accuracy, estimated route cost, utilization, entropy, oracle agreement, and CPU latency. The learned model has soft-training and truly conditional hard-inference modes.

## Interpretation guardrails

The synthetic oracle is a construction label, not proof that a primitive is uniquely necessary. Higher oracle agreement is supportive only if accuracy and cost are also competitive. A lower analytical estimate does not establish lower latency, especially at small batch sizes and on CPU.
