"""Generate reproducible Phase 3 figures from serialized source data."""
import json
from pathlib import Path
import matplotlib.pyplot as plt
data = json.loads(Path('results/metrics/phase3_expert_specialization/summary.json').read_text())
out = Path('figures/phase3_expert_specialization'); out.mkdir(parents=True, exist_ok=True)
names = ['mlp','graph','attention']; families = ['feature','relational','contextual']; matrix = data['matrix']
def save(name, title, ylabel='Accuracy'):
    plt.title(title); plt.ylabel(ylabel); plt.tight_layout(); plt.savefig(out / name, dpi=150); plt.close()
plt.imshow([matrix[n] for n in names], vmin=0, vmax=1, cmap='viridis'); plt.colorbar(label='Test accuracy'); plt.xticks(range(3),families); plt.yticks(range(3),names); save('performance_heatmap.png','Architecture × task performance')
ranks = [[sorted([matrix[n][i] for n in names], reverse=True).index(matrix[n][i])+1 for i in range(3)] for n in names]; plt.imshow(ranks, vmin=1,vmax=3,cmap='viridis_r'); plt.colorbar(label='Rank (1=best)'); plt.xticks(range(3),families); plt.yticks(range(3),names); save('rank_heatmap.png','Architecture × task rank','Rank')
for field, filename, title in [('parameters','parameter_counts.png','Parameter counts'),('estimated_forward_flops','flops.png','Estimated forward FLOPs'),('batch_latency_ms','latency.png','CPU batch latency (ms)')]:
    plt.bar(names,[data['cells'][f'{n}__feature']['per_seed'][0][field] for n in names]); save(filename,title,field)
for axis, filename in [('estimated_forward_flops','performance_vs_flops.png'),('batch_latency_ms','performance_vs_latency.png')]:
    for n in names: plt.scatter([data['cells'][f'{n}__feature']['per_seed'][0][axis]]*3,matrix[n],label=n)
    plt.legend(); plt.xlabel(axis); save(filename,'Performance efficiency frontier')
plt.bar(families,[data['specialization']['margins'][f] for f in families]); save('specialization_margins.png','Specialization margins')
for family, filename in [('relational','relational_control.png'),('contextual','contextual_control.png')]:
    control = data['controls'][('relational_node_feature_permutation' if family=='relational' else 'contextual_query_key_negation')]; x=range(3); plt.bar([i-.2 for i in x],[control[f'{n}__{family}']['original_accuracy'] for n in names],.4,label='original'); plt.bar([i+.2 for i in x],[control[f'{n}__{family}']['controlled_accuracy'] for n in names],.4,label='control'); plt.xticks(x,names); plt.legend(); save(filename,f'{family} structure control')
for n in names:
    values=[row['test_accuracy'] for f in families for row in data['cells'][f'{n}__{f}']['per_seed']]; plt.plot(range(len(values)),values,'o-',label=n)
plt.legend(); save('per_seed_variation.png','Per-seed test accuracy')
