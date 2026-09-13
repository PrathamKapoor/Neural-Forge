import torch
from torch import nn
from torch.nn import functional as F
from neuroforge.datasets import (
    Phase6ExpertSpecializationDataset,
    apply_phase6_marker_ablation,
    apply_phase6_random_marker,
    apply_phase6_structural_control,
    apply_phase6_token_permutation,
)


def test_phase6_dataset_shapes_and_neutral_marker():
    for family in ("feature", "relational", "contextual"):
        ds = Phase6ExpertSpecializationDataset(family, "train", samples=40, seed=123)
        assert len(ds) == 40
        assert ds.features.shape == (40, 12, 8)
        assert ds.targets.shape == (40,)

        # Check marker properties
        for i in range(40):
            ch4 = ds.features[i, :, 4]
            # Exactly one element is >= 0.9 (the marker = 1.0)
            marker_count = (ch4 >= 0.9).sum().item()
            assert marker_count == 1
            # Other elements should be near zero (noise < 0.2)
            non_markers = ch4[ch4 < 0.9]
            assert non_markers.abs().max().item() < 0.2


def test_phase6_marker_neutrality_diagnostics():
    """Marker must not leak label information or task-family information."""
    seed = 42
    # 1. Label neutrality
    for family in ("feature", "relational", "contextual"):
        train_ds = Phase6ExpertSpecializationDataset(family, "train", 400, seed)
        test_ds = Phase6ExpertSpecializationDataset(family, "test", 200, seed)
        clf = nn.Linear(12, 2)
        opt = torch.optim.Adam(clf.parameters(), lr=0.01)
        for _ in range(30):
            opt.zero_grad()
            F.cross_entropy(clf(train_ds.features[:, :, 4]), train_ds.targets).backward()
            opt.step()
        test_acc = float((clf(test_ds.features[:, :, 4]).argmax(1) == test_ds.targets).float().mean())
        assert 0.38 <= test_acc <= 0.62, f"{family} marker leaked label: acc={test_acc}"

    # 2. Task family neutrality
    train_feats, train_fams = [], []
    test_feats, test_fams = [], []
    for idx, family in enumerate(("feature", "relational", "contextual")):
        tr = Phase6ExpertSpecializationDataset(family, "train", 300, seed)
        te = Phase6ExpertSpecializationDataset(family, "test", 150, seed)
        train_feats.append(tr.features[:, :, 4])
        train_fams.append(torch.full((len(tr),), idx, dtype=torch.long))
        test_feats.append(te.features[:, :, 4])
        test_fams.append(torch.full((len(te),), idx, dtype=torch.long))
    X_tr, y_tr = torch.cat(train_feats, dim=0), torch.cat(train_fams, dim=0)
    X_te, y_te = torch.cat(test_feats, dim=0), torch.cat(test_fams, dim=0)

    clf_fam = nn.Sequential(nn.Linear(12, 24), nn.ReLU(), nn.Linear(24, 3))
    opt_fam = torch.optim.Adam(clf_fam.parameters(), lr=0.01)
    for _ in range(30):
        opt_fam.zero_grad()
        F.cross_entropy(clf_fam(X_tr), y_tr).backward()
        opt_fam.step()
    fam_acc = float((clf_fam(X_te).argmax(1) == y_te).float().mean())
    assert 0.22 <= fam_acc <= 0.45, f"Marker leaked task family: acc={fam_acc}"


def test_phase6_controls_and_permutations():
    ds = Phase6ExpertSpecializationDataset("contextual", "test", samples=20, seed=99)
    x = ds.features

    # Marker ablation replaces channel 4 with noise (no element >= 0.9)
    ablated = apply_phase6_marker_ablation(x)
    assert (ablated[:, :, 4] >= 0.9).sum().item() == 0

    # Random marker moves the marker to a different position
    rand_marker = apply_phase6_random_marker(x)
    for i in range(20):
        old_q = int(x[i, :, 4].argmax().item())
        new_q = int(rand_marker[i, :, 4].argmax().item())
        assert old_q != new_q
        assert rand_marker[i, new_q, 4] >= 0.9

    # Token permutation permutes tokens together with marker
    permuted = apply_phase6_token_permutation(x)
    assert permuted.shape == x.shape
    for i in range(20):
        # Query marker is still present at exactly one position
        assert (permuted[i, :, 4] >= 0.9).sum().item() == 1
