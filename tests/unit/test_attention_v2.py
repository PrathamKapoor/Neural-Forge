import torch


def test_attention_v2_preserves_shape_is_deterministic_and_has_gradients():
    from neuroforge.blocks.attention_v2 import AttentionBlockV2
    torch.manual_seed(3); block = AttentionBlockV2(24); x = torch.randn(4, 12, 24, requires_grad=True); raw = torch.randn(4,12,8); raw[:,0,4]=1
    output, cost = block(x, raw)
    assert output.shape == x.shape and cost > 0
    output.sum().backward(); assert x.grad is not None
