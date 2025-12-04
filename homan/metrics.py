
def batch_mask_iou(ref, pred, eps=0.000001):
    """
    Args:
        ref (torch.Tensor): [..., H, W]
        pred (torch.Tensor): [..., H, W]
    """
    ref = ref.float()
    pred = pred.float()
    if ref.max() > 1 or ref.min() < 0:
        raise ValueError(
            "Ref mask should have values in [0, 1], "
            f"not [{ref.min(), ref.max()}]"
        )
    if pred.max() > 1 or pred.min() < 0:
        raise ValueError(
            "Ref mask should have values in [0, 1], "
            f"not [{pred.min(), pred.max()}]"
        )

    inter = ref * pred
    union = (ref + pred).clamp(0, 1)
    # ious = inter.sum(1).sum(1).float() / (union.sum(1).sum(1).float() + eps)
    ious = inter.sum(dim=(-2, -1)).float() / (union.sum(dim=(-2, -1)).float() + eps)
    return ious
