import torch
import torch.nn.functional as F

def generate_consensus_label(labels_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Takes the noisy weak labels from GLAD-L, GLAD-S2, and RADD, and returns:
    1. A consensus target probability mask [0.0 - 1.0]
    2. A confidence mask (weight) to be used during loss calculation [0.0 - 1.0]
    
    Args:
        labels_dict: Dictionary containing 'gladl', 'glads2', 'radd' tensors.
                     Each tensor is assumed to be of shape (1, H, W) and binarized (or pseudo-prob).
    Returns:
        target: Float tensor of shape (1, H, W) representing consensus deforestation.
        confidence: Float tensor of shape (1, H, W) representing model's confidence in the label.
    """
    masks = []
    
    # Extract only valid masks
    for key in ["gladl", "glads2", "radd"]:
        if labels_dict.get(key) is not None:
            # Ensure it is a tensor of floats mapped to [0, 1]
            mask = labels_dict[key].clone().detach().float()
            # Normalize if max > 1 (e.g. 255)
            if mask.max() > 1.0:
                mask = mask / 255.0
            masks.append(mask)

    if len(masks) == 0:
        # Fallback if no labels found
        return None, None

    stacked_masks = torch.stack(masks, dim=0) # Shape: (Num_sources, 1, H, W)
    
    # 1. Consensus Target (Mean probability)
    # E.g., if 2 out of 3 say deforestation, prob = 0.66
    target = stacked_masks.mean(dim=0)
    
    # 2. Confidence Mask
    # Confidence is high if sources agree (all 0 or all 1), 
    # and low if they disagree (some 0, some 1).
    # We can measure agreement inversely to the variance of the predictions.
    # Variance of Bernoulli: p * (1-p). Max variance is at p=0.5.
    variance = target * (1 - target)
    
    # Normalize variance to [0, 1] (max variance for Bernoulli is 0.25)
    normalized_variance = variance / 0.25
    
    # Confidence is inverse of variance
    confidence = 1.0 - normalized_variance
    
    return target, confidence
