import torch

import torch
import torch.nn.functional as F

def get_flattened_weights(module):
    """Flatten all the parameters of a module into a single 1D tensor."""
    return torch.cat([param.data.view(-1) for param in module.parameters()])

def calculate_similarity(model):
    """Calculate the similarity between slow_encoder and encoder using cosine similarity and L2 norm.
    
    Args:
        model (torch.nn.Module): The model containing slow_encoder and encoder.
        
    Returns:
        dict: A dictionary containing cosine similarity and L2 norm.
    """
    # Get flattened weights for both encoders
    slow_encoder_weights = get_flattened_weights(model.slow_encoder)
    encoder_weights = get_flattened_weights(model.encoder)

    # Calculate cosine similarity
    cosine_similarity = F.cosine_similarity(slow_encoder_weights.unsqueeze(0), encoder_weights.unsqueeze(0)).item()
    
    # Calculate L2 norm
    l2_norm = torch.norm(slow_encoder_weights - encoder_weights).item()
    
    return {
        'cosine_similarity': cosine_similarity,
        'l2_norm_similarity': l2_norm
    }

# Example usage with a model
# model = YourModel()  # Assuming you have your model defined elsewhere
# similarity = calculate_similarity(model)
# print(f"Cosine Similarity: {similarity['cosine_similarity']}")
# print(f"L2 Norm: {similarity['l2_norm']}")
