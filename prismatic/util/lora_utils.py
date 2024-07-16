import torch

def capture_initial_weights(model, target_modules):
    """
    Capture the initial weights of the LoRA layers in the model.

    Args:
        model (torch.nn.Module): The model containing LoRA layers.
        target_modules (list): List of module names to target for LoRA.

    Returns:
        dict: A dictionary containing the initial weights of the LoRA layers.
    """
    initial_weights = {}
    for name, param in model.named_parameters():
        if any(target in name for target in target_modules):
            initial_weights[name] = param.clone().detach()
    return initial_weights

def measure_lora_weight_change(model, initial_weights, target_modules):
    """
    Measure the average weight change across the LoRA layers in the model.

    Args:
        model (torch.nn.Module): The model containing LoRA layers.
        initial_weights (dict): A dictionary containing the initial weights of the LoRA layers.
        target_modules (list): List of module names to target for LoRA.

    Returns:
        float: The average weight change across the LoRA layers.
    """
    total_change = 0.0
    count = 0

    for name, param in model.named_parameters():
        if any(target in name for target in target_modules):
            initial_weight = initial_weights[name].to(param.device)
            weight_change = torch.abs(param - initial_weight).mean().item()
            del initial_weight
            total_change += weight_change
            count += 1

    if count == 0:
        raise ValueError("No LoRA layers found in the model.")
    
    average_change = total_change / count
    return average_change