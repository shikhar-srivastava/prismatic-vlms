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

def measure_lora_weight_change_per_layer(model, initial_weights, target_modules):
    """
    Measure the average weight change across the LoRA layers in the model for each layer.

    Args:
        model (torch.nn.Module): The model containing LoRA layers.
        initial_weights (dict): A dictionary containing the initial weights of the LoRA layers.
        target_modules (list): List of module names to target for LoRA.

    Returns:
        dict: A dictionary where keys are layer numbers and values are the average weight change for that layer.
    """
    layer_changes = {}

    for name, param in model.named_parameters():
        if any(target in name for target in target_modules):
            # Extract the layer number based on the "layers.X" pattern
            try:
                # Split by '.' and find the index of 'layers', then get the next element as the layer number
                parts = name.split('.')
                layer_index = parts.index('layers') + 1
                layer_number = parts[layer_index]  # This should be the layer number (e.g., '9', '10')
            except (ValueError, IndexError):
                raise ValueError(f"Failed to extract layer number from parameter name: {name}")

            initial_weight = initial_weights[name].to(param.device)
            weight_change = torch.abs(param - initial_weight).mean().item()
            del initial_weight
            
            # Update the layer change dictionary
            if layer_number not in layer_changes:
                layer_changes[layer_number] = {
                    'total_change': 0.0,
                    'count': 0
                }
            
            layer_changes[layer_number]['total_change'] += weight_change
            layer_changes[layer_number]['count'] += 1

    if not layer_changes:
        raise ValueError("No LoRA layers found in the model.")
    
    # Calculate the average change for each layer
    average_layer_changes = {}
    for layer_number, data in layer_changes.items():
        average_layer_changes[layer_number] = data['total_change'] / data['count']
    
    return average_layer_changes