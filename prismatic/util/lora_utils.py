import torch
from prismatic.overwatch import initialize_overwatch
# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def capture_initial_weights(model, target_modules = None):
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
        if target_modules is None or any(target in name for target in target_modules):
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

import torch

import torch

def log_weight_change_detailed(model, initial_weights):
    """
    Measure and log the weight change across various levels of granularity in the model, including:
    0. Aggregate total average weight change across the network (including LayerNorm).
    1. Aggregate per layer average weight change (including LayerNorm).
    2. Aggregate per layer average weight change for attention and MLP blocks separately (excluding LayerNorm).
    3. Aggregate per named parameter weight change, excluding LayerNorm parameters.

    Args:
        model (torch.nn.Module): The full finetuned model (GPTNeoX architecture).
        initial_weights (dict): A dictionary containing the initial weights of the model's layers.

    Returns:
        dict: A dictionary containing detailed weight change logs:
              - 'total': Average weight change across the entire network (including LayerNorm).
              - 'layers': Average weight change per layer (including breakdowns for attention and MLP blocks).
              - 'parameters': Average weight change per named parameter (excluding LayerNorms).
    """
    # Initialize accumulators for total weight change
    total_weight_change = 0.0
    total_param_count = 0
    
    # Dictionary to store layer, block, and parameter-level changes
    block_changes = {}
    param_changes = {}

    for name, param in model.named_parameters():
        # Ensure the parameter exists in initial_weights and has the same shape
        if name not in initial_weights:
            overwatch.error(f"Skipping {name}: not found in initial weights.")
            continue
        if param.shape != initial_weights[name].shape:
            overwatch.error(f"Skipping {name}: shape mismatch between initial and current parameters.")
            continue

        # Measure the weight change for the parameter
        initial_weight = initial_weights[name].to(param.device)
        
        # Check for NaN values in the current or initial weights
        if torch.isnan(param).any() or torch.isnan(initial_weight).any():
            overwatch.error(f"Skipping {name} due to NaN values in the weights.")
            continue

        # Compute weight change, using higher precision
        weight_change = torch.abs(param - initial_weight).mean()

        # # Ensure weight_change is finite
        # if not torch.isfinite(weight_change):
        #     overwatch.info(f"Skipping parameter {name} due to non-finite weight change.")
        #     overwatch.info(f"Weight change: {weight_change}")
        #     overwatch.info(f"Initial weight: {initial_weight}")
        #     overwatch.info(f"Current weight: {param}")
        #     continue

        # Convert to float for logging if necessary
        weight_change = weight_change.item()

        # Clamp weight change to avoid extreme values (for debugging)
        weight_change = max(min(weight_change, 1e6), -1e6)

        # Update the total network-wide weight change (including LayerNorm)
        total_weight_change += weight_change
        total_param_count += 1

        # If it's not a LayerNorm, store the weight change for this specific parameter
        if 'layernorm' not in name.lower():
            param_changes[name] = weight_change

        # Extract the layer number based on the "layers.X" pattern
        try:
            parts = name.split('.')
            if 'layers' in parts:
                layer_index = parts.index('layers') + 1
                layer_number = parts[layer_index]

                # Check if the parameter belongs to the attention or MLP block
                if 'attention' in name:
                    block_type = 'attention'
                elif 'mlp' in name:
                    block_type = 'mlp'
                else:
                    block_type = 'other'

                # Initialize the layer in the dictionary if not present
                if layer_number not in block_changes:
                    block_changes[layer_number] = {
                        'attention': {
                            'total_change': 0.0,
                            'count': 0
                        },
                        'mlp': {
                            'total_change': 0.0,
                            'count': 0
                        },
                        'layer': {
                            'total_change': 0.0,
                            'count': 0
                        }
                    }

                # Update the block (attention, mlp) and layer-level aggregates
                if block_type == 'attention':
                    block_changes[layer_number]['attention']['total_change'] += weight_change
                    block_changes[layer_number]['attention']['count'] += 1
                elif block_type == 'mlp':
                    block_changes[layer_number]['mlp']['total_change'] += weight_change
                    block_changes[layer_number]['mlp']['count'] += 1

                # Update the aggregate for the entire layer (including LayerNorm)
                block_changes[layer_number]['layer']['total_change'] += weight_change
                block_changes[layer_number]['layer']['count'] += 1

        except (ValueError, IndexError):
            overwatch.error(f"Failed to extract layer number from parameter name: {name}")
            continue

    if total_param_count == 0:
        raise ValueError("No parameters found to compute weight changes.")

    # Calculate the total average change across the network (including LayerNorm)
    total_average_change = total_weight_change / total_param_count if total_param_count > 0 else 0.0

    # Calculate the average change for each block and layer
    average_block_changes = {}
    for layer_number, block_data in block_changes.items():
        average_block_changes[layer_number] = {
            'attention': block_data['attention']['total_change'] / block_data['attention']['count']
            if block_data['attention']['count'] > 0 else 0.0,
            'mlp': block_data['mlp']['total_change'] / block_data['mlp']['count']
            if block_data['mlp']['count'] > 0 else 0.0,
            'layer': block_data['layer']['total_change'] / block_data['layer']['count']
            if block_data['layer']['count'] > 0 else 0.0  # Including LayerNorm
        }

    # Return the detailed logging information
    return {
        'total': total_average_change,              # 0. Aggregate total average weight change over the network
        'layers': average_block_changes,            # 1. Per layer average weight change and breakdowns (attention, mlp)
        'parameters': param_changes                 # 3. Per named parameter weight change (excluding LayerNorms)
    }
