import torch

def load_state_dict(path):
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    print(f"{checkpoint['model'].keys()}")
    state_dict = checkpoint['model']['llm_backbone']
    return state_dict

def compare_state_dicts(state_dict1, state_dict2):
    keys1 = set(state_dict1.keys())
    keys2 = set(state_dict2.keys())
    
    common_keys = keys1.intersection(keys2)
    unique_keys1 = keys1 - keys2
    unique_keys2 = keys2 - keys1
    
    equal_weights = True
    shape_mismatch = False
    total_weight_change = 0.0
    total_elements = 0
    
    for key in common_keys:
        if state_dict1[key].shape != state_dict2[key].shape:
            shape_mismatch = True
            print(f"Shape mismatch in layer {key}: {state_dict1[key].shape} vs {state_dict2[key].shape}")
        else:
            weight_difference = torch.abs(state_dict1[key] - state_dict2[key])
            total_weight_change += torch.sum(weight_difference).item()
            weight_change_this_layer = torch.sum(weight_difference).item() / state_dict1[key].numel()
            total_elements += state_dict1[key].numel()
            if not torch.equal(state_dict1[key], state_dict2[key]):
                print(f"Weight mismatch in layer {key}: Weight Change: {weight_change_this_layer}")
                equal_weights = False
    
    if unique_keys1:
        print(f"Unique keys in model 1: {unique_keys1}")
    if unique_keys2:
        print(f"Unique keys in model 2: {unique_keys2}")
    
    average_weight_change = total_weight_change / total_elements if total_elements > 0 else float('inf')
    
    return equal_weights, shape_mismatch, average_weight_change, unique_keys1, unique_keys2

def analyze_models(path1, path2):
    state_dict1 = load_state_dict(path1)
    state_dict2 = load_state_dict(path2)
    
    equal_weights, shape_mismatch, average_weight_change, unique_keys1, unique_keys2 = compare_state_dicts(state_dict1, state_dict2)
    
    if equal_weights and not shape_mismatch:
        print("The models have identical weights.")
    elif not equal_weights and not shape_mismatch:
        print("The models have different weights but similar shapes.")
    elif shape_mismatch:
        print("The models have some layers with different shapes.")
    else:
        print("Unexpected comparison result.")
    
    print(f"Average weight change across all layers: {average_weight_change}")
    print(f"Unique keys in model 1: {unique_keys1}")
    print(f"Unique keys in model 2: {unique_keys2}")

if __name__ == "__main__":
    model_path1 = "/scratch/ssrivas9/prismatic-vlms/runs/soft-stage-0-after-llava-vqav2/checkpoints/latest-checkpoint.pt"
    model_path2 = "/scratch/ssrivas9/prismatic-vlms/runs/soft-stage-0-after-llava-vqav2-soft-alpha-0.1/checkpoints/step-001294-epoch-00-loss=0.0196.pt"
    
    analyze_models(model_path1, model_path2)
