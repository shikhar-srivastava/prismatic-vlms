#!/usr/bin/env python3
"""Debug script to understand DenseNet layer mapping and activation accumulation."""

import torch
import torch.nn as nn
from collections import OrderedDict

# Simplified DenseNet components for debugging
class _DenseLayer(nn.Module):
    def __init__(self, num_input_features: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, 
                               kernel_size=1, stride=1, bias=False)
        
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, 
                               kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        concat_features = torch.cat(x, 1) if isinstance(x, list) else x
        out = self.conv1(self.relu1(self.norm1(concat_features)))
        out = self.conv2(self.relu2(self.norm2(out)))
        return out

class _DenseBlock(nn.Module):
    def __init__(self, num_layers: int, num_input_features: int, growth_rate: int, bn_size: int = 4):
        super().__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size
            )
            self.layers.append(layer)

    def forward(self, init_features: torch.Tensor) -> torch.Tensor:
        features = [init_features]
        
        for layer in self.layers:
            new_features = layer(features)
            features.append(new_features)
            
        return torch.cat(features, 1)

class _Transition(nn.Module):
    def __init__(self, num_input_features: int, num_output_features: int):
        super().__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, 
                              kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(self.relu(self.norm(x)))
        out = self.pool(out)
        return out

class SimpleDenseNet(nn.Module):
    def __init__(self, block_config: tuple, growth_rate: int = 64, 
                 num_init_features: int = 64, compression: float = 0.8):
        super().__init__()
        
        # Stem
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Dense blocks and transitions
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # Add dense block
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                growth_rate=growth_rate
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features = num_features + num_layers * growth_rate

            # Add transition layer (except after last block)
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=int(num_features * compression)
                )
                self.features.add_module(f'transition{i+1}', trans)
                num_features = int(num_features * compression)

    def forward(self, x):
        return self.features(x)

def get_blocks_for_analysis(model):
    """Extract blocks same way as main script."""
    blocks = []
    for name, module in model.features.named_children():
        if name.startswith('denseblock'):
            # Add the entire dense block
            blocks.append(module)
            # Also add individual dense layers for detailed analysis
            for layer in module.layers:
                blocks.append(layer)
        elif name.startswith('transition'):
            # Add transition layers
            blocks.append(module)
    return blocks

def analyze_layer_mapping():
    """Analyze layer mapping and feature accumulation."""
    print("=== DenseNet Layer Mapping Analysis ===\n")
    
    # Create models
    models = {
        'densenet14': SimpleDenseNet(block_config=(3, 4, 4, 3), growth_rate=64, compression=0.8),
        'densenet18': SimpleDenseNet(block_config=(4, 4, 4, 6), growth_rate=64, compression=0.8),
        'densenet34': SimpleDenseNet(block_config=(6, 8, 12, 8), growth_rate=64, compression=0.75),
    }
    
    for model_name, model in models.items():
        print(f"\n--- {model_name.upper()} ---")
        blocks = get_blocks_for_analysis(model)
        
        # Calculate expected feature counts
        num_features = 64  # initial features after stem
        layer_idx = 0
        
        print(f"Block {layer_idx:2d}: Stem (64 features)")
        
        block_config = {
            'densenet14': (3, 4, 4, 3),
            'densenet18': (4, 4, 4, 6), 
            'densenet34': (6, 8, 12, 8)
        }[model_name]
        
        compression = 0.8 if model_name != 'densenet34' else 0.75
        growth_rate = 64
        
        for dense_block_idx, num_layers in enumerate(block_config):
            layer_idx += 1
            
            # Features grow within dense block
            dense_block_output_features = num_features + num_layers * growth_rate
            print(f"Block {layer_idx:2d}: DenseBlock{dense_block_idx+1} ({num_features} -> {dense_block_output_features} features)")
            
            # Individual dense layers
            for i in range(num_layers):
                layer_idx += 1
                layer_input_features = num_features + i * growth_rate
                print(f"Block {layer_idx:2d}: DenseLayer{dense_block_idx+1}.{i+1} ({layer_input_features} input features, +{growth_rate} output)")
            
            # Transition layer (if not last block)
            if dense_block_idx < len(block_config) - 1:
                layer_idx += 1
                transition_output_features = int(dense_block_output_features * compression)
                print(f"Block {layer_idx:2d}: Transition{dense_block_idx+1} ({dense_block_output_features} -> {transition_output_features} features)")
                num_features = transition_output_features
            else:
                num_features = dense_block_output_features
        
        # Identify problematic layers
        problem_layers = {
            'densenet14': 4,
            'densenet18': 5,
            'densenet34': 7
        }
        
        print(f"\n*** Layer {problem_layers[model_name]} (problematic layer) analysis:")
        
        # Trace what layer {problem_layers[model_name]} actually is
        block_idx = 0
        num_features = 64
        
        for dense_block_idx, num_layers in enumerate(block_config):
            block_idx += 1  # Dense block itself
            if block_idx == problem_layers[model_name]:
                print(f"Layer {problem_layers[model_name]} = DenseBlock{dense_block_idx+1} OUTPUT")
                print(f"  -> Concatenates ALL features from block: {num_features + num_layers * growth_rate} total channels")
                print(f"  -> Growth from {num_features} to {num_features + num_layers * growth_rate} channels")
                break
                
            for i in range(num_layers):
                block_idx += 1  # Individual dense layer
                if block_idx == problem_layers[model_name]:
                    print(f"Layer {problem_layers[model_name]} = DenseLayer{dense_block_idx+1}.{i+1} OUTPUT")
                    print(f"  -> Adds {growth_rate} new channels to existing {num_features + i * growth_rate}")
                    break
            else:
                continue
            break
            
            if dense_block_idx < len(block_config) - 1:
                block_idx += 1  # Transition layer
                if block_idx == problem_layers[model_name]:
                    dense_block_output = num_features + num_layers * growth_rate
                    print(f"Layer {problem_layers[model_name]} = Transition{dense_block_idx+1} OUTPUT")
                    print(f"  -> Compresses {dense_block_output} features by {compression}")
                    break
                num_features = int((num_features + num_layers * growth_rate) * compression)

if __name__ == "__main__":
    analyze_layer_mapping() 