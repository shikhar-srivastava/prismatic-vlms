import os
import json

# Define the root directory and the datasets of interest
root_dir = 'evaluations'
datasets = ['vqa-v2', 'text-vqa', 'gqa']

# Initialize the result dictionary
result = {}

# Iterate through each model folder under the root directory
for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)
    if os.path.isdir(model_path):
        model_result = {}
        
        # Iterate through each dataset folder under the model folder
        for dataset in datasets:
            dataset_path = os.path.join(model_path, dataset)
            if os.path.isdir(dataset_path):
                # Determine the specific path to metrics.json based on the dataset
                if dataset == 'vqa-v2':
                    metrics_path = os.path.join(dataset_path, 'vqa-v2-slim', 'prism-clip+7b', 'metrics.json')
                elif dataset == 'text-vqa':
                    metrics_path = os.path.join(dataset_path, 'text-vqa-slim', 'prism-clip+7b', 'metrics.json')
                elif dataset == 'gqa':
                    metrics_path = os.path.join(dataset_path, 'gqa-slim', 'prism-clip+7b', 'metrics.json')
                
                # Parse the metrics.json file
                if os.path.isfile(metrics_path):
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                        summary = metrics.get('summary', {})
                        
                        # Capture accuracies based on the dataset
                        if dataset == 'vqa-v2':
                            accuracy = summary.get('accuracy')
                            if accuracy is not None:
                                model_result['vqa-v2'] = accuracy
                        elif dataset == 'text-vqa':
                            ocr_accuracy = summary.get('accuracy__TextVQA-OCR')
                            pure_accuracy = summary.get('accuracy__TextVQA-Pure')
                            if ocr_accuracy is not None:
                                model_result['textvqa-ocr'] = ocr_accuracy
                            if pure_accuracy is not None:
                                model_result['textvqa-pure'] = pure_accuracy
                        elif dataset == 'gqa':
                            accuracy = summary.get('accuracy')
                            if accuracy is not None:
                                model_result['gqa'] = accuracy
        
        # Add the model results to the final result dictionary
        if model_result:
            result[model_name] = model_result

# Save the final JSON to results.json
with open('results_B.json', 'w') as f:
    json.dump(result, f, indent=2)
