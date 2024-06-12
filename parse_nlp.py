import os
import json

# Define the root directory and the datasets of interest
root_dir = 'evaluations/'
datasets = {
    'vqa-v2': 'vqa-v2-slim',
    'text-vqa': 'text-vqa-slim',
    'gqa': 'gqa-slim',
    'refcoco': 'refcoco-slim'
}
nlu_datasets = ["wsc273", "winogrande", "lambada_standard", "arc_easy", "arc_challenge"]

# Initialize the result dictionary
result = {}

# Iterate through each model folder under the root directory
for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)
    # if ('phi' not in model_path) and ('pythia' not in model_path):
    #     continue
    if os.path.isdir(model_path):
        print(f"Processing model: {model_name}")
        model_result = {}
        
        # Iterate through each dataset folder under the model folder
        for dataset, dataset_slim in datasets.items():
            dataset_path = os.path.join(model_path, dataset, dataset_slim, 'prism-clip+7b', 'metrics.json')
            if os.path.isfile(dataset_path):
                # Parse the metrics.json file
                with open(dataset_path, 'r') as f:
                    metrics = json.load(f)
                    summary = metrics.get('summary', {})
                    
                    # Capture accuracies based on the dataset
                    if dataset == 'vqa-v2':
                        accuracy = summary.get('accuracy')
                        if accuracy is not None:
                            model_result['vqa-v2'] = accuracy / 100.0
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
                            model_result['gqa'] = accuracy / 100.0
                    elif dataset == 'refcoco':
                        accuracy = summary.get('accuracy__RefCOCO')
                        if accuracy is not None:
                            model_result['refcoco'] = accuracy

        # Parse NLU/NLG results
        nlu_path = os.path.join(model_path, 'nlp/nlu', 'results.json')
        if os.path.isfile(nlu_path):
            with open(nlu_path, 'r') as f:
                nlu_results = json.load(f).get('results', {})
                for nlu_dataset in nlu_datasets:
                    if nlu_dataset in nlu_results:
                        acc = nlu_results[nlu_dataset].get('acc,none')
                        if acc is not None:
                            model_result[nlu_dataset] = acc

        # Parse MMLU results
        mmlu_path = os.path.join(model_path, 'nlp/mmlu', 'results.json')
        if os.path.isfile(mmlu_path):
            with open(mmlu_path, 'r') as f:
                mmlu_results = json.load(f).get('results', {})
                mmlu_acc = mmlu_results.get('mmlu', {}).get('acc,none')
                if mmlu_acc is not None:
                    model_result['mmlu'] = mmlu_acc

        # Add the model results to the final result dictionary
        if model_result:
            result[model_name] = model_result

# Save the final JSON to results_A.json
with open('results_nlp.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Results saved to results_nlp.json")