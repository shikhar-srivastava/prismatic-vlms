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
nlu_datasets = ["wsc273", "winogrande", "lambada_standard", "arc_easy", "arc_challenge", "triviaqa", "webqs"]

# Initialize the result dictionary
result = {}

# Iterate through each model folder under the root directory
for model_name in os.listdir(root_dir):
    model_path = os.path.join(root_dir, model_name)
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
                    
                    # Capture accuracies and standard errors based on the dataset
                    if dataset == 'vqa-v2':
                        accuracy = summary.get('accuracy')
                        accuracy_stderr = summary.get('accuracy_stderr')
                        if accuracy is not None and accuracy_stderr is not None:
                            model_result['vqa-v2'] = (accuracy / 100.0, accuracy_stderr / 100.0)
                    elif dataset == 'text-vqa':
                        ocr_accuracy = summary.get('accuracy__TextVQA-OCR')
                        ocr_accuracy_stderr = summary.get('accuracy_stderr__TextVQA-OCR')
                        pure_accuracy = summary.get('accuracy__TextVQA-Pure')
                        pure_accuracy_stderr = summary.get('accuracy_stderr__TextVQA-Pure')
                        if ocr_accuracy is not None and ocr_accuracy_stderr is not None:
                            model_result['textvqa-ocr'] = (ocr_accuracy, ocr_accuracy_stderr)
                        if pure_accuracy is not None and pure_accuracy_stderr is not None:
                            model_result['textvqa-pure'] = (pure_accuracy, pure_accuracy_stderr)
                    elif dataset == 'gqa':
                        accuracy = summary.get('accuracy')
                        accuracy_stderr = summary.get('accuracy_stderr')
                        if accuracy is not None and accuracy_stderr is not None:
                            model_result['gqa'] = (accuracy / 100.0, accuracy_stderr / 100.0)
                    elif dataset == 'refcoco':
                        accuracy = summary.get('accuracy__RefCOCO')
                        accuracy_stderr = summary.get('accuracy_stderr__RefCOCO')
                        if accuracy is not None and accuracy_stderr is not None:
                            model_result['refcoco'] = (accuracy, accuracy_stderr)

        # Parse NLU/NLG results
        nlu_path = os.path.join(model_path, 'nlp/nlu', 'results.json')
        if os.path.isfile(nlu_path):
            with open(nlu_path, 'r') as f:
                nlu_results = json.load(f).get('results', {})
                for nlu_dataset in nlu_datasets:
                    if nlu_dataset in nlu_results:
                        acc = nlu_results[nlu_dataset].get('acc,none')
                        acc_stderr = nlu_results[nlu_dataset].get('acc_stderr,none')
                        if acc is not None and acc_stderr is not None:
                            model_result[nlu_dataset] = (acc, acc_stderr)

        # Parse triviaqa results
        triviaqa_path = os.path.join(model_path, 'nlp/triviaqa', 'results.json')
        if os.path.isfile(triviaqa_path):
            with open(triviaqa_path, 'r') as f:
                triviaqa_results = json.load(f).get('results', {})
                if 'triviaqa' in triviaqa_results:
                    exact_match = triviaqa_results['triviaqa'].get('exact_match,remove_whitespace')
                    exact_match_stderr = triviaqa_results['triviaqa'].get('exact_match_stderr,remove_whitespace')
                    if exact_match is not None and exact_match_stderr is not None:
                        model_result['triviaqa'] = (exact_match, exact_match_stderr)

        # Parse webqs results
        webqs_path = os.path.join(model_path, 'nlp/webqs', 'results.json')
        if os.path.isfile(webqs_path):
            with open(webqs_path, 'r') as f:
                webqs_results = json.load(f).get('results', {})
                if 'webqs' in webqs_results:
                    exact_match = webqs_results['webqs'].get('exact_match,none')
                    exact_match_stderr = webqs_results['webqs'].get('exact_match_stderr,none')
                    if exact_match is not None and exact_match_stderr is not None:
                        model_result['webqs'] = (exact_match, exact_match_stderr)

        # Add the model results to the final result dictionary
        if model_result:
            result[model_name] = model_result

# Save the final JSON to results_nlp.json
with open('results_std.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"Results saved to results_std.json")