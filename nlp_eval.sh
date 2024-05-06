accelerate launch --main_process_port 29505 -m lm_eval --model hf \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True" \
    --tasks wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs\
    --log_samples \
    --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/vila_base_llm/nlp/nlu"\
    --batch_size auto

accelerate launch --main_process_port 29505 -m lm_eval --model hf \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True" \
    --tasks triviaqa \
    --log_samples \
    --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/vila_base_llm/nlp/triviaqa"\
    --batch_size 8

accelerate launch --main_process_port 29505 -m lm_eval --model hf \
    --model_args "pretrained=meta-llama/Llama-2-7b-hf,trust_remote_code=True" \
    --tasks mmlu \
    --log_samples \
    --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/vila_base_llm/nlp/mmlu"\
    --batch_size 4