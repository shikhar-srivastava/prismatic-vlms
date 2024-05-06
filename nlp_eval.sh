# python nlp_evaluation.py --checkpoint_path /scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7 --write_path /scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results
# python nlp_evaluation.py --checkpoint_path reproduction-llava-v15+7b  --write_path /scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results

# STAGE 1

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs\
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/nlp/nlu"\
#     --batch_size 16

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks triviaqa \
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/nlp/triviaqa"\
#     --batch_size 8

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks mmlu \
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/nlp/mmlu"\
#     --batch_size 4

# # STAGE 2

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs\
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/nlp/nlu"\
#     --batch_size 16

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks triviaqa \
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/nlp/triviaqa"\
#     --batch_size 8

# accelerate launch -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks mmlu \
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/nlp/mmlu"\
#     --batch_size 4



#wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs \
#wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs\

# # VILA

# accelerate launch --main_process_port 29505 -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/vila/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs\
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/vila/nlp/nlu"\
#     --batch_size 16

# accelerate launch --main_process_port 29505 -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/vila/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks triviaqa \
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/vila/nlp/triviaqa"\
#     --batch_size 8

# accelerate launch --main_process_port 29505 -m lm_eval --model hf \
#     --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/vila/checkpoint_llm_only,trust_remote_code=True" \
#     --tasks mmlu \
#     --log_samples \
#     --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/vila/nlp/mmlu"\
#     --batch_size 4

# VILA STAGE 1 (meta-llama/Llama-2-7b-hf)

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