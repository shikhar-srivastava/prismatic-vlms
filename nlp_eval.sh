python nlp_evaluation.py --checkpoint_path /scratch/ssrivas9/prismatic-vlms/runs/reproduction-llava-v15+7b+stage-align+x7 --write_path /scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results
python nlp_evaluation.py --checkpoint_path reproduction-llava-v15+7b  --write_path /scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results

accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/checkpoint_llm_only,trust_remote_code=True" \
    --tasks lambada_openai \
    --log_samples \
    --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage1_results/"\
    --batch_size 16


accelerate launch -m lm_eval --model hf \
    --model_args "pretrained=/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/checkpoint_llm_only,trust_remote_code=True" \
    --tasks lambada_openai \
    --log_samples  \
    --output_path "/scratch/ssrivas9/prismatic-vlms/evaluations/stage2_results/"\
    --batch_size 16


#wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs \
#wsc273,arc_easy,arc_challenge,winogrande,lambada_standard,webqs\