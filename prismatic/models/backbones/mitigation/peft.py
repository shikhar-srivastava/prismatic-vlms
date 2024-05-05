from peft import (
    LoraConfig,
    PrefixTuningConfig, #Prefix-Tuning
    PromptEncoderConfig, #P-Tuning
    PromptTuningConfig, # Prompt Tuning
    IA3Config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    set_peft_model_state_dict,
    prepare_model_for_kbit_training
)
from prismatic.overwatch import initialize_overwatch

import warnings
# Suppress HF Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def get_lora_target_modules(llm_model):
    return ["q_proj", "v_proj"]
def get_ia3_target_feedforward_modules(llm_model):
    target_modules, feedforward_modules = ["q_proj", "v_proj", "down_proj"], ["down_proj"]
    
    return target_modules, feedforward_modules

def apply_lora(llm_model, lora_r, lora_target_modules, lora_alpha, lora_dropout):
    llm_model = prepare_model_for_kbit_training(llm_model)
    loraconfig = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, target_modules=lora_target_modules,
        lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM"
    )
    llm_model = get_peft_model(llm_model, loraconfig)
    return llm_model

def apply_prefix(llm_model):
    peft_config = PrefixTuningConfig(
        task_type="CAUSAL_LM", 
        inference_mode=False, num_virtual_tokens=20)
    llm_model = get_peft_model(llm_model, peft_config)
    return llm_model

def apply_ptune(llm_model):
    peft_config = PromptEncoderConfig(
        task_type="CAUSAL_LM",
        num_virtual_tokens=20, 
        encoder_hidden_size=llm_model.config.hidden_size)
    llm_model = get_peft_model(llm_model, peft_config)
    return llm_model

def apply_prompt(llm_model):
    peft_config = PromptTuningConfig(
        task_type = "CAUSAL_LM",
        prompt_tuning_init="RANDOM",
        num_virtual_tokens=20,
    )
    llm_model = get_peft_model(llm_model, peft_config)
    return llm_model

def apply_olf(llm_model):
    print('Applying Output Layer Freezing')
    # implemented for the case of Llama2 llm
    main_model_attr = getattr(llm_model, 'llama2', None)
    if main_model_attr is not None and hasattr(main_model_attr, 'layers'):
        last_layer = main_model_attr.layers[-1]
        # Freeze all parameters in the last layer.
        for param in last_layer.parameters():
            param.requires_grad = False
        return llm_model
    else:
        raise ValueError("LLM architecture does not have the expected 'layers' attribute or main model attribute.")

def apply_ia3(llm_model):
    print('Applying IA3')
    target_modules, feedforward_modules = get_ia3_target_feedforward_modules(llm_model)
    peft_config = IA3Config(
            task_type="CAUSAL_LM", target_modules=target_modules, feedforward_modules=feedforward_modules)
    llm_model = get_peft_model(llm_model, peft_config)
    return llm_model

def freeze_model_weights(model, freeze = True):
    for name, param in model.named_parameters():
        param.requires_grad = False if freeze else True
    return model

def apply_mitigation(llm_model, mitigation_type):
    if mitigation_type is None:
        return llm_model
    else:
        overwatch.info(f"Applying mitigation: {mitigation_type}")
        
    if mitigation_type == 'lora':
        llm_model = apply_lora(llm_model, lora_r=4, lora_target_modules=get_lora_target_modules(llm_model), lora_alpha=16, lora_dropout=0.05)
    elif mitigation_type == 'prefix':
        llm_model = apply_prefix(llm_model)
    elif mitigation_type == 'ptune':
        llm_model = apply_ptune(llm_model)
    elif mitigation_type == 'prompt':
        llm_model = apply_prompt(llm_model)
    elif mitigation_type == 'olf':
        llm_model = apply_olf(llm_model)
    elif mitigation_type == 'ia3':
        llm_model = apply_ia3(llm_model)
    else:
        raise ValueError(f"Mitigation type {mitigation_type} not supported")
    return llm_model 