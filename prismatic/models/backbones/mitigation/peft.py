from peft import (
    LoraConfig,
    PrefixTuningConfig, #Prefix-Tuning
    PromptEncoderConfig, #P-Tuning
    PromptTuningConfig, # Prompt Tuning
    IA3Config,
    get_peft_model,
    get_peft_model_state_dict,
    prepare_model_for_kbit_training,
    PeftModelForCausalLM)
from prismatic.overwatch import initialize_overwatch

import warnings
# Suppress HF Deprecation Warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

def get_all_linear_layers(llm_model):
    # Picked from DataBricks: https://www.databricks.com/blog/efficient-fine-tuning-lora-guide-llms
    import re
    model_modules = str(llm_model.modules)
    pattern = r'\((\w+)\): Linear'
    linear_layer_names = re.findall(pattern, model_modules)
    names = []
    # Print the names of the Linear layers
    for name in linear_layer_names:
        names.append(name)
    target_modules = list(set(names))
    return target_modules
    
def get_ia3_target_feedforward_modules(llm_model, run_id = None):
    if run_id is None:
        target_modules, feedforward_modules = ["q_proj", "v_proj", "down_proj"], ["down_proj"]
    elif 'pythia' in run_id:
        target_modules = ["query_key_value", "attention.dense"]
        feedforward_modules = ["attention.dense"]
    else:
        target_modules = ["q_proj", "v_proj", "down_proj"]
        feedforward_modules = ["down_proj"]
    
    return target_modules, feedforward_modules

def apply_lora(llm_model, lora_r, lora_target_modules, lora_alpha, lora_dropout):
    overwatch.info(f"Applying lora.",ctx_level=2)
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

# def apply_olf(llm_model):
#     print('Applying Output Layer Freezing')
#     # implemented for the case of Llama2 llm
#     main_model_attr = getattr(llm_model, 'llama2', None)
#     if main_model_attr is not None and hasattr(main_model_attr, 'layers'):
#         last_layer = main_model_attr.layers[-1]
#         # Freeze all parameters in the last layer.
#         for param in last_layer.parameters():
#             param.requires_grad = False
#         return llm_model
#     else:
#         raise ValueError("LLM architecture does not have the expected 'layers' attribute or main model attribute.")

def fetch_last_layer(llm_model):
    try:
        if isinstance(llm_model, PeftModelForCausalLM):
            # If the model is wrapped with PEFT (LoraModel)
            model_layers = llm_model.base_model.model.model.layers
        else:
            # If the model is a regular model
            if hasattr(llm_model, 'transformer'):
                model_layers = llm_model.transformer.layers
            elif hasattr(llm_model, 'model'):
                model_layers = llm_model.model.layers
            else:
                raise ValueError("No layers attribute found in the model.")
        
            model = llm_model
        
        if hasattr(model, 'layers'):
            model_layers = model.layers
        elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
            model_layers = model.encoder.layers
        elif hasattr(model, 'block'):  # Sometimes layers are stored under 'block'
            model_layers = model.block
        else:
            raise ValueError("No layers attribute found in the model.")
        last_layer = model_layers[-1]
        return last_layer

    except Exception as e:
        raise ValueError(f"An error occurred while trying to fetch the last layer: {str(e)}")

def apply_olf(llm_model):
    overwatch.info(f"Freezing Last Layer", ctx_level = 2)
    last_layer = fetch_last_layer(llm_model)
    for param in last_layer.parameters():
        param.requires_grad = False
    return llm_model

def apply_ia3(llm_model, run_id=None):
    except Exception as e:
        raise ValueError(f"An error occurred while trying to freeze the last layer: {str(e)}")



def apply_ia3(llm_model):
    print('Applying IA3')
    target_modules, feedforward_modules = get_ia3_target_feedforward_modules(llm_model, run_id)
    peft_config = IA3Config(
            task_type="CAUSAL_LM", target_modules=target_modules, feedforward_modules=feedforward_modules)
    llm_model = get_peft_model(llm_model, peft_config)
    return llm_model

def freeze_model_weights(model, freeze = True):
    for name, param in model.named_parameters():
        param.requires_grad = False if freeze else True
    return model

def apply_mitigation(llm_model, cfg):
    if isinstance(cfg, dict):
        mitigation_type = cfg.get("mitigation", None)
        olf = cfg.get("olf", False)
        run_id = cfg.get("run_id", None)
    elif hasattr(cfg, 'mitigation'):
        mitigation_type = getattr(cfg, 'mitigation', None)
        olf = getattr(cfg, 'olf', False)
        run_id = getattr(cfg, 'run_id', None)
    else:
        mitigation_type = None
        olf = False
    
    if mitigation_type is None:
        return llm_model
    else:
        overwatch.info(f"Applying mitigation: {mitigation_type}!")
        
    if 'lora' in mitigation_type:
        if isinstance(cfg, dict):
            lora_rank = cfg.get("lora_rank")
            lora_alpha = cfg.get("lora_alpha")
            lora_target_modules = cfg.get("lora_target_modules")
        else:
            lora_rank = getattr(cfg, 'lora_rank')
            lora_alpha = getattr(cfg, 'lora_alpha')
            lora_target_modules = getattr(cfg, 'lora_target_modules')

        overwatch.info(f"Applying LORA with rank {lora_rank} and alpha {lora_alpha}", ctx_level=1)
        llm_model = apply_lora(llm_model, lora_r=lora_rank, \
                               lora_target_modules=lora_target_modules, lora_alpha=lora_alpha, lora_dropout=0.05)
    elif mitigation_type == 'prefix':
        llm_model = apply_prefix(llm_model)
    elif mitigation_type == 'ptune':
        llm_model = apply_ptune(llm_model)
    elif mitigation_type == 'prompt':
        llm_model = apply_prompt(llm_model)
    elif mitigation_type == 'ia3':
        llm_model = apply_ia3(llm_model, run_id)
    else:
        raise ValueError(f"Mitigation type {mitigation_type} not supported")
    if olf == True:
        llm_model = apply_olf(llm_model)

    return llm_model 