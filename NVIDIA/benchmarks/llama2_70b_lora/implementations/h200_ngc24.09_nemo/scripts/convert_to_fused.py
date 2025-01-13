import torch
import torch.nn as nn
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention
from huggingface_hub import login
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, type=str, help="Path to the model location")
parser.add_argument("--hf_token", required=True, type=str, help="Huggingface token with llama access")
args = parser.parse_args()
login(args.hf_token)

model = LlamaForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    use_cache=False,
    torch_dtype=torch.bfloat16,
    use_flash_attention_2=True,
    max_position_embeddings=8192,
)

def replace_modules(module):
    has_bias = module.q_proj.bias is not None
    qkv_weight = torch.cat([module.q_proj.weight.data, module.k_proj.weight.data, module.v_proj.weight.data], dim=0)
    module.qkv_proj = nn.Linear(module.hidden_size, qkv_weight.shape[0], bias=has_bias)
    module.qkv_proj.weight.data = qkv_weight
    if has_bias:
        qkv_bias = torch.cat([module.q_proj.bias, module.k_proj.bias, module.v_proj.bias], dim=0)
        module.qkv_proj.bias.data = qkv_bias
    del module.q_proj
    del module.k_proj
    del module.v_proj
    module.dim1 = module.num_heads * module.head_dim
    module.dim2 = module.num_key_value_heads * module.head_dim

for name, module in model.named_modules():
    if isinstance(module, LlamaAttention):
        print("********\n"*5 + "REPLACING ATTENTION\n" + "********\n"*5)
        replace_modules(module)


model.config.save_pretrained(args.model_dir + "/my_config")
model.save_pretrained(args.model_dir + "/llama2-7b")
