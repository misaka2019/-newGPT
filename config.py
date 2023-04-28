import torch
from contextlib import nullcontext
import wandb


class modelConfig:
    vocab_size = 12243
    block_size = 1024
    embedding_size = 768
    use_bias = False
    hidden_dropout_prob = 0.1
    num_layers = 24
    hidden_size = 768
    intermediate_size = 1536
    attention_key_size = 128
    activation = "swish"
    normalization = "softmax_plus"
    attention_scale = True
    attention_dropout = 0.1
    hidden_dropout = 0.1
    eps = 1e-12
    deepnorm = True


block_size = 1024
batch_size = 4
grad_clip = 0.5
learning_rate = 1e-4
max_iters = 600000  # total number of training iterations
weight_decay = 1e-2
grad_accum_steps = 16
eval_interval = 100
eval_iters = 200
log_interval = 10

device = 'cuda'
dtype = 'float32'

out_dir = 'model/gau_model'
data_path = 'data/train.bin'
always_save_checkpoint = False
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
config = {k: globals()[k] for k in config_keys}
wandb.init(project='GPT', name='GAU_bate', config=config)
