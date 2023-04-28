import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from model.layer import GAULayer, Norm, RMSNorm


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.word_embedding = nn.Embedding(config.vocab_size, config.embedding_size)

        self.ln = RMSNorm(config.hidden_size, eps=config.eps, bias=config.use_bias)
        self.dropout = nn.Dropout(config.hidden_dropout)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=config.use_bias)
        # paperwithcode
        # self.word_embeddings.weight = self.lm_head.weight
        self.layers = nn.ModuleList([GAULayer(config) for _ in range(config.num_layers)])
        print("number of parameters: %.2fM" % (self.get_num_params() / 1e6,))
        sinusoidal_id = self.get_sinusoidal_id(
            config.block_size, config.attention_key_size
        )
        self.word_embedding.weight = self.lm_head.weight
        self.register_buffer("sin_pos", sinusoidal_id.sin(), persistent=False)
        self.register_buffer("cos_pos", sinusoidal_id.cos(), persistent=False)
        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))
        if config.deepnorm:
            # init deep norm
            init_scale = math.pow(8.0 * config.num_layers, 0.25)
            for name, p in self.named_parameters():
                if (
                        "fc1" in name
                        or "fc2" in name
                        or "out_proj" in name
                        or "v_proj" in name
                ):
                    p.data.div_(init_scale)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.word_embedding.weight.numel()
        return n_params

    def forward(self, idx, target=None):
        b, t = idx.size()
        assert t <= self.config.block_size
        token_embed = self.word_embedding(idx)
        x = self.dropout(token_embed)
        sinusoidal_pos = self.sin_pos[:, :t, :], self.cos_pos[:, :t, :]
        for i, g in enumerate(self.layers):
            x = g(x, sinusoidal_pos=sinusoidal_pos)
        x = self.ln(x)
        if target is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(b * t, -1), target.view(b * t), ignore_index=-1)
        else:
            logits = self.lm_head(x[:, [-1], :])
            logits = logits
            loss = None

        return logits, loss

    def get_sinusoidal_id(self, max_length, output_dim):
        position_ids = torch.arange(0, max_length, dtype=torch.float32)
        indices = torch.arange(0, output_dim // 2, dtype=torch.float32)
        indices = torch.pow(10000.0, -2 * indices / output_dim)
        sinusoidal_id = torch.einsum("n,d->nd", position_ids, indices)
        return sinusoidal_id[None, :, :]
