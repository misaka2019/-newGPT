import numpy as np
from model.gpt import GPT
from config import *
import os
from torch.optim import AdamW
import math
import torch
import random


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[f'{split}/loss'] = losses.mean()
    model.train()
    return out



def get_lr(it):
    if it<=1000:
        return learning_rate
    if it<=10000:
        return learning_rate*0.7
    if it<=50000:
        return learning_rate*0.1


def train(model, optimizer, always_save_checkpoint, out_dir, gradient_accumulation_steps=1):
    best_val_loss = float("inf")
    model.train()
    accum_loss = 0.0
    for iter_num in range(max_iters):
        lr = get_lr(iter_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        x, y = get_batch('train')
        logits, loss = model(x, y)
        loss = loss / gradient_accumulation_steps
        accum_loss += loss.item()
        loss.backward()

        # accumulate gradients and step the optimizer only after every N steps
        if (iter_num + 1) % gradient_accumulation_steps == 0:
            # clip the gradient
            if grad_clip != 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        if (iter_num + 1) % log_interval == 0:
            avg_loss = accum_loss / log_interval
            accum_loss = 0.0
            wandb.log({'lr': lr,
                       "iter": iter_num,
                       'train/loss': avg_loss,
                       })

        if iter_num % eval_interval == 0:
            out = estimate_loss()
            val_loss = out['val/loss']
            wandb.log({'val/loss': val_loss})
            print(f"Iteration {iter_num}/{max_iters}, Val Loss: {val_loss:.4f}")

            if val_loss < best_val_loss or always_save_checkpoint:
                best_val_loss = val_loss
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': modelConfig,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    # 'config': config,
                }
                print(f"Saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))


def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + block_size]).astype(np.int64)) for i in ix])
    if device == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


if __name__ == "__main__":
    seed_everything(1234)
    # build vocab
    vocab = open(out_dir + "/vocab.txt", 'r').read().strip().split('\n')
    vocab_size = len(vocab)
    print('vocab_size:', vocab_size)

    data = np.memmap(data_path, dtype='uint16', mode='r+')
    train_data = data[:round(len(data) * 0.9)]
    val_data = data[round(len(data) * 0.9):]
    modelConfig.block_size = block_size
    modelConfig.vocab_size = vocab_size
    model = GPT(modelConfig)
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,
                      weight_decay=weight_decay,
                      betas=(0.9, 0.95),
                      )
    model.to(device)
    train(model, optimizer, always_save_checkpoint, out_dir)
