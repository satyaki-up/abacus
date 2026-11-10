**An opinionated PyTorch-native framework for training modern Transformers.**

Based on other libraries out there, in both the PyTorch and Jax ecosystems ([torchtitan](https://github.com/pytorch/torchtitan), [megatron-lm](https://github.com/NVIDIA/Megatron-LM), [olmo-core](https://github.com/allenai/Olmo-core), [metaseq](https://github.com/facebookresearch/metaseq), [deepspeed](https://github.com/deepspeedai/deepspeed), [marin](https://github.com/marin-community/marin), [flax](https://github.com/google/flax), [equinox](https://github.com/patrick-kidger/equinox), [galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron)).


> [!NOTE]
> Under active development. Here be dragons.

## Quick start

There are sample scripts in the `examples` directory - see the comment at the top of the file. The `prep_data` script downloads a toy dataset, tokenizes it and stored the tokenized file for training.  
```bash
git clone https://github.com/satyaki-up/abacus
uv run python scripts/prep_data.py
uv run python examples/train_gpt_dp.py --limit 100
```


## Features

- Only modern Transformers are supported. For example, we don't support encoder-decoder style models like T5. Only SwiGLU activation function since that's the most common (no ReLU, GeLU).  
- Data Parallel is supported, both vanilla DDP as well as FSDP/ZeRO-3.  
- Tensor Parallel is supported, along with Megatron-style Sequence Parallel for Dropouts and Norms.  
- Pipeline Parallel is supported only via the bubble-heavy GPipe schedule. 1F1B and DualPipe will be added.  
- Context Parallel is supported in the Llama-3 all-gather style, not blockwise Ring Attention or Ulysses. The latter will be added.  
- Only intra-node Nvidia GPUs are supported, with 8-rank nodes or within the same NVLink domain. For example, Tensor Parallel can have max degree 8.
- Only AdamW is supported. Muon will be added eventually.  
- RoPE and QK-norm is supported.  
- Long context (YaRN etc) is not supported.  
- Only dense models are supported, not Mixture-of-Experts (hence, no Expert Parallel).  
- Minimal custom kernel support. FlashAttention and Liger kernels will be added.  
- Vanilla SDPA is supported. No support for alternating RoPE-NoPE or linear attention variants yet.  
- WSD LR schedule because most recent open models prefer that over cosine decay.  
- Sophisticated dataloaders like Zephon/Grain not supported yet.  
- Activation checkpointing and gradient accumulation not supported.  
- Distributed checkpointing not supported.  
- Mixed precision training not supported.  
- Computation and communication are mostly synchronous. Overlapping compute-comms (like via comms kernels or GPU initiated networking) not yet supported.  
- Diagnostic scripts for GPUs, NCCL, file system.  
- Huggingface tokenizer is used. Will add support for tiktoken and gigatoken.  

All the features above that're not yet supported, will be added.  

## Tests

```bash
uv run pytest tests/test_attention.py
uv run pytest tests -q
```

