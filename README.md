# Basilisk

Industrial grade LLM pretraining.  
Based on other libraries out there, in both the pytorch and jax ecosystems ([torchtitan](https://github.com/pytorch/torchtitan), [megatron-lm](https://github.com/NVIDIA/Megatron-LM), [olmo-core](https://github.com/allenai/Olmo-core), [metaseq](https://github.com/facebookresearch/metaseq), [deepspeed](https://github.com/deepspeedai/deepspeed), [marin](https://github.com/marin-community/marin), [flax](https://github.com/google/flax), [equinox](https://github.com/patrick-kidger/equinox), [galvatron](https://github.com/PKU-DAIR/Hetu-Galvatron)).


### Pending

- Configs
- Kernels
- FSDP/Zero-3, CP, PP
- MoE (router, dispatch/combine), EP
- Comms-compute overlap tricks (eg. bucketing in DDP)
- Checkpointing
- Activation checkpointing, gradient accumulation
- Mixed precision training
- Muon
- Long context (YaRN etc)
- Inference-specific optimizations (KV cache)


### Tests

```bash
uv run pytest tests/test_attention.py
```

