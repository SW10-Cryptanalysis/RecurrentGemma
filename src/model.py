import torch
import logging
from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM
from easy_logging import EasyFormatter
from src.classes.config import Config

handler = logging.StreamHandler()
handler.setFormatter(EasyFormatter())
logger = logging.getLogger(__name__)
logger.addHandler(handler)


def get_model(config: Config) -> RecurrentGemmaForCausalLM:
    conf = RecurrentGemmaConfig(
        vocab_size=config.vocab_size,
        max_position_embeddings=config.max_context,
        hidden_size=config.dims,
        intermediate_size=config.dims * 3,
        num_hidden_layers=config.layers,
        num_attention_heads=config.att_heads,
        head_dim=config.head_dim,
        attention_window_size=config.attention_window_size,
        torch_dtype=torch.bfloat16,
        # Tokens
        pad_token_id=config.pad_token_id,
        bos_token_id=config.bos_token_id,
        eos_token_id=config.eos_token_id,
        # Activation & flash
        hidden_activation="gelu_fast",
        attn_implementation="flash_attention_2",
        rnn_hidden_size=config.dims * 4,
        use_cache=False,
    )

    model = RecurrentGemmaForCausalLM(conf)
    print("ReccurentGemma loaded!")
    print(f"Parameters:       {model.num_parameters():,}")
    print(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model
