import torch
from transformers import RecurrentGemmaConfig, RecurrentGemmaForCausalLM
from config import Config


def get_model():
    conf = RecurrentGemmaConfig(
        vocab_size=Config.vocab_size,
        max_position_embeddings=Config.max_context,
        hidden_size=Config.dims,
        intermediate_size=Config.dims * 4,
        num_hidden_layers=Config.layers,
        num_attention_heads=Config.att_heads,
        head_dim=Config.head_dim,
        attention_window_size=Config.attention_window_size,
        hidden_activation="gelu_fast",
    )

    model = RecurrentGemmaForCausalLM(conf)
    # model = torch.compile(model, mode="reduce-overhead")
    print("ReccurentGemma loaded!")
    print(f"Parameters:       {model.num_parameters():,}")
    print(f"VRAM for Weights: {(model.get_memory_footprint() / 1e9):.4f} GB")

    return model
