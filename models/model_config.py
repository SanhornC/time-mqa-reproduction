from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class ModelConfig:
    # Default Model selection
    model_name: str = "mistralai/Mistral-7B-v0.1" 
    
    # LoRA Configuration (from paper's Table 2)
    use_lora: bool = True
    lora_rank: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.0
    lora_target_modules: list = None
    
    # Training Configuration (from paper's hyperparameters)
    max_steps: int = 4000
    warmup_steps: int = 1000
    batch_size_per_device: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 5e-5
    embedding_learning_rate: float = 1e-5
    weight_decay: float = 0.1
    
    # Optimizer and Scheduler
    optimizer: str = "adamw_8bit"  # 8-bit AdamW from paper
    lr_scheduler_type: str = "cosine"
    
    # Model-specific settings
    max_seq_length: int = 2048
    
    # Logging and Saving
    logging_steps: int = 10
    save_steps: int = 500
    eval_steps: int = 500
    save_total_limit: int = 3
    
    # Hardware optimization
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    def __post_init__(self):
        """Set model-specific default configurations based on paper."""
        if self.lora_target_modules is None:
            # Paper Table 2: LoRA target modules
            self.lora_target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]

@dataclass 
class DataConfig:
    """Configuration for data processing."""
    
    # Dataset settings (from paper)
    samples_per_task: int = 1400           # Paper: 1,400 QA pairs per task
    total_tsqa_samples: int = 7000         # Paper: 1,400 × 5 tasks = 7,000
    include_openorca: bool = True
    openorca_samples: int = 3000           # Paper: 30% of 10k total
    total_training_samples: int = 10000   

    # Data splits
    test_size: float = 0.2
    val_size: float = 0.15
    
    # Data processing
    max_length: int = 2048
    truncation: bool = True
    padding: str = "max_length"

# Model-specific configurations matching the paper
LLAMA_CONFIG = ModelConfig(
    model_name="meta-llama/Meta-Llama-3-8B",  # or "meta-llama/Meta-Llama-3-8B"
)

MISTRAL_CONFIG = ModelConfig(
    model_name="mistralai/Mistral-7B-v0.1",
)

QWEN_CONFIG = ModelConfig(
    model_name="Qwen/Qwen2.5-7B",
)

# Configuration mapping
MODEL_CONFIGS = {
    "llama": LLAMA_CONFIG,
    "mistral": MISTRAL_CONFIG,
    "qwen": QWEN_CONFIG
}

def get_model_config(model_name: str) -> ModelConfig:
    model_name = model_name.lower()
    
    if model_name in MODEL_CONFIGS:
        return MODEL_CONFIGS[model_name]
    elif "llama" in model_name:
        return LLAMA_CONFIG
    elif "mistral" in model_name:
        return MISTRAL_CONFIG  
    elif "qwen" in model_name:
        return QWEN_CONFIG
    else:
        raise ValueError(f"Unknown model: {model_name}")

def get_data_config() -> DataConfig:
    return DataConfig()