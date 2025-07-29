import sys
sys.path.append("../")

import torch
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    BitsAndBytesConfig
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    prepare_model_for_kbit_training
)
from typing import Optional, Dict, Any
import logging


from models.model_config import ModelConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TimeMQAModel:
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.peft_model = None
        
    def load_model_and_tokenizer(self):
        logger.info(f"Loading model: {self.config.model_name}")
        
        # Configure 8-bit quantization for memory efficiency
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_8bit_use_double_quant=True,
            bnb_8bit_quant_type="nf8",
            bnb_8bit_compute_dtype=torch.bfloat16
        )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add pad token if missing (common for Llama/Mistral)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        # Load model with quantization
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            attn_implementation="eager"  # Standard attention
        )
        
        # Prepare model for k-bit training
        self.model = prepare_model_for_kbit_training(self.model)
        
        logger.info("Model and tokenizer loaded successfully")
        
    def setup_lora(self):
        if not self.config.use_lora:
            logger.info("LoRA disabled, using full fine-tuning")
            return
            
        logger.info("Setting up LoRA configuration...")
        
        # LoRA configuration from paper's Table 2
        lora_config = LoraConfig(
            r=self.config.lora_rank,                    # 16
            lora_alpha=self.config.lora_alpha,          # 16  
            target_modules=self.config.lora_target_modules,
            lora_dropout=self.config.lora_dropout,      # 0.0
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        
        # Apply LoRA to the model
        self.peft_model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        self.peft_model.print_trainable_parameters()
        
        logger.info("LoRA setup completed")
        
    def get_model(self):
        return self.peft_model if self.peft_model is not None else self.model
    
    def get_tokenizer(self):
        return self.tokenizer
    
    def prepare_for_training(self):
        model = self.get_model()
        
        # Enable gradient checkpointing for memory efficiency
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Set model to training mode
        model.train()
        
        return model
    
    def save_model(self, output_dir: str):
        logger.info(f"Saving model to {output_dir}")
        
        if self.peft_model is not None:
            # Save LoRA adapters
            self.peft_model.save_pretrained(output_dir)
        else:
            # Save full model
            self.model.save_pretrained(output_dir)
        
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        
        logger.info("Model saved successfully")
    
    @classmethod
    def from_pretrained(cls, model_path: str, config: ModelConfig):
        instance = cls(config)
        instance.load_model_and_tokenizer()
        
        # Load LoRA weights if they exist
        try:
            from peft import PeftModel
            instance.peft_model = PeftModel.from_pretrained(
                instance.model, 
                model_path
            )
            logger.info(f"Loaded LoRA weights from {model_path}")
        except Exception as e:
            logger.warning(f"Could not load LoRA weights: {e}")
            logger.info("Using base model weights")
        
        return instance

def create_time_mqa_model(model_name: str) -> TimeMQAModel:
    from models.model_config import get_model_config

    # breakpoint()
    config = get_model_config(model_name)
    # breakpoint()
    model = TimeMQAModel(config)
    
    # Load model and setup LoRA
    model.load_model_and_tokenizer()
    model.setup_lora()
    
    return model

# Example usage functions
def load_mistral_7b() -> TimeMQAModel:
    return create_time_mqa_model("mistral")

def load_llama_8b() -> TimeMQAModel:
    return create_time_mqa_model("llama")

def load_qwen_7b() -> TimeMQAModel:
    return create_time_mqa_model("qwen")

def test_model_loading(model_name: str = "mistral"):
    """Test model loading with proper error handling."""
    try:
        logger.info(f"Testing {model_name} model loading...")
        
        # Test model creation
        model = create_time_mqa_model(model_name)
        
        # Test basic properties
        tokenizer = model.get_tokenizer()
        model_obj = model.get_model()
        
        logger.info(f"✅ {model_name.upper()} model loaded successfully")
        logger.info(f"Model config: {model.config.model_name}")
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        logger.info(f"LoRA enabled: {model.config.use_lora}")
        logger.info(f"LoRA rank: {model.config.lora_rank}")
        logger.info(f"Target modules: {model.config.lora_target_modules}")
        
        # Test tokenization
        test_text = "<QUE> What is the trend in this time series? <ANS> The time series shows an upward trend. </END>"
        tokens = tokenizer(test_text, return_tensors="pt")
        logger.info(f"Test tokenization successful. Input length: {tokens['input_ids'].shape[1]}")
        
        # Test model forward pass (without gradient computation)
        model_obj.eval()
        with torch.no_grad():
            outputs = model_obj(**tokens)
            logger.info(f"Test forward pass successful. Logits shape: {outputs.logits.shape}")
        
        return model
        
    except Exception as e:
        logger.error(f"❌ Failed to load {model_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Test all models
    models_to_test = ["mistral", "llama", "qwen"]
    
    for model_name in models_to_test:
        print(f"\n{'='*50}")
        print(f"Testing {model_name.upper()} Model")
        print(f"{'='*50}")
        
        model = test_model_loading(model_name)
        
        if model is not None:
            print(f"✅ {model_name.upper()} test passed")
        else:
            print(f"❌ {model_name.upper()} test failed")
        
        # Clean up GPU memory
        if model is not None:
            del model
            torch.cuda.empty_cache()