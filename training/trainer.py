import os
import sys
sys.path.append("../")
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
import torch
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
from typing import List, Optional
import logging

from models.time_mqa import TimeMQAModel
from models.model_config import ModelConfig, DataConfig
from data_provider.data_factory import PaperDataFactory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeMQADataset(Dataset):
    def __init__(self, texts: List[str], tokenizer, max_length: int = 2048):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }


class TimeMQATrainer:   
    def __init__(
        self,
        model_name: str,
        tsqa_data_path: str,
        output_dir: str = "./output",
        model_config: Optional[ModelConfig] = None,
        data_config: Optional[DataConfig] = None
    ):
        self.model_name = model_name
        self.tsqa_data_path = tsqa_data_path
        self.output_dir = output_dir
        
        # Load configs
        from models.model_config import get_model_config, get_data_config
        self.model_config = model_config or get_model_config(model_name)
        self.data_config = data_config or get_data_config()
        
        # Initialize components
        self.model_wrapper = None
        self.train_dataset = None
        self.trainer = None
        
    def setup_model(self):
        logger.info(f"Setting up {self.model_name} model...")
        from models.time_mqa import create_time_mqa_model
        self.model_wrapper = create_time_mqa_model(self.model_name)
        logger.info("Model setup completed")
        
    def prepare_data(self):
        logger.info("Preparing training data...")
        
        # Create data factory and load paper dataset
        factory = PaperDataFactory(tsqa_base_path=self.tsqa_data_path)
        training_texts = factory.create_paper_reproduction_dataset(
            model_format=self.model_name,
            include_openorca=self.data_config.include_openorca
        )
        
        logger.info(f"Loaded {len(training_texts)} training samples")
        
        # Create training dataset
        tokenizer = self.model_wrapper.get_tokenizer()
        self.train_dataset = TimeMQADataset(
            training_texts, 
            tokenizer, 
            max_length=self.data_config.max_length
        )
        
    def setup_trainer(self):
        logger.info("Setting up trainer...")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            run_name=f"time-mqa-{self.model_name}",
            overwrite_output_dir=True,
            
            # Paper training config
            max_steps=self.model_config.max_steps,
            warmup_steps=self.model_config.warmup_steps,
            per_device_train_batch_size=self.model_config.batch_size_per_device,
            gradient_accumulation_steps=self.model_config.gradient_accumulation_steps,
            learning_rate=self.model_config.learning_rate,
            weight_decay=self.model_config.weight_decay,
            lr_scheduler_type=self.model_config.lr_scheduler_type,
            
            # Optimization
            optim=self.model_config.optimizer,
            fp16=self.model_config.fp16,
            gradient_checkpointing=self.model_config.gradient_checkpointing,
            
            # Logging and saving
            logging_steps=self.model_config.logging_steps,
            save_steps=self.model_config.save_steps,
            save_total_limit=self.model_config.save_total_limit,
            
            # No evaluation during continual pre-training
            eval_strategy="no",
            dataloader_num_workers=self.model_config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=["tensorboard"],
        )
        
        # Setup trainer
        model = self.model_wrapper.prepare_for_training()
        tokenizer = self.model_wrapper.get_tokenizer()
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        
        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.train_dataset,
            data_collator=data_collator,
            tokenizer=tokenizer,
        )
        
        logger.info("Trainer setup completed")
        
    def train(self):
        if self.trainer is None:
            raise ValueError("Trainer not setup. Call setup_trainer() first.")
        
        logger.info("Starting continual pre-training...")
        
        # Train and save
        self.trainer.train()
        self.model_wrapper.save_model(self.output_dir)
        
        logger.info(f"Model saved to {self.output_dir}")
        
    def run_full_training(self):
        logger.info("Starting Time-MQA training pipeline...")
        
        self.setup_model()
        self.prepare_data()
        self.setup_trainer()
        self.train()
        
        logger.info("Training pipeline completed!")


def train_time_mqa(
    model_name: str,
    tsqa_data_path: str,
    output_dir: str = "./output",
    include_openorca: bool = True
):  
    # Create trainer with data config
    from models.model_config import get_data_config
    data_config = get_data_config()
    data_config.include_openorca = include_openorca
    
    trainer = TimeMQATrainer(
        model_name=model_name,
        tsqa_data_path=tsqa_data_path,
        output_dir=output_dir,
        data_config=data_config
    )
    
    trainer.run_full_training()


def test_model_loading(model_name: str):
    logger.info(f"Testing {model_name} model loading...")
    
    try:
        from models.time_mqa import create_time_mqa_model
        model = create_time_mqa_model(model_name)
        
        tokenizer = model.get_tokenizer()
        logger.info(f"{model_name.upper()} model loaded successfully")
        logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
        
        # Test tokenization
        test_text = "<QUE> What is the trend in this time series? <ANS> The time series shows an upward trend. </END>"
        tokens = tokenizer(test_text, return_tensors="pt")
        logger.info(f"Test tokenization successful. Input length: {tokens['input_ids'].shape[1]}")
        
        del model
        torch.cuda.empty_cache()
        return True
        
    except Exception as e:
        logger.error(f"Failed to load {model_name}: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Train Time-MQA model")
    parser.add_argument("--model", type=str, default="mistral", 
                       choices=["llama", "mistral", "qwen"],
                       help="Model to train")
    parser.add_argument("--data_path", type=str, required=True,
                       help="Path to TSQA dataset")
    parser.add_argument("--output_dir", type=str, default="./finetune-output",
                       help="Output directory")
    parser.add_argument("--no_openorca", action="store_true",
                       help="Disable OpenOrca data")
    parser.add_argument("--test_model_loading", action="store_true",
                       help="Test model loading only")
    
    args = parser.parse_args()
    
    if args.test_model_loading:
        success = test_model_loading(args.model)
        logger.info(f"{'✅' if success else '❌'} {args.model.upper()} test {'passed' if success else 'failed'}")
    else:
        train_time_mqa(
            model_name=args.model,
            tsqa_data_path=args.data_path,
            output_dir=args.output_dir,
            include_openorca=not args.no_openorca
        )