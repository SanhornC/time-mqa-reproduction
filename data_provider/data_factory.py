import sys
sys.path.append("../")

import pandas as pd
import logging
import random
from typing import List, Dict
from data_provider.data_loader import TSQA_Dataloader
from datasets import load_dataset

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PaperDataFactory:
    def __init__(self, tsqa_base_path: str):
        logging.info("Initializing PaperDataFactory...")
        self.tsqa_loader = TSQA_Dataloader(root_path=tsqa_base_path)
        self.tsqa_data = self.tsqa_loader.load_all_tasks()
        
        # Verify we have the expected tasks
        expected_tasks = ['forecasting', 'imputation', 'classification', 'anomaly_detection', 'open_ended_QA']
        missing_tasks = [task for task in expected_tasks if task not in self.tsqa_data]
        if missing_tasks:
            logging.warning(f"Missing tasks: {missing_tasks}")
    
    def _format_for_llama(self, row: pd.Series) -> str:
        question = row['question']
        answer = row['answer']
        return f"<|begin_of_text|> <QUE> {question} <ANS> {answer} </END> <|end_of_text|>"
    
    def _format_for_mistral(self, row: pd.Series) -> str:
        question = row['question']
        answer = row['answer']
        return f"<s> <QUE> {question} <ANS> {answer} </END> </s>"
    
    def _format_for_qwen(self, row: pd.Series) -> str:
        question = row['question']
        answer = row['answer']
        return f"<QUE> {question} <ANS> {answer} </END> <|endoftext|>"
    
    def _load_openorca_data(self, num_samples: int = 3000) -> pd.DataFrame:
        try:
            logging.info(f"Loading OpenOrca dataset...")
            # Load the OpenOrca dataset
            ds = load_dataset("Open-Orca/OpenOrca", split="train")
            
            logging.info(f"OpenOrca dataset loaded. Total samples: {len(ds)}")
            
            # Convert to pandas DataFrame for easier manipulation
            df = ds.to_pandas()
            
            # OpenOrca has different column names, need to map them
            # Check the actual column names in OpenOrca
            if 'question' in df.columns and 'response' in df.columns:
                df = df.rename(columns={'response': 'answer'})
            elif 'system_prompt' in df.columns and 'response' in df.columns:
                # If OpenOrca uses system_prompt + question format
                df['question'] = df['system_prompt'] + " " + df.get('question', '')
                df = df.rename(columns={'response': 'answer'})
            elif 'conversation' in df.columns:
                # Handle conversation format if present
                # This would need custom parsing based on OpenOrca's actual format
                logging.warning("OpenOrca conversation format detected. Custom parsing required.")
                return pd.DataFrame()
            else:
                logging.error(f"Unexpected OpenOrca format. Columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Sample the required number of instances
            if len(df) < num_samples:
                logging.warning(f"OpenOrca has only {len(df)} samples, less than requested {num_samples}")
                sampled_df = df.copy()
            else:
                sampled_df = df.sample(n=num_samples, random_state=42)
            
            # Keep only question and answer columns
            final_df = sampled_df[['question', 'answer']].copy()
            final_df['task_source'] = 'openorca'
            
            logging.info(f"Successfully loaded {len(final_df)} OpenOrca samples")
            return final_df
            
        except Exception as e:
            logging.error(f"Failed to load OpenOrca data: {e}")
            return pd.DataFrame()
    
    
    def sample_tsqa_data(self, samples_per_task: int = 1400) -> pd.DataFrame:
        logging.info(f"Sampling {samples_per_task} instances from each task...")
        
        sampled_dfs = []
        total_samples = 0
        
        for task_name, df in self.tsqa_data.items():
            if len(df) < samples_per_task:
                logging.warning(
                    f"Task '{task_name}' has only {len(df)} samples, "
                    f"less than required {samples_per_task}. Using all available samples."
                )
                sampled_df = df.copy()
            else:
                # Sample with fixed random state for reproducibility
                sampled_df = df.sample(n=samples_per_task, random_state=42)
            
            sampled_df['task_source'] = task_name
            sampled_dfs.append(sampled_df)
            total_samples += len(sampled_df)
            logging.info(f"Sampled {len(sampled_df)} samples from {task_name}")
        
        # Combine all sampled data
        combined_df = pd.concat(sampled_dfs, ignore_index=True)
        logging.info(f"Total TSQA samples: {len(combined_df)}")
        
        return combined_df
    
    def create_training_dataset(self, 
                              samples_per_task: int = 1400, 
                              model_format: str = 'llama',
                              include_openorca: bool = False,
                              openorca_samples: int = 3000) -> List[str]:
        
        if not self.tsqa_data:
            logging.error("TSQA data not loaded. Cannot create training dataset.")
            return []
        
        # Sample TSQA data (7,000 samples as per paper)
        tsqa_df = self.sample_tsqa_data(samples_per_task)
        
        # Add OpenOrca data if requested
        combined_df = tsqa_df.copy()
        
        if include_openorca:
            logging.info(f"Loading OpenOrca data...")
            openorca_df = self._load_openorca_data(openorca_samples)
            
            if not openorca_df.empty:
                combined_df = pd.concat([tsqa_df, openorca_df], ignore_index=True)
                logging.info(f"Combined dataset: {len(tsqa_df)} TSQA + {len(openorca_df)} OpenOrca = {len(combined_df)} total")
                
                # Verify the 70% TSQA / 30% OpenOrca ratio as per paper
                tsqa_ratio = len(tsqa_df) / len(combined_df)
                openorca_ratio = len(openorca_df) / len(combined_df)
                logging.info(f"Dataset composition: {tsqa_ratio:.1%} TSQA, {openorca_ratio:.1%} OpenOrca")
            else:
                logging.warning("Failed to load OpenOrca data. Using only TSQA data.")
        
        # Select formatter based on model
        if model_format.lower() == 'llama':
            formatter = self._format_for_llama
        elif model_format.lower() == 'mistral':
            formatter = self._format_for_mistral
        elif model_format.lower() == 'qwen':
            formatter = self._format_for_qwen
        else:
            raise ValueError(f"Unsupported model format: {model_format}")
        
        logging.info(f"Formatting data for {model_format} model...")
        formatted_data = combined_df.apply(formatter, axis=1).tolist()
        # breakpoint()
        # Shuffle for better training
        random.seed(42)
        random.shuffle(formatted_data)
        
        logging.info(f"Created training dataset with {len(formatted_data)} samples")
        return formatted_data
    
    def get_task_distribution(self) -> Dict[str, int]:
        return {task: len(df) for task, df in self.tsqa_data.items()}
    
    def create_paper_reproduction_dataset(self, model_format: str = 'llama', include_openorca: bool = True) -> List[str]:
        logging.info("Creating paper reproduction dataset...")
        logging.info("Configuration: 1,400 samples per task × 5 tasks = 7,000 TSQA samples")
        
        if include_openorca:
            logging.info("Including 3,000 OpenOrca samples (70% TSQA / 30% OpenOrca ratio)")
        else:
            logging.info("OpenOrca disabled - using only TSQA data")
        
        return self.create_training_dataset(
            samples_per_task=1400,
            model_format=model_format,
            include_openorca=include_openorca,
            openorca_samples=3000
        )

# if __name__ == '__main__':
#     # Test the factory
#     tsqa_root_path = "/home/sanhorn2/time-mqa/data/tsqa"  # Adjust path as needed
    
#     try:
#         # Initialize factory
#         factory = PaperDataFactory(tsqa_base_path=tsqa_root_path)
        
#         # Show task distribution
#         distribution = factory.get_task_distribution()
#         print("\n--- TSQA Task Distribution ---")
#         for task, count in distribution.items():
#             print(f"{task}: {count} samples")
        
#         # Test without OpenOrca first
#         print("\n--- Testing TSQA-only dataset ---")
#         training_data_tsqa_only = factory.create_paper_reproduction_dataset(
#             model_format='llama', 
#             include_openorca=False
#         )
#         print(f"TSQA-only dataset: {len(training_data_tsqa_only)} samples")
#         if training_data_tsqa_only:
#             print(f"Sample: {training_data_tsqa_only[0][:200]}...")
        
#         # Test with OpenOrca (full paper reproduction)
#         print("\n--- Testing Full Paper Reproduction (with OpenOrca) ---")
#         training_data_full = factory.create_paper_reproduction_dataset(
#             model_format='llama', 
#             include_openorca=True
#         )
#         print(f"Full dataset: {len(training_data_full)} samples")
#         if training_data_full:
#             print(f"Sample: {training_data_full[0][:200]}...")
        
#         # Test different model formats
#         print("\n--- Testing Different Model Formats ---")
#         for model_format in ['mistral', 'qwen']:
#             print(f"\n{model_format.upper()} Format:")
#             training_data = factory.create_training_dataset(
#                 samples_per_task=10,  # Small sample for testing
#                 model_format=model_format,
#                 include_openorca=False
#             )
#             print(f"Samples: {len(training_data)}")
#             if training_data:
#                 print(f"Format: {training_data[0][:150]}...")
        
#     except Exception as e:
#         print(f"Error: {e}")
#         import traceback
#         traceback.print_exc()
#         print("Please ensure the TSQA data path is correct and files exist.")