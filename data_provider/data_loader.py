import os
import json
import logging
import pandas as pd
import torch
import re
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TSQA_Dataloader:
    def __init__(self, root_path: str):
        self.root_path = root_path
        self.task_files = [
            "forecasting.csv",
            "imputation.csv", 
            "classification.csv",
            "anomaly detection.csv",  
            "open_ended_QA.csv"
        ]
        
    def _parse_qa_string(self, qa_string: str) -> Tuple[str, str]:
        if pd.isna(qa_string) or qa_string == "":
            return None, None
            
        try:
            # First try parsing as-is (proper JSON)
            qa_dict = json.loads(qa_string)
            return qa_dict.get("question"), qa_dict.get("answer")
        except json.JSONDecodeError:
            try:
                # Handle malformed JSON - add missing braces
                if qa_string.startswith('"question":'):
                    fixed_string = "{" + qa_string + "}"
                    qa_dict = json.loads(fixed_string)
                    return qa_dict.get("question"), qa_dict.get("answer")
            except (json.JSONDecodeError, AttributeError) as e:
                # logging.warning(f"Failed to parse QA string: {str(e)}")
                return None, None
        except (TypeError, AttributeError):
            return None, None
    
    def load_all_tasks(self) -> Dict[str, pd.DataFrame]:
        logging.info("Loading all TSQA task data...")
        task_data = {}
        
        for file_name in self.task_files:
            file_path = os.path.join(self.root_path, file_name)
            if not os.path.exists(file_path):
                logging.warning(f"File not found: {file_path}. Skipping.")
                continue
                
            df = pd.read_csv(file_path)
            # breakpoint()
            if 'QA_list' not in df.columns:
                logging.warning(f"'QA_list' column not found in {file_name}. Skipping.")
                continue
            
            # Parse QA strings
            qa_pairs = df['QA_list'].apply(self._parse_qa_string)
            # breakpoint()
            df[['question', 'answer']] = pd.DataFrame(qa_pairs.tolist(), index=df.index)
            # breakpoint()
            # Drop rows where parsing failed
            df.dropna(subset=['question', 'answer'], inplace=True)
            df.drop(columns=['QA_list'], inplace=True)
            
            # Store by task name
            task_name = file_name.split('.')[0]
            task_data[task_name] = df
            logging.info(f"Loaded {len(df)} samples for task: {task_name}")
        
        return task_data

class Dataset_TSQA(Dataset):
    def __init__(self, root_path: str, flag: str = 'train', test_size: float = 0.2, val_size: float = 0.15):
        assert flag in ['train', 'val', 'test'], "flag must be one of 'train', 'val', 'test'"
        self.root_path = root_path
        self.flag = flag
        self.data = []
        
        # Load data
        self.__read_data__(test_size, val_size)
    
    def __read_data__(self, test_size: float, val_size: float):
        # Use the TSQA_Dataloader to get organized data
        loader = TSQA_Dataloader(self.root_path)
        task_data = loader.load_all_tasks()
        
        if not task_data:
            raise RuntimeError("No data could be loaded. Check dataset path and file integrity.")
        
        # Combine all tasks
        all_dfs = []
        for task_name, df in task_data.items():
            df['task_source'] = task_name
            all_dfs.append(df)
        
        master_df = pd.concat(all_dfs, ignore_index=True)
        logging.info(f"Total samples from all tasks: {len(master_df)}")
        
        # Split data
        train_val_df, test_df = train_test_split(
            master_df, test_size=test_size, random_state=42, stratify=master_df['task_source']
        )
        
        train_df, val_df = train_test_split(
            train_val_df, 
            test_size=val_size / (1 - test_size), 
            random_state=42,
            stratify=train_val_df['task_source']
        )
        
        # Select appropriate split
        if self.flag == 'train':
            final_df = train_df
        elif self.flag == 'val':
            final_df = val_df
        else:  # test
            final_df = test_df
        
        # Convert to list of dictionaries
        self.data = final_df[['question', 'answer', 'task_source']].to_dict('records')
        logging.info(f"Loaded {len(self.data)} samples for '{self.flag}' split.")
    
    def __getitem__(self, index: int) -> Dict[str, str]:
        return self.data[index]
    
    def __len__(self) -> int:
        return len(self.data)

# if __name__ == '__main__':
#     # Test the data loader
#     try:
        # tsqa_root_path = "/home/sanhorn2/time-mqa/data/tsqa"  # Adjust path as needed
        
#         # Test the base loader
#         loader = TSQA_Dataloader(tsqa_root_path)
#         task_data = loader.load_all_tasks()
        
#         print(f"\n--- Task Data Summary ---")
#         for task_name, df in task_data.items():
#             print(f"{task_name}: {len(df)} samples")
        
#         # Test the PyTorch dataset
#         train_dataset = Dataset_TSQA(root_path=tsqa_root_path, flag='train')
#         val_dataset = Dataset_TSQA(root_path=tsqa_root_path, flag='val')
#         test_dataset = Dataset_TSQA(root_path=tsqa_root_path, flag='test')
        
#         print(f"\n--- Dataset Splits ---")
#         print(f"Training set: {len(train_dataset)} samples")
#         print(f"Validation set: {len(val_dataset)} samples")
#         print(f"Test set: {len(test_dataset)} samples")
        
#         # Show sample
#         if len(train_dataset) > 0:
#             sample = train_dataset[0]
#             print(f"\n--- Sample from Training Set ---")
#             print(f"Task: {sample['task_source']}")
#             print(f"Question: {sample['question'][:200]}...")
#             print(f"Answer: {sample['answer'][:200]}...")
            
#         # Test with DataLoader
#         from torch.utils.data import DataLoader
        
#         train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
#         batch = next(iter(train_loader))
        
#         print(f"\n--- Batch Information ---")
#         print(f"Batch keys: {batch.keys()}")
#         print(f"Batch size: {len(batch['question'])}")
#         print(f"Tasks in batch: {batch['task_source']}")
        
#     except Exception as e:
#         print(f"Error: {e}")
#         print("Please ensure the TSQA data path is correct and files exist.")