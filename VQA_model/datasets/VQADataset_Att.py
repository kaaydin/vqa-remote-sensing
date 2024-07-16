from glob import glob
import os
import torch
from torch.utils.data import Dataset
import numpy as np


question_type_to_idx = {
    "count": 3,
    "presence": 0,
    "area": 2,
    "comp": 1,
}

class VQADataset(Dataset):
    def __init__(self, textual_path, visual_path):
        self.textual_path = textual_path
        self.visual_path = visual_path

        # List files and sort them to maintain a consistent order
        self.text_files = sorted(glob(os.path.join(textual_path, "*.pt")))
        self.image_files = sorted(glob(os.path.join(visual_path, "*.pt")))

        ## Create array with integers from 0 to len(self.image_files) and 0 to len(self.text_files)
        self.image_ids = np.arange(len(self.image_files))
        self.text_ids = np.arange(len(self.text_files))      

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        
        item = self.text_ids[idx]
        ## Load text information
        text = torch.load(os.path.join(self.textual_path, f"{item}.pt"))

        ## Retrieve text information
        question = text["question"]
        answer = text["answer"]
        question_type_str = text["question_type"]
        question_type_idx = question_type_to_idx[question_type_str]
        image_id = int(text["image_id"])
        #question_str = text["question_string"]

        # Load the image associated with this Q/A pair
        image = torch.load(os.path.join(self.visual_path, f"{image_id}.pt"))

        return question, answer, image, question_type_idx, question_type_str