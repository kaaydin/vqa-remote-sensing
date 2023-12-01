from glob import glob
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np


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

        ## Create dictionary with image_id as key and image as value
        self.images = {}
        progress_bar = tqdm(self.image_ids, desc="Loading images", total=len(self.image_ids))
        for image_id in progress_bar:
            # Load the image
            image = torch.load(os.path.join(visual_path, f"{image_id}.pt"))
            self.images[image_id] = image        

    def __len__(self):
        return len(self.text_ids)

    def __getitem__(self, idx):
        
        item = self.text_ids[idx]
        ## Load text information
        text = torch.load(os.path.join(self.textual_path, f"{item}.pt"))

        ## Retrieve text information
        question = text["question"]
        answer = text["answer"]
        question_type = text["question_type"]
        image_id = int(text["image_id"])

        # Load the image associated with this Q/A pair
        image = self.images[image_id]

        return question, answer, image, question_type