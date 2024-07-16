from glob import glob
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

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
        self.files = sorted(glob(os.path.join(textual_path, "*.pt")))

        # Store separate items for all questions and answers
        self.items = []
        progress_bar = tqdm(self.files, desc="Loading files", total=len(self.files))
        for file_path in progress_bar:
            # Load the Q/A pairs from the file
            qa_pairs = torch.load(file_path)
            for qa_pair in qa_pairs:
                self.items.append(qa_pair)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        question = item["question"]
        answer = item["answer"]
        question_type_str = item["question_type"]
        question_type_idx = question_type_to_idx[question_type_str]
        image_id = int(item["image_id"])

        # Load the image associated with this Q/A pair
        image = torch.load(os.path.join(self.visual_path, f"{image_id}.pt"))

        return question, answer, image, question_type_idx, question_type_str
