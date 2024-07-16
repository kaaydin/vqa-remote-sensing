from glob import glob
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class VQADataset(Dataset):
    def __init__(self, textual_path, visual_path, load_images=False):
        self.textual_path = textual_path
        self.visual_path = visual_path

        # List files and sort them to maintain a consistent order
        self.files = sorted(glob(os.path.join(textual_path, "*.pt")))
        self.image_files = sorted(glob(os.path.join(visual_path, "*.pt")))

        # Store separate items for all questions and answers
        self.items = []
        progress_bar = tqdm(self.files, desc="Loading files", total=len(self.files))
        for file_path in progress_bar:
            # Load the Q/A pairs from the file
            qa_pairs = torch.load(file_path)
            for qa_pair in qa_pairs:
                self.items.append(qa_pair)
        if load_images:
            self.image_ids = []
            progress_bar = tqdm(self.items, desc="Loading image IDs", total=len(self.items))
            for item in progress_bar:
                image_id = int(item["image_id"])
                if image_id not in self.image_ids:
                    self.image_ids.append(image_id)
            # sort image ids
            self.image_ids.sort()
            self.images = {}
            progress_bar = tqdm(self.image_ids, desc="Loading images", total=len(self.image_ids))
            for image_id in progress_bar:
                # Load the image
                image = torch.load(os.path.join(visual_path, f"{image_id}.pt"))
                self.images[image_id] = image
        else:
            self.images = None


    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]

        question = item["question"]
        answer = item["answer"]
        question_type = item["question_type"]
        image_id = int(item["image_id"])

        # Load the image associated with this Q/A pair
        if self.images is not None:
            image = self.images[image_id]
        else:
            image = torch.load(os.path.join(self.visual_path, f"{image_id}.pt"))

        return question, answer, image, question_type