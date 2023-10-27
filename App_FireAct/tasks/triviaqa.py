import json
import re
import string
from .base import BaseTask, DATA_DIR
from collections import Counter


class TriviaQATask(BaseTask):
    def __init__(self, split):
        data_file = f"{DATA_DIR}/triviaqa/{split}.json"
        self.data = json.load(open(data_file))

    def __getitem__(self, idx):
        return self.data["Data"][idx]["Question"]

    def __len__(self):
        return len(self.data["Data"])

    def evaluate(self, idx, answer):
        return 1
    
    def get_prompt(self):
        with open(f"{DATA_DIR}/../prompts/triviaqa_multiqueries.txt", "r") as fin:
            prompt = fin.read() 
        return prompt