import torch
import json
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import os
from PyPDF2 import PdfFileReader

class MultiSourceDataset(Dataset):
    def __init__(self, tokenizer, data_dir, books_dir, block_size=512):
        self.examples = []

        for filename in os.listdir(data_dir):
            if filename.endswith('.json'):
                file_path = os.path.join(data_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for item in data:
                        if 'text' in item and len(item['text']) > 0:
                            encoded_text = tokenizer.encode(item['text'], add_special_tokens=True)
                            if len(encoded_text) > 0:
                                self.examples.append(encoded_text)

        for filename in os.listdir(books_dir):
            if filename.endswith('.pdf'):
                pdf_path = os.path.join(books_dir, filename)
                with open(pdf_path, 'rb') as pdf_file:
                    pdf_reader = PdfFileReader(pdf_file)
                    for page_num in range(pdf_reader.numPages):
                        page_text = pdf_reader.getPage(page_num).extractText()
                        if len(page_text) > 0:
                            encoded_text = tokenizer.encode(page_text, add_special_tokens=True)
                            if len(encoded_text) > 0:
                                self.examples.append(encoded_text)

        self.block_size = block_size

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        encoded_text = torch.tensor(self.examples[i])
        if len(encoded_text) < self.block_size:
            padding = torch.zeros(self.block_size - len(encoded_text), dtype=torch.long)
            encoded_text = torch.cat((encoded_text, padding), dim=0)
        elif len(encoded_text) > self.block_size:
            encoded_text = encoded_text[:self.block_size]
        return encoded_text

def main(save_path='/home/drumea/Desktop/droneAI/NLPAI/Antonela'):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    data_dir = '/home/drumea/Desktop/droneAI/NLPdataset/data'
    books_dir = '/home/drumea/Desktop/droneAI/NLPdataset/databooks'

    dataset = MultiSourceDataset(tokenizer, data_dir, books_dir)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()

    optimizer = AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(data_loader))

    num_epochs = 1
    total_steps = num_epochs * len(data_loader)
    
    progress_bar = tqdm(total=total_steps, desc='Training')

    for epoch in range(num_epochs):
        for batch in data_loader:
            inputs = batch.to(device)
            outputs = model(inputs, labels=inputs)
            loss = outputs.loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            progress_bar.update(1)
            progress_bar.set_postfix({'Loss': loss.item()})

    progress_bar.close()
    model.save_pretrained(save_path)

if __name__ == "__main__":
    main()