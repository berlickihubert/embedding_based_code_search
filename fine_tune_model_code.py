import numpy as np
from usearch.index import Index, Matches
from datasets import load_dataset
from model_code import CodeEmbeddingModel
import re
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

device = "cuda" if torch.cuda.is_available() else "cpu"

def fine_tune_code_model(split = "test"):    
    ds_default = load_dataset("CoIR-Retrieval/cosqa", "default")
    ds_corpus = load_dataset("CoIR-Retrieval/cosqa", "corpus")
    ds_queries = load_dataset("CoIR-Retrieval/cosqa", "queries")

    ds_corpus = ds_corpus["corpus"]
    ds_queries = ds_queries["queries"]
    ds_default_train = ds_default["train"]
    ds_default_valid = ds_default["valid"]
    ds_default_test = ds_default["test"]
    print("Datasets loaded.")

    corpus_ids_in_train = set(row['corpus-id'] for row in ds_default_train)
    query_ids_in_train = set(row['query-id'] for row in ds_default_train)

    query_ids_in_test = set(row['query-id'] for row in ds_default_test)
    corpus_ids_in_test = set(row['corpus-id'] for row in ds_default_test)

    class CodeQueryDataset(Dataset):
        def __init__(self, code_texts, query_texts):
            self.code_texts = code_texts
            self.query_texts = query_texts

        def __len__(self):
            return len(self.code_texts)

        def __getitem__(self, idx):
            return self.code_texts[idx], self.query_texts[idx]

    if split == "train":
        train_dataset = CodeQueryDataset(
            code_texts=[row['text'] for row in ds_corpus if row['_id'] in corpus_ids_in_train],
            query_texts=[row['text'] for row in ds_queries if row['_id'] in query_ids_in_train],
        )
    else:
        train_dataset = CodeQueryDataset(
            code_texts=[row['text'] for row in ds_corpus if row['_id'] in corpus_ids_in_test],
            query_texts=[row['text'] for row in ds_queries if row['_id'] in query_ids_in_test],
        )


    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)

    print("Number of batches in training data:", len(train_loader))

    code_model = CodeEmbeddingModel()

    criterion = nn.CosineEmbeddingLoss()
    criterion_contrastive = nn.CosineEmbeddingLoss()
    optimizer_code = optim.Adam(list(code_model.model.parameters()), lr=1e-5)
    errors = 0
    for epoch in range(1):
        code_model.model.train()
        epoch_loss = 0

        for i, (code_texts, query_texts) in enumerate(train_loader):
            try:
                if (i+1) % 50 == 0:
                    print(f"Batch {i+1}/{len(train_loader)}")

                code_embeddings = code_model.get_sentence_embedding(code_texts)
                query_embeddings = code_model.get_sentence_embedding(query_texts)

                target_positive = torch.ones(code_embeddings.size(0)).to(device)
                loss_positive = criterion(code_embeddings, query_embeddings, target_positive)


                if code_embeddings.size(0) > 1:
                    query_shifted = torch.roll(query_embeddings, shifts=1, dims=0)
                    contrastive_loss = criterion_contrastive(code_embeddings, query_shifted, -torch.ones(code_embeddings.size(0)).to(device))
                else:
                    contrastive_loss = torch.tensor(0.0, device=device)

                total_loss = loss_positive + contrastive_loss

                optimizer_code.zero_grad()
                total_loss.backward()
                optimizer_code.step()

                epoch_loss += total_loss.item()
            except Exception as e:
                errors += 1
                print(f"Error in batch {i+1} : {e}")

            if split == "train" and i > (len(train_loader) / 2):
                break # This already takes really long :( so we limit to half an epoch
                

            torch.cuda.empty_cache()

        torch.save(code_model.model.state_dict(), f"fine-tuned_models/code_model_finetuned_{split}.pth")
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/(len(train_loader))}")

    print(f"Total errors during training: {errors}")

