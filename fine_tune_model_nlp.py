import numpy as np
from usearch.index import Index, Matches
from datasets import load_dataset
from model_nlp_query import NaturalSentenceEmbeddingModel
import re
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset


def fine_tune_nlp_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    train_dataset = CodeQueryDataset(
        code_texts=[row['text'] for row in ds_corpus if row['_id'] in corpus_ids_in_test],
        query_texts=[row['text'] for row in ds_queries if row['_id'] in query_ids_in_test],
    )


    train_loader = DataLoader(train_dataset, batch_size=3, shuffle=True)
    print("Number of batches in training data:", len(train_loader))

    nlp_model = NaturalSentenceEmbeddingModel()

    criterion = nn.CosineEmbeddingLoss()
    optimizer_code = optim.Adam(list(nlp_model.model.parameters()), lr=1e-6)
    errors = 0
    for epoch in range(1):
        nlp_model.model.train()
        epoch_loss = 0

        for i, (code_texts, query_texts) in enumerate(train_loader):
            try:
                if (i+1) % 10 == 0:
                    print(f"Batch {i+1}/{len(train_loader)}")

                code_embeddings = nlp_model.get_sentence_embedding(code_texts)
                query_embeddings = nlp_model.get_sentence_embedding(query_texts)
                target = torch.ones(code_embeddings.size(0)).to(device)
                loss = criterion(code_embeddings, query_embeddings, target)

                optimizer_code.zero_grad()
                loss.backward()
                optimizer_code.step()

                epoch_loss += loss.item()
            except Exception as _:
                errors += 1
                print(f"Error in batch {i+1}")

            torch.cuda.empty_cache()

        torch.save(nlp_model.model.state_dict(), "fine-tuned_models/nlp_model_finetuned.pth")
        print(f"Epoch {epoch+1}, Loss: {epoch_loss/(len(train_loader))}")

    print(f"Total errors during training: {errors}")
        
