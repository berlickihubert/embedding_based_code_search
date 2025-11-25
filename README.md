# Embedding-based Code Search

A system for retrieving relevant code fragments based on a query.

## File Descriptions:

- `demo.ipynb`: A notebook demonstrating the usage of the embedding-based code search system.

- `create_vector_db_from_CoSQA.py`: Contains the implementation for creating vector databases from the CoSQA dataset. It extracts embeddings for code and NLP descriptions and saves them into vector databases.

- `evaluate_model.py`: Implements the evaluation of the embedding models using **Recall@10**, **MRR@10**, and **NDCG@10** metrics.

- `code_search_api.py`: An object-oriented implementation for embedding-based code search. It includes methods to create vector databases, extract function bodies, and perform search queries.

- `model_code.py`: Defines the `CodeEmbeddingModel` class, which generates embeddings for code snippets using the **codesage/codesage-base-v2** model.

- `model_nlp_query.py`: Defines the `NaturalSentenceEmbeddingModel` class, which generates embeddings for natural language queries using the **sentence-transformers/all-MiniLM-L6-v2** model.

- `fine_tune_model_code.py`: Contains the implementation for fine-tuning the `CodeEmbeddingModel`.

- `fine_tune_model_nlp.py`: Contains the implementation for fine-tuning the `NaturalSentenceEmbeddingModel`.

- `requirements.txt`: Lists the dependencies required to run the project.

- `databases`: Stores `usearch` vector databases.

## How to Run

It is recommended to use a virtual environment. Install dependencies using the following command:

```
.\.venv\Scripts\pip3.exe install -r requirements.txt
```

If you want to use CUDA, run the following commands:
```
.\.venv\Scripts\pip3.exe uninstall torch torchvision -y
.\.venv\Scripts\pip3.exe install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

You can find the demo in `demo.ipynb`.

## Insights

### Fine-tuning:
- contrastive loss must be used