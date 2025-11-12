import numpy as np
from usearch.index import Index, Matches
from datasets import load_dataset
from model_nlp_query import NaturalSentenceEmbeddingModel
from model_code import CodeEmbeddingModel

def evaluate_model(code_model_path = None, nlp_model_path = None, split = "test"):
    ds_default = load_dataset("CoIR-Retrieval/cosqa", "default")
    ds_corpus = load_dataset("CoIR-Retrieval/cosqa", "corpus")
    ds_queries = load_dataset("CoIR-Retrieval/cosqa", "queries")
    print("Datasets loaded.")


    db_query = Index(
        ndim=384, # Define the number of dimensions in input vectors
        metric='cos', # Choose 'l2sq', 'haversine' or other metric, default = 'ip'
        dtype='f32', # Quantize to 'f16' or 'i8' if needed, default = 'f32'
        connectivity=16, # How frequent should the connections in the graph be, optional
        expansion_add=128, # Control the recall of indexing, optional
        expansion_search=64, # Control the quality of search, optional
    )

    db_code = Index(
        ndim=1024, # Define the number of dimensions in input vectors
        metric='cos', # Choose 'l2sq', 'haversine' or other metric, default = 'ip'
        dtype='f32', # Quantize to 'f16' or 'i8' if needed, default = 'f32'
        connectivity=16, # How frequent should the connections in the graph be, optional
        expansion_add=128, # Control the recall of indexing, optional
        expansion_search=64, # Control the quality of search, optional
    )

    try: 
        if nlp_model_path is None:
            db_query.load('databases/db_query_cosqa.usearch')
            print("Query vector database loaded from file")
        else:
            db_query.load('databases/db_query_cosqa_finetuned.usearch')
            print("Query vector database loaded from file for finetuned model")
    except FileNotFoundError:
        print("No existing query vector database file.")

    try:
        if code_model_path is None:
            db_code.load('databases/db_code_cosqa.usearch')
            print("Code vector database loaded from file")
        else:
            db_code.load(f'databases/db_code_cosqa_finetuned_{split}.usearch')
            print("Code vector database loaded from file for finetuned model") 

    except FileNotFoundError:
        print("No existing code vector database file.")


    ds_corpus = ds_corpus["corpus"]
    ds_queries = ds_queries["queries"]
    ds_default_train = ds_default["train"]
    ds_default_valid = ds_default["valid"]
    ds_default_test = ds_default["test"]

    code_embedding_model = CodeEmbeddingModel(model_path=code_model_path)
    code_embedding_model.model.eval()
    nlp_embedding_model = NaturalSentenceEmbeddingModel(model_path=nlp_model_path)

    recall10_code_sum = 0
    recall10_query_sum = 0

    mrr10_code_sum = 0
    mrr10_query_sum = 0

    ndcg10_code_sum = 0
    ndcg10_query_sum = 0

    divisor = 0

    query_ids_in_test = set(row['query-id'] for row in ds_default_test)
    for i, row in enumerate(ds_queries):
        if (row['_id'] not in query_ids_in_test):
            continue

        query_text = row['text']
        code_embedding = code_embedding_model.get_sentence_embedding([query_text]).cpu().detach().numpy()[0]
        description_embedding = nlp_embedding_model.get_sentence_embedding([query_text]).cpu().detach().numpy()[0]

        matches_queries: Matches = db_query.search(description_embedding, 10)
        matches_code: Matches = db_code.search(code_embedding, 10)

        integer_id = int(row['_id'][1:])

        # print("Matches for queries:")
        # for match in matches_queries:
        #     print(f"Key: {match.key}, Distance: {match.distance}")

        # print("Matches for code:")
        # for match in matches_code:
        #     print(f"Key: {match.key}, Distance: {match.distance}")

        # Calculate Recall@10
        if(any(match.key == integer_id for match in matches_queries)):
            recall10_query_sum += 1
        if(any(match.key == integer_id for match in matches_code)):
            recall10_code_sum += 1

        # Calculate MRR@10
        for idx, match in enumerate(matches_queries):
            if match.key == integer_id:
                mrr10_query_sum += 1 / (idx + 1)
                break
        for idx, match in enumerate(matches_code):
            if match.key == integer_id:
                mrr10_code_sum += 1 / (idx + 1)
                break

        # Calculate NDCG@10
        for idx, match in enumerate(matches_queries):
            if match.key == integer_id:
                ndcg10_query_sum += 1 / np.log2(idx + 2)
                break
        for idx, match in enumerate(matches_code): 
            if match.key == integer_id:
                ndcg10_code_sum += 1 / np.log2(idx + 2)
                break

        divisor += 1

    print()
    print(f"Recall@10 for code embeddings: {recall10_code_sum}/{divisor} = {recall10_code_sum/divisor:.4f}")
    print(f"Recall@10 for query embeddings: {recall10_query_sum}/{divisor} = {recall10_query_sum/divisor:.4f}\n")

    print(f"MRR@10 for code embeddings: {mrr10_code_sum}/{divisor} = {mrr10_code_sum/divisor:.4f}")
    print(f"MRR@10 for query embeddings: {mrr10_query_sum}/{divisor} = {mrr10_query_sum/divisor:.4f}\n")

    print(f"NDCG@10 for code embeddings: {ndcg10_code_sum}/{divisor} = {ndcg10_code_sum/divisor:.4f}")
    print(f"NDCG@10 for query embeddings: {ndcg10_query_sum}/{divisor} = {ndcg10_query_sum/divisor:.4f}")

