import numpy as np
from usearch.index import Index, Matches
from datasets import load_dataset
from model_nlp_query import NaturalSentenceEmbeddingModel
from model_code import CodeEmbeddingModel
import re

def create_vector_db_from_cosqa(code_model_path = None, nlp_model_path = None, split = "test"):

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
        else:
            db_query.load('databases/db_query_cosqa_finetuned.usearch')

        print("Query vector database loaded from file")
    except FileNotFoundError:
        print("No existing query vector database file.")

    try:
        if code_model_path is None:
            db_code.load('databases/db_code_cosqa.usearch')
        else:
            db_code.load(f'databases/db_code_cosqa_finetuned_{split}.usearch')

        print("Code vector database loaded from file")
    except FileNotFoundError:
        print("No existing code vector database file.")


    # vector = np.array([0.2, 0.6, 0.4])
    # db_query.add(42, vector)
    # matches: Matches = db_query.search(vector, 10)
    # print(matches)
    # print(db_query[42])
    # db_query.save('db_query.usearch')

    # assert len(index) == 1
    # assert len(matches) == 1
    # assert matches[0].key == 42
    # assert matches[0].distance <= 0.001
    # assert np.allclose(index[42], vector)

    ds_corpus = ds_corpus["corpus"]
    ds_queries = ds_queries["queries"]
    ds_default_train = ds_default["train"]
    ds_default_valid = ds_default["valid"]
    ds_default_test = ds_default["test"]

    code_embedding_model = CodeEmbeddingModel(model_path = code_model_path)
    code_embedding_model.model.eval()
    nlp_embedding_model = NaturalSentenceEmbeddingModel(model_path= nlp_model_path)

    def extract_function_description(function_body):
        match = re.search(r'"""(.*?)"""', function_body, re.DOTALL)
        if match:
            return match.group(1).strip()

        return ""


    corpus_ids_in_test = set(row['corpus-id'] for row in ds_default_test)
    for i, row in enumerate(ds_corpus):
        if row['_id'] not in corpus_ids_in_test:
            continue

        if i % 50 == 0:
            print(f"Processing row {i}/{len(ds_corpus)}")
        code_text = row['text']
        code_embedding = code_embedding_model.get_sentence_embedding([code_text]).cpu().detach().numpy()[0]
        description_embedding = nlp_embedding_model.get_sentence_embedding([extract_function_description(code_text)]).cpu().detach().numpy()[0]

        try:
            db_query.add(i+1, description_embedding)
        except Exception as e:
            pass

        try:
            db_code.add(i+1, code_embedding)
        except Exception as e:
            pass


    if code_model_path is None:
        db_code.save('databases/db_code_cosqa.usearch')
    else:
        db_code.save(f'databases/db_code_cosqa_finetuned_{split}.usearch')
    
    if nlp_model_path is None:
        db_query.save('databases/db_query_cosqa.usearch')
    else:
        db_query.save('databases/db_query_cosqa_finetuned.usearch')