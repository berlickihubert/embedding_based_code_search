import numpy as np
from usearch.index import Index, Matches
from model_nlp_query import NaturalSentenceEmbeddingModel
from model_code import CodeEmbeddingModel
import re
import os


class CodeSearchAPI:
    def __init__(self):
        self.db_query = Index(
            ndim=384,
            metric='cos',
            dtype='f32',
            connectivity=16,
            expansion_add=128,
            expansion_search=64,
        )

        self.db_code = Index(
            ndim=1024,
            metric='cos',
            dtype='f32',
            connectivity=16,
            expansion_add=128,
            expansion_search=64,
        )

        try: 
            self.db_query.load('databases/db_query_demo.usearchh')
            print("Query vector database loaded from file")
        except FileNotFoundError:
            print("No existing query vector database file.")

        try: 
            self.db_code.load('databases/db_code_demo.usearch')
            print("Code vector database loaded from file")
        except FileNotFoundError:
            print("No existing code vector database file.")

        self.nlp_embedding_model = NaturalSentenceEmbeddingModel()
        self.code_embedding_model = CodeEmbeddingModel()

    def extract_function_bodies(self, file_path):
        import ast

        with open(file_path, 'r') as file:
            tree = ast.parse(file.read(), filename=file_path)

        function_bodies = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_name = node.name
                function_body = ast.get_source_segment(open(file_path).read(), node)
                function_bodies.append((function_name, function_body))

        return function_bodies

    def create_vector_db(self, file_path):
        try:
            if os.path.exists('databases/db_code_demo.usearch'):
                os.remove('databases/db_code_demo.usearch')
            if os.path.exists('databases/db_query_demo.usearch'):
                os.remove('databases/db_query_demo.usearch')

            self.db_query = Index(
                ndim=384,
                metric='cos',
                dtype='f32',
                connectivity=16,
                expansion_add=128,
                expansion_search=64,
            )

            self.db_code = Index(
                ndim=1024,
                metric='cos',
                dtype='f32',
                connectivity=16,
                expansion_add=128,
                expansion_search=64,
            )

            function_bodies = self.extract_function_bodies(file_path)

            for i, (func_name, func_body) in enumerate(function_bodies):
                code_embedding = self.code_embedding_model.get_sentence_embedding([func_body]).cpu().detach().numpy()[0]
                name_embedding = self.nlp_embedding_model.get_sentence_embedding([func_name]).cpu().detach().numpy()[0]
                self.db_code.add(i+1, code_embedding)
                self.db_query.add(i+1, name_embedding)

            self.db_code.save('databases/db_code_demo.usearch')
            self.db_query.save('databases/db_query_demo.usearch')
        except Exception as e:
            print("Error, check if database doesn't already exist")

    def search(self, query):
        query_embedding = self.nlp_embedding_model.get_sentence_embedding([query]).cpu().detach().numpy()[0]
        matches_query: Matches = self.db_query.search(query_embedding, 10)

        code_embedding = self.code_embedding_model.get_sentence_embedding([query]).cpu().detach().numpy()[0]
        matches_code: Matches = self.db_code.search(code_embedding, 10)
        # print("Query matches to function names:")
        # print([{"key": match.key, "distance": match.distance} for match in matches_query])
        # print("Query matches to function bodies:")
        # print([{"key": match.key, "distance": match.distance} for match in matches_code])
        return(matches_query, matches_code)
