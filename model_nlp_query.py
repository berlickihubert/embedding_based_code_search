from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"


class NaturalSentenceEmbeddingModel:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2', model_path=None):
        print("Using device for nlp embedding model:", device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")

        self.model = self.model.to(device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_sentence_embedding(self, sentence):
        encoded_input = self.tokenizer(sentence, padding=True, truncation=True, return_tensors='pt').to(device)
        model_output = self.model(**encoded_input)

        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings


if __name__ == "__main__":
    embedding_model = NaturalSentenceEmbeddingModel()
    sentence = ['Sentence one.', 'Sentence two.']
    embeddings = embedding_model.get_sentence_embedding(sentence)
    print(embeddings.size())
