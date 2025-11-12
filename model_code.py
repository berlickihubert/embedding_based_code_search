from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F


device = "cuda" if torch.cuda.is_available() else "cpu"


class CodeEmbeddingModel:
    def __init__(self, model_name="codesage/codesage-base-v2", model_path=None):
        print("Using device for code embedding model:", device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, add_eos_token=True)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        if model_path is not None:
            self.model.load_state_dict(torch.load(model_path))
            print(f"Loaded model weights from {model_path}")

        self.model = self.model.to(device)

    def mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def get_sentence_embedding(self, sentence):
        inputs = self.tokenizer(sentence,  padding=True, truncation=True, return_tensors="pt").to(device)
        model_outputs = self.model(**inputs)
        embedding = self.mean_pooling(model_outputs, inputs['attention_mask'])
        embedding = F.normalize(embedding, p=2, dim=1)
        return embedding
    

if __name__ == "__main__":
    embedding_model = CodeEmbeddingModel()
    sentence = ['Sentence one.', 'Sentence two.']
    embeddings = embedding_model.get_sentence_embedding(sentence)
    print(embeddings.size())