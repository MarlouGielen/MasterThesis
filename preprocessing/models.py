from transformers import RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel

class EmbeddingModel:
    def __init__(self, embedding_method):
        print("EmbeddingModel __init__")
        self.embedding_method = embedding_method
        self.model, self.tokenizer = self.get_model_and_tokenizer(embedding_method)
    
    def get_model_and_tokenizer(embedding_method):
        if embedding_method == "CodeBERT":
            tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
            model = RobertaModel.from_pretrained("microsoft/codebert-base")
            model.eval()  # set model to evaluation mode, disabling dropout and batch normalization lay
        elif embedding_method == "UniXcoder":
            tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
            model = AutoModel.from_pretrained("microsoft/unixcoder-base")
            model.eval() # TODO, check if model.eval() set model to evaluation mode, disabling dropout and batch normalization lay
            print("UniXcoder model loaded")
        elif embedding_method == "CodeBERTa":
            tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
            model = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")
            model.eval()  # set model to evaluation mode, disabling dropout and batch normalization lay
        return model, tokenizer