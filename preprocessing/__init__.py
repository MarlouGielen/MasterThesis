# this init .py makes the folder a package
from preprocessing.models import EmbeddingModel #get_model_and_tokenizer


embedding_method = "UniXcoder"
model, tokenizer = EmbeddingModel.get_model_and_tokenizer(embedding_method)
print("EmbeddingModel object created")
print(model)
print(tokenizer)