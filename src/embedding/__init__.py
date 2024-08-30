import torch


def load_model_tokenizer(embedding_method = "UniXcoder"):
    """
    Function to load the model and tokenizer for the embedding method.

    :param embedding_method (str): the embedding method to use

    :return tokenizer: the tokenizer to use
    :return model: the model to use
    """
    if embedding_method == "DistilBERT":
        from transformers import DistilBertModel, DistilBertTokenizer
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        model = DistilBertModel.from_pretrained("distilbert-base-uncased")
    elif embedding_method == "UniXcoder":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/unixcoder-base")
        model = AutoModel.from_pretrained("microsoft/unixcoder-base")
        #print('MAX LENGTH MODEL: ', tokenizer.model_max_length)
    elif embedding_method == "CodeBERT":
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
        model = RobertaModel.from_pretrained("microsoft/codebert-base")
    elif embedding_method == "CodeBERTa":
        from transformers import RobertaTokenizer, RobertaModel
        tokenizer = RobertaTokenizer.from_pretrained("huggingface/CodeBERTa-small-v1")
        model = RobertaModel.from_pretrained("huggingface/CodeBERTa-small-v1")
    elif embedding_method == "GraphCodeBERT":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
        model = AutoModel.from_pretrained("microsoft/graphcodebert-base")
    elif embedding_method == "CodeT5":
        from transformers import AutoModel, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("Salesforce/codet5-base")
        model = AutoModel.from_pretrained("Salesforce/codet5-base")
    else:
        raise ValueError("Unknown embedding method specified, try 'UniXcoder', 'CodeBERT', 'CodeBERTa', 'GraphCodeBERT', or 'CodeT5'")

    return tokenizer, model


def get_embedding(cell, tokenizer, model, embedding_method, print_why=False):
    """
    Get the embedding of content and explain the prediction using SHAP values.

    :param content (str): the content to embed
    :param tokenizer: the tokenizer to use
    :param model: the model to use
    :param cell_index (int): the index of the cell

    :return embedding (torch.Tensor): the embedding of the content
    """

    source = cell.source
    idx = 0
    try:
        idx = cell.mc_idx  
    except:
        None  
    try:
        idx = cell.nb_idx
    except:
        None

    if not source or source == "" or source is None:
        #print("Input content is empty, idx:", idx) 
        None

    source = source.replace("\n", " ") # Remove newlines
    source = source.replace("\t", " ") # Remove tabs
    source = source.strip() # Remove leading/trailing whitespaces

    inputs = tokenizer(source, return_tensors="pt", padding=True, truncation=True, max_length=256) # Reduce max_length

    try:
        outputs = model(**inputs)
        cell.emb = len(source)
    except:
        outputs = None

    if len(source) > 1024 and outputs is None:
        source = source[:1024]
        
    try:
        outputs = model(**inputs)
        cell.emb = 1024
    except:
        outputs = None

    if len(source) > 512 and outputs is None:
        source = source[:512]

    try:
        outputs = model(**inputs)
        cell.emb = 512
    except Exception as e:
        None
        cell.emb = 0


    # model failed
    if outputs is None:
        #print("Output is None, idx:", idx) 
        return torch.zeros(1, 768)
    
    # model worked
    pooled_output = torch.mean(outputs.last_hidden_state, dim=1)

    return pooled_output


def get_code_embedding(cell, tokenizer, model, embedding_method, print_why=False):
    """
    Get the embedding of content and explain the prediction using SHAP values.

    :param content (str): the content to embed
    :param tokenizer: the tokenizer to use
    :param model: the model to use
    :param cell_index (int): the index of the cell

    :return embedding (torch.Tensor): the embedding of the content
    """

    source = cell.code_source
    idx = 0
    try:
        idx = cell.mc_idx  
    except:
        None  
    try:
        idx = cell.nb_idx
    except:
        None

    if not source or source == "" or source is None:
        #print("Input content is empty, idx:", idx) 
        None

    source = source.replace("\n", " ") # Remove newlines
    source = source.replace("\t", " ") # Remove tabs
    source = source.strip() # Remove leading/trailing whitespaces

    inputs = tokenizer(source, return_tensors="pt", padding=True, truncation=True, max_length=256) # Reduce max_length

    try:
        outputs = model(**inputs)
        cell.emb = len(source)
    except:
        outputs = None

    if len(source) > 1024 and outputs is None:
        source = source[:1024]
        
    try:
        outputs = model(**inputs)
        cell.emb = 1024
    except:
        outputs = None

    if len(source) > 512 and outputs is None:
        source = source[:512]

    try:
        outputs = model(**inputs)
        cell.emb = 512
    except Exception as e:
        None
        cell.emb = 0


    # model failed
    if outputs is None:
        #print("Output is None, idx:", idx) 
        return torch.zeros(1, 768)
    
    # model worked
    pooled_output = torch.mean(outputs.last_hidden_state, dim=1)

    return pooled_output