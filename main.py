import os

from src.data_processing import load_data, save_notebooks_to_json
from src.code_classification.code_classification import classify_nb_cells
from src.similarity.calculate_similarities import calculate_similarities

# Next function
import time
import openai
import pandas as pd

model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF"

def local_LLM(model, instruction, content, temperature = 0.5, max_retries=5):

    for attempt in range(max_retries):
        try:
            client = openai(base_url="http://localhost:1234/v1", api_key="lm-studio")
            response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": instruction},
                {"role": "user", "content": content}
            ],
            temperature=temperature,
            )

            if response.choices:
                return response.choices[0].message['content']
            else:
                return None
    
        except:
            print(f"Attempt {attempt + 1} failed")
            time.sleep(20 * (2 ** attempt))  # Exponential backoff
            response = None

    return response


def get_explanation_nb(notebooks, model, data_path, temperature = 0.5, max_retries=5):
    
    # make pd dataframe with nb_idx, sum_source, sum_keywords
    nb_expl = pd.DataFrame(columns=['nb_idx', 'sum_source', 'sum_keywords'])
    # instruction_sum_nb = "Give a summary of around 3 sentences, about the provided Python notebook. Please summarize the content of the notebook. Answer with a summary of around 3 sentences, no additional text."
    # instruction_sum_nb_words = "Explain with around keywords, what the Python notebook is mainly about. Reply with the list of keywords separated by commas. The keywords should characterize the notebooks content and be specific for this notebook, such as the types of models used, the types of machine learning operations used, the types of evaluation metrics or hyperparametertuning. Answer with comma separated keywords only. You only answer with the keywords separated by commas, no additional text."
    
    instruction_sum_nb = (
        "Give a summary of around 3 sentences, about the provided Python notebook. "
        "Please summarize the content of the notebook. Answer with a summary of around 3 sentences, no additional text."
    )
    instruction_sum_nb_words = (
        "Explain with a list of keywords, what the Python notebook is mainly about. "
        "Reply with the list of keywords separated by commas. The keywords should characterize the notebook's content and be specific for this notebook, "
        "such as the types of models used, the types of machine learning operations used, the types of evaluation metrics or hyperparameter tuning. "
        "Answer with comma separated keywords only. You only answer with the keywords separated by commas, no additional text."
    )

    for nb in notebooks:
        nb_idx = nb.nb_idx
        source = nb.source
        response_summary = local_LLM(model, instruction_sum_nb, source, temperature, max_retries)
        response_keywords = local_LLM(model, instruction_sum_nb_words, source, temperature, max_retries)
    
        nb_expl = nb_expl.append({'nb_idx': nb_idx, 'sum_source': response_summary, 'sum_keywords': response_keywords}, ignore_index=True)  

    # save to csv
    nb_expl.to_csv(data_path, index=False)

    return nb_expl


# markdown cell, code cell


def get_explanation_cell(notebooks, model, data_path, temperature = 0.5, max_retries=5):
    
    nb_expl = pd.DataFrame(columns=['nb_idx', 'mc_idx', 'sum_cell', 'sum_keywords', 'classification', 'keywords_classification'])
    instruction_sum_code_cell = "Give a summary of max 3 sentences, about the provided Python code snippet. Please summarize the content of the cell. Summarize what happens in the cell, what is the semantic meaning, the intention. Answer with a summary of max 3 sentences, no additional text. "
    instruction_sum_markdown_cell = "Give a summary of 1 sentence, about the provided markdown cell. Please summarize the content of the cell. Summarize what happens in the cell, what is the semantic meaning, the intention. Answer with a summary of max 3 sentences, no additional text. "
    
    instruction_code_cell_words = "Explain with a list of keywords, what the Python code snippet is mainly about. Reply with the list of keywords separated by commas. The keywords should characterize the cell's content and be specific for this cell, such as the types of models used, the types of machine learning operations used, the types of evaluation metrics or hyperparametertuning. Answer with comma separated keywords only. You only answer with the keywords separated by commas, no additional text."
    instruction_markdown_cell_words = "Explain with a list of keywords, what the markdown cell is mainly about. Reply with the list of keywords separated by commas. The keywords should characterize the cell's content and be specific for this cell, such as the types of models used, the types of machine learning operations used, the types of evaluation metrics or hyperparametertuning. Answer with comma separated keywords only. You only answer with the keywords separated by commas, no additional text."

    instruction_code_cell_classification = "Classify the code cell. Reply with the classification of the code cell. The classification should be one of the following: 'Environment', 'Data_Extraction', 'Exploratory_Data_Analysis', 'Data_Transform', 'Model_Train', 'Model_Evaluation', 'Model_Interpretation', 'Hyperparameter_Tuning', 'Visualization', 'Debug', 'Data_Export','Other'. Answer with the classification only, no additional text. Only answer with the classificaiton (e.g. 'Exploratory_Data_Analysis'). Answer with the classification only, no additional text."
    instruction_markdown_cell_classification = "Classify the markdown cell. Reply with the classification of the markdown cell. The classification should be one of the following: 'Description_next_cell', 'Result_previous_cell', 'Headline', 'Summary', 'Image', 'Other'. Answer with the classification only, no additional text. Only answer with the classificaiton (e.g. 'Description_next_cell'). Answer with the classification only, no additional text."  
    
    instruction_code_cell_keywords_classification = "Classify the code cell with keywords. Reply with the keywords for the chosen classification, which is one of the following: 'Environment', 'Data_Extraction', 'Exploratory_Data_Analysis', 'Data_Transform', 'Model_Train', 'Model_Evaluation', 'Model_Interpretation', 'Hyperparameter_Tuning', 'Visualization', 'Debug', 'Data_Export','Other'. Only answer with the keywords, in the form of a list ['','','']. Answer with the keywords only, no additional text."
    instruction_markdown_cell_keywords_classification = "Classify the markdown cell with keywords. Reply with the keywords for the chosen classification of the markdown cell, which is one of the following. 'Description_next_cell', 'Result_previous_cell', 'Headline', 'Summary', 'Image', 'Other'. Only answer with the keywords, in the form of a list ['','','']. Answer with the keywords only, no additional text."
    
    for nb in notebooks:
        for cell in nb.all_cells:
            content = cell.content
            cell_type = cell.cell_type

            if cell_type == 'code':
                response_summary = local_LLM(model, instruction_sum_code_cell, content, temperature, max_retries)
                response_keywords = local_LLM(model, instruction_code_cell_words, content, temperature, max_retries)
                response_classification = local_LLM(model, instruction_code_cell_classification, content, temperature, max_retries)
                response_keywords_classification = local_LLM(model, instruction_code_cell_keywords_classification, content, temperature, max_retries)
            else:
                response_summary = local_LLM(model, instruction_sum_markdown_cell, content, temperature, max_retries)
                response_keywords = local_LLM(model, instruction_markdown_cell_words, content, temperature, max_retries)
                response_classification = local_LLM(model, instruction_markdown_cell_classification, content, temperature, max_retries)
                response_keywords_classification = local_LLM(model, instruction_markdown_cell_keywords_classification, content, temperature, max_retries)

            nb_expl = nb_expl.append({'nb_idx': nb.nb_idx, 'mc_idx': cell.mc_idx, 'sum_cell': response_summary, 'sum_keywords': response_keywords, 'classification': response_classification, 'keywords_classification': response_keywords_classification}, ignore_index=True)

       
    # save to csv
    nb_expl.to_csv(data_path, index=False)     

    return nb_expl


# Configuration
data = "Kaggle" #TODO: for each map in data_map

data_path = 'data/data_'+ data + '/'
embedding_method = "UniXcoder"
classification_method = "all"
nb_limit=120

# Paths
input_path = os.path.join(data_path, 'raw/')
output_path = os.path.join(data_path, 'processed/') #TODO: only add embedding method here to all :)
cell_path = os.path.join(data_path, 'processed/cells/')
expl_path = os.path.join(data_path, 'processed/expl/')
images_path = os.path.join(data_path, 'images/')
stats_path = os.path.join(data_path, 'stats/')

for path in [input_path, output_path, cell_path, expl_path, images_path, stats_path]:
    if not os.path.exists(path):
        os.makedirs(path)

# Load all nb
notebooks = load_data(data_path, input_path, output_path, images_path, stats_path, nb_limit, VA_data=False, save_to_json=False)

# Classify all nb cells
class_count = classify_nb_cells(notebooks, method=classification_method)

# Calculate similarities
nb_output_path, cell_output_path, combi_output_path = calculate_similarities(notebooks, embedding_method, stats_path, all_nb=False, all_cells=True, combi=False)

# Local LLM explanation
explain_nb   = get_explanation_nb(notebooks, model, expl_path, temperature = 0.5, max_retries=5)
explain_cell = get_explanation_cell(notebooks, model, expl_path)
# explain_sim  = get_explanation_sim(notebooks, model)

# Save all nb to json
file_path, filepath_cell = save_notebooks_to_json(notebooks, output_path, cell_path, embedding_method)


print("DONE")