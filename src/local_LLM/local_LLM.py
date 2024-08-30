import time
import pandas as pd
import traceback
from openai import OpenAI


def get_explanations(notebooks, model, expl_path, all_nb=True, all_cells=True, temperature = 0.5, max_retries=5):
    """
    Get explanations csv for notebooks and cells.

    :param notebooks (lst): list of notebook objects
    :param model (str): model name
    :param expl_path (str): path to save the explanations
    :param temperature (float): temperature for the model
    :param max_retries (int): number of retries for the model

    :return explain_nb (pd.DataFrame): explanations for notebooks
    :return explain_cell (pd.DataFrame): explanations for cells

    """
    try:
        st = time.time()
        client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
        print('Making a connection takes ', time.time()-st, ' seconds')
        
        explain_nb   = get_explanation_nb(notebooks, model, expl_path, client, temperature = 0.5, max_retries=5) if all_nb else None
        explain_cell = get_explanation_cell(notebooks, model, expl_path, client, temperature = 0.5, max_retries=5) if all_cells else None

        print(f"Local llm time: {time.time()-st:.2f} seconds")
    except FileNotFoundError as fnf_error:
        print("File not found: ", fnf_error)
        explain_nb = None
        explain_cell = None
    except PermissionError as perm_error:
        print("Permission denied: ", perm_error)
        explain_nb = None
        explain_cell = None
    except Exception as e:
        print("Error in explanation: ", e)
        traceback.print_exc()
        explain_nb = None
        explain_cell = None
    
    return explain_nb, explain_cell


def local_LLM(model, instruction, client, content, temperature = 0.5, max_retries=5):
    """
    Local LLM function to get the response from the model.

    :param model (str): model name
    :param instruction (str): instruction for the model
    :param client (OpenAI): OpenAI client
    :param content (str): content to get the response
    :param temperature (float): temperature for the model
    :param max_retries (int): number of retries for the model

    :return response (str): response from the model

    """

    for attempt in range(max_retries):
        try:
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
            time.sleep(20 * (2 ** attempt))         # Exponential backoff
            response = None

    return response


def get_explanation_nb(notebooks, model, data_path, client, temperature = 0.5, max_retries=5):
    """
    Get explanations for notebooks.
    
    :param notebooks (lst): list of notebook objects
    :param model (str): model name
    :param data_path (str): path to save the explanations
    :param client (OpenAI): OpenAI client
    :param temperature (float): temperature for the model
    :param max_retries (int): number of retries for the model

    :return nb_expl (pd.DataFrame): explanations for notebooks
    """
    
   
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

    for nbi, nb in enumerate(notebooks):
        nb_idx = nb.nb_idx
        source = nb.source
        response_summary = local_LLM(model, instruction_sum_nb, client, source, temperature, max_retries)
        response_keywords = local_LLM(model, instruction_sum_nb_words, client, source, temperature, max_retries)
    
        nb_expl.loc[nbi] = [nb_idx, response_summary, response_keywords]
        #nb_expl = nb_expl.append({'nb_idx': nb_idx, 'sum_source': response_summary, 'sum_keywords': response_keywords}, ignore_index=True)  
        # print(f'------------------ {nb.nb_idx} ------------------')
        # print("SUMMARY: ", response_summary, '\n\n')
        # print("KEWYORDS: ", response_keywords, '\n\n')
        # print("SOURCE: ", nb.source, '\n\n\n\n\n\n')

    # save to csv
    full_path = data_path + 'nb_expl.csv'
    nb_expl.to_csv(full_path, index=False)

    return nb_expl


# markdown cell, code cell


def get_explanation_cell(notebooks, model, data_path, client, temperature = 0.5, max_retries=5):
    """
    Get explanations for cells.
    
    :param notebooks (lst): list of notebook objects
    :param model (str): model name
    :param data_path (str): path to save the explanations
    :param client (OpenAI): OpenAI client
    :param temperature (float): temperature for the model
    :param max_retries (int): number of retries for the model

    :return nb_expl (pd.DataFrame): explanations for cells    
    """
    
    nb_expl = pd.DataFrame(columns=['nb_idx', 'mc_idx', 'sum_cell', 'sum_keywords', 'classification', 'keywords_classification'])
    instruction_sum_code_cell = "Give a summary of max 3 sentences, about the provided Python code snippet. Please summarize the content of the cell. Summarize what happens in the cell, what is the semantic meaning, the intention. Answer with a summary of max 3 sentences, no additional text. "
    instruction_sum_markdown_cell = "Give a summary of 1 sentence, about the provided markdown cell. Please summarize the content of the cell. Summarize what happens in the cell, what is the semantic meaning, the intention. Answer with a summary of max 3 sentences, no additional text. "
    
    instruction_code_cell_words = "Explain with a list of keywords, what the Python code snippet is mainly about. Reply with the list of keywords separated by commas. The keywords should characterize the cell's content and be specific for this cell, such as the types of models used, the types of machine learning operations used, the types of evaluation metrics or hyperparametertuning. Answer with comma separated keywords only. You only answer with the keywords separated by commas, no additional text."
    instruction_markdown_cell_words = "Explain with a list of keywords, what the markdown cell is mainly about. Reply with the list of keywords separated by commas. The keywords should characterize the cell's content and be specific for this cell, such as the types of models used, the types of machine learning operations used, the types of evaluation metrics or hyperparametertuning. Answer with comma separated keywords only. You only answer with the keywords separated by commas, no additional text."

    instruction_code_cell_classification = "Classify the code cell. Reply with the classification of the code cell. The classification should be one of the following: 'Environment', 'Data_Extraction', 'Exploratory_Data_Analysis', 'Data_Transform', 'Model_Train', 'Model_Evaluation', 'Model_Interpretation', 'Hyperparameter_Tuning', 'Visualization', 'Debug', 'Data_Export','Other'. Answer with the classification only, no additional text. Only answer with the classificaiton (e.g. 'Exploratory_Data_Analysis'). Answer with the classification only, no additional text."
    instruction_markdown_cell_classification = "Classify the markdown cell. Reply with the classification of the markdown cell. The classification should be one of the following: 'Description_next_cell', 'Result_previous_cell', 'Headline', 'Summary', 'Image', 'Other'. Answer with the classification only, no additional text. Only answer with the classificaiton (e.g. 'Description_next_cell'). Answer with the classification only, no additional text."  
    
    instruction_code_cell_keywords_classification = "Classify the code cell with keywords. Reply with the keywords for the chosen classification, which is one of the following: 'Environment', 'Data_Extraction', 'Exploratory_Data_Analysis', 'Data_Transform', 'Model_Train', 'Model_Evaluation', 'Model_Interpretation', 'Hyperparameter_Tuning', 'Visualization', 'Debug', 'Data_Export','Other'. Only answer with the keywords, in the form of a list ['','','']. Answer with the keywords only, no additional text."
    instruction_markdown_cell_keywords_classification = "Classify the markdown cell with keywords. Reply with the keywords for the chosen classification of the markdown cell, which is one of the following. 'Description_next_cell', 'Result_previous_cell', 'Headline', 'Summary', 'Image', 'Other'. Only answer with the keywords, in the form of a list ['','','']. Answer with the keywords only, no additional text."
    
    len_nbs = 0
    for nb in notebooks:
        nb_len = len(nb.all_cells)
        indeces = [nb_len/4, nb_len/2, 3*(nb_len/4), nb_len-4]

        for cell in nb.all_cells:
    
            content = cell.source
            cell_type = cell.cell_type

            if cell_type == 'code':
                response_summary = local_LLM(model, instruction_sum_code_cell, client, content, temperature, max_retries)
                response_keywords = local_LLM(model, instruction_code_cell_words, client, content, temperature, max_retries)
                response_classification = local_LLM(model, instruction_code_cell_classification, client, content, temperature, max_retries)
                response_keywords_classification = local_LLM(model, instruction_code_cell_keywords_classification, client, content, temperature, max_retries)
            else:
                response_summary = local_LLM(model, instruction_sum_markdown_cell, client, content, temperature, max_retries)
                response_keywords = local_LLM(model, instruction_markdown_cell_words, client, content, temperature, max_retries)
                response_classification = local_LLM(model, instruction_markdown_cell_classification, client, content, temperature, max_retries)
                response_keywords_classification = local_LLM(model, instruction_markdown_cell_keywords_classification, client, content, temperature, max_retries)

            nb_expl = nb_expl.append({'nb_idx': nb.nb_idx, 'mc_idx': cell.mc_idx, 'sum_cell': response_summary, 'sum_keywords': response_keywords, 'classification': response_classification, 'keywords_classification': response_keywords_classification}, ignore_index=True)
        
        # print(f'------------------ {nb.nb_idx} ------------------')
        # for idx in indeces:
        #     print(f'source {cell_type} : {content}')

        len_nbs += nb_len
        
    # save to csv
    full_path = data_path + 'cell_expl.csv'
    nb_expl.to_csv(full_path, index=False)     

    return nb_expl