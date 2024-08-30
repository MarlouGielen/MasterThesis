import os
import re
import json
import nbformat as nbf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.notebook.notebook import Notebook


def load_notebooks(input_path, limit=999, print_why=False):
    """
    This function loads all notebooks in the input path and returns a list of Notebook objects.
    Remove notebooks with less than 5 code or 1 markdown cells.
    Remove cells that are not code or markdown.
    Remove cells that are empty.

    :param input_path (str): path to the folder containing the notebooks, e.g. 'data/raw/'
    :param print_why (bool): if True, the function will print the reason for skipping a notebook
    :param limit (int): maximum number of notebooks to load

    :return notebooks (list): list of Notebook objects
    """

    notebooks = []

    # Load all notebooks in the input path
    for file in os.listdir(input_path):
        # Skip files that are not a notebook
        if not file.endswith('.ipynb'):
            print('File {} is skipped, as it is not a notebook'.format(file)) if print_why else None
            continue
        else:
            file_path = os.path.join(input_path, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                nb_content = nbf.read(f, as_version=4)  # Read the notebook

                # Skip notebooks with less than 5 code or 1 markdown cells
                n_code_cells = len([cell for cell in nb_content['cells'] if cell['cell_type'] == 'code'])
                n_markdown_cells = len([cell for cell in nb_content['cells'] if cell['cell_type'] == 'markdown'])
                
                if n_code_cells < 5 or n_markdown_cells < 1:
                    print('File {} is skipped, as it has less than 5 code or 1 markdown cells'.format(file)) if print_why else None 
                    continue

                # Remove empty cells and raw cells
                nb_content['cells'] = [cell for cell in nb_content['cells'] if cell['source'].strip() != '' and re.sub(r'[\s\n\t#]', '', cell['source']) != '' and cell['cell_type'] in ['code', 'markdown']]

                # Create a notebook object and add to notebooks list
                notebook = Notebook(len(notebooks), nb_content, file, os.path.join(input_path, file))
                notebooks.append(notebook)
                print(f'[nb_idx {len(notebooks)-1}]. File {file} is processed') if print_why else None
            
            f.close()

        # Break if limit is reached
        if len(notebooks) == limit:
            print(f'Limit of {limit} notebooks reached.')
            break

    return notebooks


def get_stats(notebooks, stats_path):
    """
    This function creates plots for the stats of the notebooks and saves them in the stats_path.

    :param notebooks (list): list of Notebook objects
    :param stats_path (str): path to the folder where the plots will be saved

    :return None, though images are stored in the stats_path
    """
    # check if stats_path exists, if not create it
    if not os.path.exists(stats_path):
        os.makedirs(stats_path)

    # create plots for the stats of the notebooks
    n_code_cells = []
    n_markdown_cells = []
    n_raw_cells = []
    max_lines_per_code_cell = []
    ave_lines_per_code_cell = []
    min_lines_per_code_cell = []
    max_lines_per_markdown_cell = []
    ave_lines_per_markdown_cell = []
    min_lines_per_markdown_cell = []
    nb_idx = []

    for notebook in notebooks:
        n_code_cells.append(notebook.n_code_cells)
        n_markdown_cells.append(notebook.n_markdown_cells)
        n_raw_cells.append(notebook.n_raw_cells)
        max_lines_per_code_cell.append(notebook.max_lines_per_code_cell)
        ave_lines_per_code_cell.append(notebook.ave_lines_per_code_cell)
        min_lines_per_code_cell.append(notebook.min_lines_per_code_cell)
        max_lines_per_markdown_cell.append(notebook.max_lines_per_markdown_cell)
        ave_lines_per_markdown_cell.append(notebook.ave_lines_per_markdown_cell)
        min_lines_per_markdown_cell.append(notebook.min_lines_per_markdown_cell)
        nb_idx.append(notebook.nb_idx)

    # create DataFrame with the stats, to sort notebooks by number of cells
    data = {'nb_idx': nb_idx, 'n_code_cells': n_code_cells, 'n_markdown_cells': n_markdown_cells, 'n_raw_cells': n_raw_cells, 
            'max_lines_per_code_cell': max_lines_per_code_cell, 'ave_lines_per_code_cell': ave_lines_per_code_cell, 'min_lines_per_code_cell': min_lines_per_code_cell, 
            'max_lines_per_markdown_cell': max_lines_per_markdown_cell, 'ave_lines_per_markdown_cell': ave_lines_per_markdown_cell, 'min_lines_per_markdown_cell': min_lines_per_markdown_cell}

    df = pd.DataFrame(data)
    df = df.sort_values(by=['n_code_cells', 'n_markdown_cells'], ascending=[False, False])


    # plot number of each type of cells in bar chart with three bars, x axis the notebooks, y axis the number of cells
    fig, ax = plt.subplots(figsize=(14, 12))
    
    index = np.arange(len(df))
    bar_width = 0.45
    
    ax.barh(index, df['n_code_cells'], bar_width, label='Code cells', color='blue')
    ax.barh(index + bar_width, df['n_markdown_cells'], bar_width, label='Markdown cells', color='lightblue')
    
    ax.set_xlabel('Number of cells')
    ax.set_ylabel('Notebooks')
    ax.set_title('Number of cells in each notebook')
    ax.set_yticks(index + bar_width)
    ax.set_yticklabels(df['nb_idx'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(stats_path + 'n_cells_per_nb.png')

    # plot number of lines per code cell 
    fig, ax = plt.subplots(figsize=(14, 12))

    # sort on lines per code cell   
    df = df.sort_values(by=['ave_lines_per_code_cell', 'max_lines_per_code_cell', 'min_lines_per_code_cell'], ascending=[False, False, False])

    ax.barh(index, df['max_lines_per_code_cell'], bar_width, label='Max lines per code cell', color='lightblue')
    ax.barh(index + bar_width, df['ave_lines_per_code_cell'], bar_width, label='Ave lines per code cell', color='blue')
    ax.barh(index + 2*bar_width, df['min_lines_per_code_cell'], bar_width, label='Min lines per code cell', color='purple')

    ax.set_xlabel('Number of lines')
    ax.set_ylabel('Notebooks')
    ax.set_title('Number of lines per code cell in each notebook')
    ax.set_yticks(index + bar_width)
    ax.set_yticklabels(df['nb_idx'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(stats_path + 'n_lines_per_code_cell.png')

    # plot number of lines per markdown cell
    fig, ax = plt.subplots(figsize=(14, 12))

    # sort on lines per markdown cell   
    df = df.sort_values(by=['ave_lines_per_markdown_cell', 'max_lines_per_markdown_cell', 'min_lines_per_markdown_cell'], ascending=[False, False, False])

    ax.barh(index, df['max_lines_per_markdown_cell'], bar_width, label='Max lines per markdown cell', color='lightblue')
    ax.barh(index + bar_width, df['ave_lines_per_markdown_cell'], bar_width, label='Ave lines per markdown cell', color='blue')
    ax.barh(index + 2*bar_width, df['min_lines_per_markdown_cell'], bar_width, label='Min lines per markdown cell', color='purple')

    ax.set_xlabel('Number of lines')
    ax.set_ylabel('Notebooks')
    ax.set_title('Number of lines per markdown cell in each notebook')
    ax.set_yticks(index + bar_width)
    ax.set_yticklabels(df['nb_idx'])
    ax.legend()

    plt.tight_layout()
    plt.savefig(stats_path + 'n_lines_per_markdown_cell.png')


def add_q_numbers(notebooks, VA_data):
    """
    This function adds the corresponding question numbers to the notebooks, only for VA data.
    The VA data is structured in the following way:
    - ## Question 1     1       (recognized by markdown cell starting with "## 2.1 Question 1" )
    - ## Question 2     2       (recognized by markdown cell starting with "## 2.2 Question 2" )
    - ## Question 3a    3.1     (recognized by markdown cell starting with "intention 3a" )
    - ## Question 3b    3.2     (recognized by markdown cell starting with "intention 3b" )
    - ## Question 3c    3.3     (recognized by markdown cell starting with "intention 3c" )
    - ## Open Questions 4       (recognized by markdown cell starting with "3. Open Questions" )

    :param notebooks (list): list of Notebook objects
    :param VA_data (bool): if True, the function will add question numbers to the notebooks

    :return notebooks (list): list of Notebook objects (with question numbers for VA)
    """
    
    if not VA_data:
        return notebooks
    
    else:
        VA_notebooks = []
        for nb_content in notebooks:
            q_number = 0
            start = False
            new_cells = []

            for ci, cell in enumerate(nb_content.file['cells']):
                if "# 2. Assignment Questions" in cell['source']:
                    start = True
                if not start:
                    continue
                # 1. Add question number label to cells
                if "## 2.1 Question 1" in cell['source']:
                    q_number = 1
                if "## 2.2 Question 2" in cell['source']:
                    q_number = 2
                if "intention 3a" in cell['source']:
                    q_number = 3.1
                if "intention 3b" in cell['source']:
                    q_number = 3.2
                if "intention 3c" in cell['source']:
                    q_number = 3.3
                if "3. Open Questions" in cell['source']:
                    q_number = 4
                cell['q_number'] =  q_number
                
                # 2. store the summary EDA
                if cell['source'].startswith("Summarization of your insights in 3 sentences"):
                    # TODO: check function below
                    nb_content.set_summary_data_VA(cell['source'][len("Summarization of your insights in 3 sentences"):])
                    continue

                # 3. only store students cells (not ## markdown titles)       
                if cell['cell_type'] == 'markdown' and ( cell['source'].strip().startswith("#") or cell['source'].strip().startswith("!") or cell['source'].strip().startswith("#")):
                    continue
                else:
                    # 4. remove prefix (e.g. 'conclusion: ')
                    # TODO: remove <br> and <br/> and '...'
                    pattern = r'^(intention(?: \w+)?(?: \(.*?\))?:|insight(?: \w+)?(?: \(.*?\))?:|conclusion(?: \w+)?(?: \(.*?\))?:)[ \t\n]*'
                    cell['source'] = re.sub(pattern, '', cell['source'], count=1)
                    cell['source'] = cell['source'].replace("<br>", "").replace("<br/>", "").replace("...", "")

                    new_cells.append(cell)

                nb_content.file['cells'] = new_cells
            VA_notebooks.append(nb_content)
        return VA_notebooks


def save_notebooks_to_json(notebooks, nb_path, cell_path, embedding_method):
    """
    This function saves the processed notebooks as json files in the output path.

    :param notebooks (list): list of Notebook objects
    :param nb_path (str): path to the folder where the notebooks will be saved
    :param cell_path (str): path to the folder where the cells will be saved
    :param embedding_method (str): the embedding method used

    :return None, though json files are stored in the nb_path
    """

    for nb in notebooks:
        print(nb.nb_name)
        nb_dict = nb.to_dict()
        nb_path_json = os.path.join(nb_path, nb_dict['nb_name'] + '_' + embedding_method + '_' + nb_dict['filename'] +'.json')
        
        with open(nb_path_json, 'w') as f:
            json.dump(nb_dict, f, indent=4)
        f.close()

        
        # TODO: make cell_to_dict() in Cell Class --- save all_cells to json
        # cell_path_json = os.path.join(cell_path, nb_dict['nb_name'] + '_' + embedding_method + '_' + nb_dict['filename'] +'_all_cells.json')
        # with open(cell_path_json, 'w') as f:
        #     json.dump(nb.all_cells, f, indent=4)
        # f.close()
        cell_path_json = None
    
    
    print(f'Files saved to {nb_path_json} and cell json to {cell_path_json}')
    return nb_path_json, cell_path_json


