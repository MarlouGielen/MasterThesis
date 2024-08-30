import os
import time

from openai import OpenAI

from src.data_processing import load_data, save_notebooks_to_json
from src.code_classification.code_classification import classify_nb_cells
from src.similarity.calculate_similarities import calculate_similarities
from src.notebook.simple_notebook import SimpleNotebook
from src.cell.simple_cell import SimpleCell
from src.local_LLM.local_LLM import get_explanations


# Configuration       
d_name = "Kaggle"                                            # Kaggle, GitHub, VA
data_path = 'data/data_'+ d_name + '/'
emb_method = "DistilBERT"
classification_method = "all"
model="TheBloke/Mistral-7B-Instruct-v0.1-GGUF"
nb_limit=1000

# Paths
input_path =    os.path.join(data_path, 'raw/')
output_path =   os.path.join(data_path, 'outp/') 

nb_path =       os.path.join(output_path,   'notebooks/')
cell_path =     os.path.join(output_path,   'cells/')
expl_path =     os.path.join(output_path,   'expl/')
images_path =   os.path.join(output_path,   'images/')
stats_path =    os.path.join(output_path,   'stats/')

for path in [input_path, output_path, cell_path, expl_path, images_path, stats_path]:
    if not os.path.exists(path):
        os.makedirs(path)

t0 = time.time()   


# ----- Load data
notebooks = load_data(data_path, input_path, output_path, images_path, stats_path, nb_limit, VA_data=False, save_to_json=False)

# ----- Classify cells
class_count = classify_nb_cells(notebooks, method=classification_method)
                                                                                          
# ----- Calculate similarities
nb_output_path, cell_output_path, combi_output_path = calculate_similarities(notebooks, emb_method, output_path, all_nb=True, all_cells=True, combi=False)

# ----- Save notebooks to json
save_notebooks_to_json(notebooks, output_path, cell_path, emb_method)
print(f"Total time to save json files of notebooks and calculate similarities: {time.time()-t0:.2f} seconds")

# ----- Get explanations
nb_expl_output_path, cell_expl_output_path = get_explanations(notebooks, model, expl_path, all_nb=True, all_cells=True, temperature = 0.5, max_retries=5)

print("DONE")

# test 30 nb - 100 cells -> 3000 cells no       (3000*3000 = 9M sim)        time 
# test 25 nb - 50 cells -> 1250 cells no        (1250*1250 = 1.5M sim)      time    
# test 20 nb - 50 cells -> 1000 cells yes       (1000*1000 = 1M sim)        time 