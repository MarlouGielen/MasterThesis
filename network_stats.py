import pickle
import json
import os

# load all nb json files in processed folder
nb_json_folder = 'data/data_Kaggle/processed/'
notebooks = []
for file in os.listdir(nb_json_folder):
    if file.endswith(".json"): 
        with open(os.path.join(nb_json_folder, file), 'r') as f:
            nb_json = json.load(f)
            notebooks.append(nb_json)
print("---------------------------------------------------------------------")


# load nb similarities
nb_sim = 'data/data_Kaggle/stats/DistilBERT_nb_similarities.pkl'  
with open(nb_sim, 'rb') as f:
    nb_sim_matrix = pickle.load(f)
print("NB sim matrix:", len(nb_sim_matrix))
for sim in nb_sim_matrix:
    print(sim)
print("---------------------------------------------------------------------")

# load cell similarities
cell_sim = 'data/data_Kaggle/stats/DistilBERT_cell_similarities.pkl'
with open(cell_sim, 'rb') as f:
    cell_sim_matrix = pickle.load(f)

nb1_ = 0
nb2_ = 1
c1_ =  2
c2_ = 0

print(cell_sim_matrix[nb1_][nb2_][c1_][c2_])

print("---------------------------------------------------------------------")

# load combi similarities	

print("---------------------------------------------------------------------")

# load expl csv

print("---------------------------------------------------------------------")

# ... load sim to prev 5, sim to next 5

# ... load sim code cells

# ... expl code cells only

