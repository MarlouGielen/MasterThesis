import os
import pickle
import numpy as np
from torch.nn import functional as F
#from concurrent.futures import ThreadPoolExecutor

from src.embedding import get_embedding, load_model_tokenizer


def calculate_similarities(notebooks, embedding_method, heatmap_path, all_nb=True, all_cells=True, combi=True):

    # Load model and tokenizer
    tokenizer, model = load_model_tokenizer(embedding_method)
    #print(f"Model and tokenizer loaded for {embedding_method}")

    # Calculate similarities for all nbs
    if all_nb:
        # Get similarity matrix for notebooks
        sim_matrices, new_order = get_sim_matrix_per_nb(notebooks, tokenizer, model, embedding_method)
        nb_output_path = None #plot_heatmap(sim_matrices, new_order, heatmap_path + embedding_method + "_nb_heatmap.png")
        save_similarities(sim_matrices, heatmap_path + embedding_method + "_nb_similarities.pkl")
        #print("Notebook similarities obtained for", len(sim_matrices), "notebooks.")
    else:
        sim_matrices = None
        nb_output_path = None

    # Calculate similarities for all cells
    if all_cells:
        cell_similarities = get_sim_matrix_per_cell(notebooks, tokenizer, model, embedding_method)
        save_similarities(cell_similarities, heatmap_path + embedding_method + "_cell_similarities.pkl")
        cell_output_path = None
        # # heatmap per nb - nb combination
        # for nb_idx, nb1 in enumerate(cell_similarities):
        #     nb_cell_sim = np.array(nb1)  # Convert to numpy array for easier manipulation
            
        #     for cell_idx, cell_sim_matrix in enumerate(nb_cell_sim):
        #         cell_sim_matrix_2d = np.array(cell_sim_matrix)
        #         if cell_sim_matrix_2d.ndim != 2:
        #             raise ValueError(f"Expected 2D array, got {cell_sim_matrix_2d.ndim}D array")
                
        #     plot_heatmap(nb_cell_sim, None, heatmap_path + "_cellmap_" + embedding_method + "_nb" + str(nb_idx) + "_cell" + str(cell_idx) + "_heatmap.png")
        #     print("end of nb", nb_idx)
        
        #print("Cell similarities obtained for", len(cell_similarities), "notebooks.")
    else:
        cell_similarities = None
        cell_output_path = None

    # if combi:
    #     combi_similarities = get_sim_matrix_per_cell_combi(notebooks, tokenizer, model, embedding_method, cell_type='code')
    #     combi_output_path = save_similarities(cell_similarities, "combi_" + heatmap_path)
    #     print("Cell similarities obtained for code combi ", len(cell_similarities), "notebooks.")
    # else:
    #     combi_similarities = None
    #     combi_output_path = None

    return nb_output_path, cell_output_path, None #TODO: combi_output_path


# def plot_heatmap(similarities, order, output_path):
#     """
#     Plot the heatmap of similarities.

#     :param similarities (list): the list of similarities
#     :param order (list): the order of the notebooks
#     :param output_path (str): the output path to save the plot

#     :return path (str): the path to the saved plot
#     """

#     # Plot heatmap
#     path = plot_heatmap(similarities, order, output_path)
#     print("Heatmap plotted and saved to", path)

#     return path


def save_similarities(similarities, output_path):
    """
    Save the similarities to a file.

    :param similarities (list): the list of similarities
    :param output_path (str): the output path to save the similarities
    """

    with open(output_path, 'wb') as f:
        pickle.dump(similarities, f)

    print("Similarities saved to", output_path)

    return output_path


def get_sim_matrix_per_nb(notebooks, tokenizer, model, embedding_method):
    """
    Get the similarity matrix for each notebook in the list.

    :param notebooks (list): the list of notebooks
    :param embedding_method (str): the embedding method to use

    :return sim_matrices (list): the list of similarity matrices
    """
    sim_matrices = []

    # Get embeddings for all notebooks
    for nb1 in notebooks:
        print(f"sim nb {nb1.nb_idx}/{len(notebooks)} done ")
        emb_nb1 = get_embedding(nb1, tokenizer, model, embedding_method)
        sim_nb = []
        for nb2 in notebooks:
            emb_nb2 = get_embedding(nb2, tokenizer, model, embedding_method)
            sim_nb.append(F.cosine_similarity(emb_nb1, emb_nb2).item())
            del emb_nb2
        del emb_nb1
        sim_matrices.append(sim_nb)
        nb1.sim_matrix = sim_nb

    # determine new notebook order with most similar next to each other
    order = [0] # start with first notebook
    remaining = list(range(1, len(notebooks))) # remaining notebooks
    while remaining:
        last = order[-1]
        max_sim = -1
        max_idx = -1
        for idx in remaining:
            sim = sim_matrices[last][idx]
            if sim > max_sim:
                max_sim = sim
                max_idx = idx
        order.append(max_idx)
        remaining.remove(max_idx)

    nb1.nb_order = order
    #print("New notebook order:", order)

    return sim_matrices, order


def get_sim_matrix_per_cell(notebooks, tokenizer, model, embedding_method, cell_type='code'):
    """
    Get the similarity matrix for each cell in the notebooks.

    :param notebooks (list): the list of notebooks
    :param embedding_method (str): the embedding method to use
    :param cell_type (str): the type of cell to get embeddings for

    :return sim_matrices (list): the list of similarity matrices
    """

    #TODO: max_cell remove
    max_cell = 100000
    def compute_embedding(cell):
        return get_embedding(cell, tokenizer, model, embedding_method)

    sim_matrices = []                   # all nb1, all cells1, all nb2, all cells2
    for nb1 in notebooks:
        print(f"sim cells {nb1.nb_idx}/{len(notebooks)} done ")
        cells_nb1 = nb1.all_cells

        # Compute embeddings for cells in nb1
        # embeddings_nb1 = []
        # with ThreadPoolExecutor() as executor:
        #     embeddings_nb1 = list(executor.map(compute_embedding, cells_nb1))

        sim_cell_nb_cell = []           # all cells1, all nb2, all cells2       (do for each          emb of cell1)
        for c1, cell1 in enumerate(cells_nb1):
            emb1 = get_embedding(cell1, tokenizer, model, embedding_method)
            
            sim_nb_cell = []            # all nb2, all cells2                   (compare all nb2   to emb of cell1)
            for nb2 in notebooks:
                cells_nb2 = nb2.all_cells

                sim_cell = []           # all cells2                            (compare all cell2 to emb of cell1)
                for c2, cell2 in enumerate(cells_nb2):
                    emb2 = get_embedding(cell2, tokenizer, model, embedding_method)
                    sim_cell.append(F.cosine_similarity(emb1, emb2).item())
                    del emb2

                    if c2 == max_cell:
                        break
                
                sim_nb_cell.append(sim_cell)

                # Compute embeddings for cells in nb2
                # embeddings_nb2 = []
                # with ThreadPoolExecutor() as executor:
                #     embeddings_nb2 = list(executor.map(compute_embedding, cells_nb2))

                # Compute similarities and free up memory for embeddings_nb2
                # sim_cell = [F.cosine_similarity(emb1, emb2).item() for emb2 in embeddings_nb2]
                # sim_nb_cell.append(sim_cell)
                
                # del embeddings_nb2

            if c1 == max_cell:
                break
            
            sim_cell_nb_cell.append(sim_nb_cell)

        sim_matrices.append(sim_cell_nb_cell)

        nb1.cell_sim_matrix = sim_cell_nb_cell

        # # Clear embeddings_nb1 to free memory
        # del embeddings_nb1

    return sim_matrices


def plot_heatmap(similarities, new_order, heatmap_path):
    """
    Plot the heatmap of similarities.

    :param similarities (list): the list of similarities
    :param new_order (list): the list of new indices
    :param heatmap_path (str): the output path to save the plot
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    import numpy as np
    import mplcursors

    
    similarities = np.array(similarities)

    if new_order is not None:
        # re-order similarites based on new_order list of new indices and name labels
        similarities_order = similarities[new_order, :]         # reorder rows based on new_order
        similarities_order = similarities_order[:, new_order]   # reorder columns based on inversed new_order

        labels_x = new_order
        labels_y = new_order

    else:
        similarities_order = similarities
        labels_x = np.arange(len(similarities))
        labels_y = np.arange(len(similarities))


    lensim = len(similarities) / 2
    fig, ax = plt.subplots(figsize=(lensim, lensim))
    sns.heatmap(similarities_order, annot=False, ax=ax, cmap="YlGnBu", cbar_kws={"label": "Similarity"})

    # Set the tick labels
    ax.set_xticks(np.arange(len(labels_x)) + 0.5)
    ax.set_yticks(np.arange(len(labels_y)) + 0.5)
    ax.set_xticklabels(labels_x, rotation=90, ha='center')
    ax.set_yticklabels(labels_y, rotation=0)

    # Interactive cursor for displaying similarity values
    # cursor = mplcursors.cursor(ax, hover=True)
    # @cursor.connect("add")
    # def on_add(sel):
    #     x, y = sel.target
    #     sel.annotation.set_text(f"{similarities_order[int(y), int(x)]:.2f}")

    # Save the plot
    fig_path = os.path.join(heatmap_path)
    plt.savefig(fig_path)
    plt.show()

    # return output_path
    return heatmap_path

