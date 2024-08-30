import time

from .nb_processing import load_notebooks, get_stats, add_q_numbers, save_notebooks_to_json
from .cell_processing import load_cell_obj


def load_data(data_path, input_path, output_path, images_path, stats_path, limit, VA_data, save_to_json=False):
    """
    A function to load the data and process it.

    :param data_path (str): path to data folder
    :param input_path (str): path to input data
    :param output_path (str): path to save output data
    :param images_path (str): path to save images
    :param stats_path (str): path to save stats
    :param VA_data (bool): True if data is from VA, False otherwise
    """
    
    t0 = time.time()

    # Load notebooks
    notebooks = load_notebooks(input_path, limit, print_why=False)  # load notebooks
    print(f"Loaded {len(notebooks)} notebooks in {time.time()-t0:.2f} seconds")
    notebooks = add_q_numbers(notebooks, VA_data)                   # add question numbers to cells, for VA data only
    #get_stats(notebooks, stats_path)                                # plot images in stats folder

    t1 = time.time()

    # Load cells, per notebook
    notebooks = load_cell_obj(notebooks, images_path)
    print(f"Loaded cells in {time.time()-t1:.2f} seconds")

    print(f"Total data load time: {time.time()-t0:.2f} seconds")

    return notebooks