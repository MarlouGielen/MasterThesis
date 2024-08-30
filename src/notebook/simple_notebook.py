import os 
import json

class SimpleNotebook:
    def __init__(self, nb_idx, source, all_cells):
        self.nb_idx = nb_idx
        self.source = source
        self.all_cells = all_cells

        self.sim_matrix = []
        self.cell_sim_matrix = []
        self.nb_order = []
