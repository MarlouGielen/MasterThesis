import os 
import json

class Notebook:
    """
    A class to represent a Jupyter Notebook file.

    Attributes
    ----------
    nb_idx : int    
        index of the notebook in the list of notebooks
    nb_name : str
        name of the notebook
    file : dict
        dictionary containing the notebook file
    filename : str
        name of the notebook file
    filepath : str
        path to the notebook file
        
    source : str
        full source code of the notebook file
    code_source : str
        source code of the code cells in the notebook file
    markdown_source : str
        source code of the markdown cells in the notebook file

    n_cells : int
        number of cells in the notebook file
    n_code_cells : int
        number of code cells in the notebook file
    n_markdown_cells : int
        number of markdown cells in the notebook file
    n_raw_cells : int
        number of raw cells in the notebook file
    n_outputs : int
        number of code cells with outputs in the notebook file

    r_code_cells : float
        ratio of code cells to total cells in the notebook file
    r_markdown_cells : float
        ratio of markdown cells to total cells in the notebook file
    r_raw_cells : float
        ratio of raw cells to total cells in the notebook file
    r_outputs : float
        ratio of code cells with outputs to code cells in the notebook file

    n_exceptions : int
        number of cells with exceptions in the notebook file
    r_exceptions : float
        ratio of cells with exceptions to total cells in the notebook file

    total_duration : float
        total execution time of the notebook file

    """
    def __init__(self, nb_idx, file, filename, file_path):

        self.file = file
        self.filepath = file_path 
        self.filename = filename

        self.nb_idx = nb_idx #TODO: define nb_idx
        self.nb_name = 'd' + str(nb_idx).zfill(4)  

        # get source code
        self.source = ' \n '.join([cell['source'] for cell in file['cells']])
        self.code_source = ' \n '.join([cell['source'] for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.markdown_source = ' \n '.join([cell['source'] for cell in file['cells'] if cell['cell_type'] == 'markdown'])

        # count cells
        self.n_cells = len(file['cells'])
        self.n_code_cells = len([cell for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.n_markdown_cells = len([cell for cell in file['cells'] if cell['cell_type'] == 'markdown'])
        self.n_raw_cells = len([cell for cell in file['cells'] if cell['cell_type'] == 'raw'])
        self.n_outputs = len([cell for cell in file['cells'] if 'outputs' in cell])

        # calculate ratios cell occurence
        self.r_code_cells = self.n_code_cells / self.n_cells
        self.r_markdown_cells = self.n_markdown_cells / self.n_cells
        self.r_raw_cells = self.n_raw_cells / self.n_cells
        self.r_outputs = self.n_outputs / self.n_code_cells # ratio of code cells with outputs to code cells

        # count how often ['metadata']['papermill']['exception'] == True (e.g. an error occurred during execution)
        exceptions_true = 0
        exceptions_false = 0
        for cell in file['cells']:
            if 'papermill' in cell:
                if 'exception' in cell['metadata']['papermill']:
                    if cell['metadata']['papermill']['exception'] == True:
                        exceptions_true += 1
                    else:
                        exceptions_false += 1
        self.n_exceptions = exceptions_true
        self.r_exceptions = exceptions_true / self.n_cells

        # total execution time
        #self.total_duration = sum([cell['metadata']['papermill']['duration'] for cell in file['cells']])

        # count lines
        self.n_lines = sum([len(cell['source'].split('\n')) for cell in file['cells']])
        self.n_lines_code = sum([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.n_lines_markdown = sum([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'markdown'])
        self.lines_per_cell = [len(cell['source'].split('\n')) for cell in file['cells']]
        self.lines_per_code_cell = [len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'code']
        self.lines_per_markdown_cell = [len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'markdown']
        
        self.ave_lines_per_cell = self.n_lines / self.n_cells
        self.ave_lines_per_code_cell = sum([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'code']) / self.n_code_cells
        self.ave_lines_per_markdown_cell = sum([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'markdown']) / self.n_markdown_cells
        self.max_lines_per_cell = max([len(cell['source'].split('\n')) for cell in file['cells']])
        self.max_lines_per_code_cell = max([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.max_lines_per_markdown_cell = max([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'markdown'])
        self.min_lines_per_cell = min([len(cell['source'].split('\n')) for cell in file['cells']])
        self.min_lines_per_code_cell = min([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.min_lines_per_markdown_cell = min([len(cell['source'].split('\n')) for cell in file['cells'] if cell['cell_type'] == 'markdown'])
        
        # count characters
        self.n_chars = sum([len(cell['source']) for cell in file['cells']])
        self.n_chars_code = sum([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.n_chars_markdown = sum([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'markdown'])
        self.chars_per_cell = [len(cell['source']) for cell in file['cells']]
        self.chars_per_code_cell = [len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'code']
        self.chars_per_markdown_cell = [len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'markdown']

        self.ave_chars_per_line = sum([len(cell['source']) for cell in file['cells']]) / self.n_lines    
        
        self.ave_chars_per_cell = self.n_chars / self.n_cells
        self.ave_chars_per_code_cell = sum([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'code']) / self.n_code_cells
        self.ave_chars_per_markdown_cell = sum([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'markdown']) / self.n_markdown_cells
        self.max_chars_per_cell = max([len(cell['source']) for cell in file['cells']])
        self.max_chars_per_code_cell = max([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.max_chars_per_markdownell = max([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'markdown'])
        self.min_chars_per_cell = min([len(cell['source']) for cell in file['cells']])
        self.min_chars_per_code_cell = min([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'code'])
        self.min_chars_per_markdown_cell = min([len(cell['source']) for cell in file['cells'] if cell['cell_type'] == 'markdown'])

        # calculate ratios line and character occurence
        self.r_lines_code = self.n_lines_code / self.n_lines
        self.r_lines_markdown = self.n_lines_markdown / self.n_lines

        self.r_chars_markdown = self.n_chars_markdown / self.n_chars
        self.r_chars_code = self.n_chars_code / self.n_chars

        self.summary_data_VA = None

        self.sim_matrix = []
        self.cell_sim_matrix = []
        self.nb_order = []
   
    def set_summary_data_VA(self, summary_data_VA):
        self.summary_data_VA = summary_data_VA
    

    def get_cells_of_type(self, cell_type):
        """
        Get all cells of a specific type.

        :param cell_type (str): type of the cell
        :return cells (lst): list of cells of the specified type
        """

        if cell_type == 'code':
            cells = self.code_cells
        elif cell_type == 'markdown':
            cells = self.markdown_cells
        elif cell_type == 'all':
            cells = self.all_cells
        else:
            print("Unknown cell type specified, try 'code', 'markdown', or 'all' ")

        return cells
    

    def to_dict(self):
        return {
            'nb_idx': self.nb_idx,
            'nb_name': self.nb_name,
            #'file': self.file,
            'filename': self.filename,
            'filepath': self.filepath,
            'source': self.source,
            'code_source': self.code_source,
            'markdown_source': self.markdown_source,
            'n_cells': self.n_cells,
            'n_code_cells': self.n_code_cells,
            'n_markdown_cells': self.n_markdown_cells,
            'n_raw_cells': self.n_raw_cells,
            'n_outputs': self.n_outputs,
            'r_code_cells': self.r_code_cells,
            'r_markdown_cells': self.r_markdown_cells,
            'r_raw_cells': self.r_raw_cells,
            'r_outputs': self.r_outputs,
            'n_exceptions': self.n_exceptions,
            'r_exceptions': self.r_exceptions,
            'n_lines': self.n_lines,
            'n_lines_code': self.n_lines_code,
            'n_lines_markdown': self.n_lines_markdown,
            'lines_per_cell': self.lines_per_cell,
            'lines_per_code_cell': self.lines_per_code_cell,
            'lines_per_markdown_cell': self.lines_per_markdown_cell,
            'ave_lines_per_cell': self.ave_lines_per_cell,
            'ave_lines_per_code_cell': self.ave_lines_per_code_cell,
            'ave_lines_per_markdown_cell': self.ave_lines_per_markdown_cell,
            'max_lines_per_cell': self.max_lines_per_cell,
            'max_lines_per_code_cell': self.max_lines_per_code_cell,
            'max_lines_per_markdown_cell': self.max_lines_per_markdown_cell,
            'min_lines_per_cell': self.min_lines_per_cell,
            'min_lines_per_code_cell': self.min_lines_per_code_cell,
            'min_lines_per_markdown_cell': self.min_lines_per_markdown_cell,
            'n_chars': self.n_chars,
            'n_chars_code': self.n_chars_code,
            'n_chars_markdown': self.n_chars_markdown,
            'chars_per_cell': self.chars_per_cell,
            'chars_per_code_cell': self.chars_per_code_cell,
            'chars_per_markdown_cell': self.chars_per_markdown_cell,
            'ave_chars_per_line': self.ave_chars_per_line,
            'ave_chars_per_cell': self.ave_chars_per_cell,
            'ave_chars_per_code_cell': self.ave_chars_per_code_cell,
            'ave_chars_per_markdown_cell': self.ave_chars_per_markdown_cell,
            'max_chars_per_cell': self.max_chars_per_cell,
            'max_chars_per_code_cell': self.max_chars_per_code_cell,
            'max_chars_per_markdownell': self.max_chars_per_markdownell,
            'min_chars_per_cell': self.min_chars_per_cell,
            'min_chars_per_code_cell': self.min_chars_per_code_cell,
            'min_chars_per_markdown_cell': self.min_chars_per_markdown_cell,
            'r_lines_code': self.r_lines_code,
            'r_lines_markdown': self.r_lines_markdown,
            'r_chars_markdown': self.r_chars_markdown,
            'r_chars_code': self.r_chars_code,
            'all_cells': [cell.to_dict() for cell in self.all_cells],
            'code_cells': [cell.to_dict() for cell in self.code_cells],
            'markdown_cells': [cell.to_dict() for cell in self.markdown_cells],
            'sim_matrix': self.sim_matrix,
            'cell_sim_matrix': self.cell_sim_matrix,
            'nb_order': self.nb_order,
            'summary_data_VA': self.summary_data_VA if hasattr(self, 'summary_data_VA') else None,
        }
    


    def save_notebook(self, output_path):
        """
        Save the notebook as a json file.

        :param output_path (str): path to save the notebook
        """

        with open(os.path.join(output_path, self.filename), 'w') as f:
            json.dump(self.file, f)

    
