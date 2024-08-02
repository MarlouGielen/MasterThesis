# superclass
class Cell:
    """
    A class to represent a cell in a Jupyter notebook.

    Attributes
    ----------
    source : str
        The source code or markdown content of the cell.
    mc_idx : int
        The index of the notebook to which the cell belongs.
    embedding : dict
        The embedding of the cell content.
    classification : dict
        The classification of the cell content.
    keywords : dict
        The keywords extracted from the cell content.
    q_number : int
        The question number of the cell.
    duration : float
        The duration of cell execution.
    exception : bool
        True if an error occurred during cell execution.
    """
    
    def __init__(self, cell, mc_idx, nb_idx=0):
        self.source = cell['source']
        self.mc_idx = mc_idx
        self.nb_idx = nb_idx

        self.embedding = {}

        self.classification = {}
        self.keywords = {}

        self.summary = None

        # conditional attributes
        if 'q_number' in cell:
            self.q_number = cell['q_number']
        else:
            self.q_number = None
        if 'metadata' in cell:
            if 'papermill' in cell['metadata']:
                if 'duration' in cell['metadata']['papermill']:
                    self.duration = cell['metadata']['papermill']['duration']       #TODO: analyze execution time trends in notebooks
                if 'exception' in cell['metadata']['papermill']:
                    self.exception = cell['metadata']['papermill']['exception']     # if True an error occurred during execution
        else:
            self.duration = None
            self.exception = None

    def set_embedding(self, emb_type, embedding):
        self.embedding[emb_type] = embedding

    def set_classification(self, class_type, classification):
        self.classification[class_type] = classification
    
    def set_classification_keywords(self, class_type, classification, keywords):
        self.classification[class_type] = classification
        self.keywords[class_type] = keywords

    def set_summary(self, sum_type, summary):
        self.summary = summary

    def to_dict(self):
        return {
            'source': self.source,
            'mc_idx': self.mc_idx,
            'embedding': self.embedding,
            'classification': self.classification,
            'keywords': self.keywords,
            'summary': self.summary,
            'q_number': getattr(self, 'q_number', None),
            'duration': getattr(self, 'duration', None),
            'exception': getattr(self, 'exception', None)
        }
    
# inheritance
class CodeCell(Cell):
    def __init__(self, cell, c_idx, mc_idx, nb_idx):
        super().__init__(cell, mc_idx, nb_idx)
        self.cell_type = cell['cell_type']
        #self.outputs = cell['outputs']
    
    def set_image_path(self, image_path, c_idx, o_idx, img_idx):
        if hasattr(self, 'image_path'):
            self.image_path.append((image_path, c_idx, o_idx, img_idx))
        self.image_path = [(image_path, c_idx, o_idx, img_idx)]

    def set_output(self, output_source, c_idx, o_idx, mc_idx):
        output = {'source': output_source}
        self.output = OutputCell(output, c_idx, o_idx, mc_idx)

    def to_dict(self):
        cell_dict = super().to_dict()
        output = getattr(self, 'output', None)
        cell_dict.update({
                'cell_type': self.cell_type,
                'image_path': getattr(self, 'image_path', []),
                'output': output.to_dict() if output else None
            })
        return cell_dict


# inheritance
class MarkCell(Cell):
    def __init__(self, cell, m_idx, mc_idx, nb_idx):
        super().__init__(cell, mc_idx, nb_idx)
        self.cell_type = cell['cell_type']

    def to_dict(self):
        cell_dict = super().to_dict()
        cell_dict.update({
            'cell_type': self.cell_type
        })
        return cell_dict


# inheritance
class OutputCell(Cell):
    def __init__(self, cell, c_idx, o_idx, mc_idx):
        super().__init__(cell, mc_idx)
        self.cell_type = 'output'
        self.mc_idx = mc_idx
        self.c_idx = c_idx
        self.o_idx = o_idx

    def to_dict(self):
        cell_dict = super().to_dict()
        cell_dict.update({
            'cell_type': self.cell_type,
            'c_idx': self.c_idx,
            'o_idx': self.o_idx
        })
        return cell_dict



    