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
        self.classification = None          
        self.class_probability = {}         
        self.detailed_scores = {}      
        self.keywords = {}     

        self.emb = 0                        
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
            'nb_idx': self.nb_idx,
            'embedding': self.embedding,
            'classification': self.classification,
            'keywords': self.keywords,
            'summary': self.summary,
            'q_number': self.q_number,
            'duration': self.duration,
            'exception': self.exception,
            'classification': self.classification,
            'class_probability': self.class_probability,
            'detailed_scores': self.detailed_scores,
            'emb': self.emb
        }
    
    @classmethod
    def from_dict(cls, cell_dict):
        cell = cls(
            cell={'source': cell_dict['source']},
            mc_idx=cell_dict['mc_idx'],
            nb_idx=cell_dict['nb_idx']
        )
        cell.embedding = cell_dict['embedding']
        cell.classification = cell_dict['classification']
        cell.keywords = cell_dict['keywords']
        cell.summary = cell_dict['summary']
        cell.q_number = cell_dict['q_number']
        cell.duration = cell_dict['duration']
        cell.exception = cell_dict['exception']
        return cell
    



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
        output = self.output.to_dict()
        cell_dict.update({
            'cell_type': self.cell_type,
            'image_path': self.image_path,
            'output': output
        })
        return cell_dict

    @classmethod
    def from_dict(cls, cell_dict):
        cell = cls(
            cell={'source': cell_dict['source'], 'cell_type': cell_dict['cell_type']},
            c_idx=None, mc_idx=cell_dict['mc_idx'], nb_idx=cell_dict['nb_idx']
        )
        cell.embedding = cell_dict['embedding']
        cell.classification = cell_dict['classification']
        cell.summary = cell_dict['summary']
        cell.q_number = cell_dict['q_number']
        cell.duration = cell_dict['duration']
        cell.exception = cell_dict['exception']
        cell.image_path = cell_dict['image_path']
        if cell_dict['output']:
            cell.output = OutputCell.from_dict(cell_dict['output'])
        return cell



# inheritance
class MarkCell(Cell):
    def __init__(self, cell, m_idx, mc_idx, nb_idx):
        super().__init__(cell, mc_idx, nb_idx)
        self.cell_type = cell['cell_type']
        
    def __init__(self, cell, m_idx, mc_idx, nb_idx):
        super().__init__(cell, mc_idx, nb_idx)
        self.cell_type = cell['cell_type']

    def to_dict(self):
        cell_dict = super().to_dict()
        cell_dict.update({'cell_type': self.cell_type})
        return cell_dict

    @classmethod
    def from_dict(cls, cell_dict):
        cell = cls(
            cell={'source': cell_dict['source'], 'cell_type': cell_dict['cell_type']},
            m_idx=None, mc_idx=cell_dict['mc_idx'], nb_idx=cell_dict['nb_idx']
        )
        cell.embedding = cell_dict['embedding']
        cell.classification = cell_dict['classification']
        cell.summary = cell_dict['summary']
        cell.q_number = cell_dict['q_number']
        cell.duration = cell_dict['duration']
        cell.exception = cell_dict['exception']
        return cell


# inheritance
class OutputCell(Cell):
    def __init__(self, source, c_idx, o_idx, mc_idx):
        super().__init__(source, mc_idx)
        self.source = source
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



    