import os

from src.utils.helper_functions import save_image_from_data
from src.cell.cell import CodeCell, MarkCell


def load_cell_obj(notebooks, images_path): #TODO: add consecutive code and markdown options ?
    """
    A function to load the cell objects of the notebooks.

    :param notebooks (list): list of notebook objects
    :param images_path (str): path to save images

    :return notebooks_cell_obj (list): list of notebook objects with cell objects
    """
    
    notebooks_cell_obj = []

    # create images folder
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    # iterate over notebooks
    for nb_idx, nb in enumerate(notebooks):
        all_cells = []
        all_code_cells = []
        all_markdown_cells = []
        mc_idx, m_idx, c_idx = 0, 0, 0
        img_idx = 0


        # iterate over cells
        i = 0        
        while i < len(nb.file['cells']):
            cell = nb.file['cells'][i]

            # handle code cell
            if cell['cell_type'] == 'code':
                codecell = CodeCell(cell, c_idx, mc_idx, nb_idx)
                
                # process outputs of code cell
                if 'outputs' in cell and len(cell['outputs']) > 0:

                    output_source = []
                    for o_idx, output in enumerate(cell['outputs']):
                        codecell.set_image_path(None, c_idx, o_idx, None) # set image path to None as default
                        if 'text' in output:
                            output_source.append(output['text'])
                        elif 'data' in output:
                            output_data = output['data']
                            # save image
                            if "image/png" in output_data:
                                image_path = save_image_from_data(output_data['image/png'], images_path, img_idx, nb.nb_name + '_c' + str(c_idx).zfill(3) + '_o' + str(o_idx).zfill(3) )
                                codecell.set_image_path(image_path, c_idx, o_idx, img_idx)
                                img_idx += 1
                                
                            # add plain text or html text
                            if 'text/plain' in output_data:
                                output_source.append(output_data['text/plain'])
                            elif 'text/html' in output_data:
                                output_source.append(output_data['text/html'])

                    codecell.set_output(output_source, c_idx, o_idx, mc_idx)
                else:
                    codecell.set_output([], c_idx, 0, mc_idx)
                    codecell.set_image_path(None, c_idx, 0, None)

                all_cells.append(codecell)
                all_code_cells.append(all_cells[-1])

                c_idx += 1
                mc_idx += 1
                i += 1

                if mc_idx != i:
                    raise ValueError('mc_idx:', mc_idx, 'i:', i)

            # handle markdown cell
            elif cell['cell_type'] == 'markdown':
                mark_cell = MarkCell(cell, m_idx, mc_idx, nb_idx)

                all_cells.append(mark_cell)
                all_markdown_cells.append(all_cells[-1])
                m_idx += 1
                mc_idx += 1
                i += 1

                if mc_idx != i:
                    raise ValueError('mc_idx:', mc_idx, 'i:', i)

            # skip other cell types    
            else: 
                i += 1
                print('Encountered cell type:', cell['cell_type'],' skipping cell in load_cell_obj ')

            # add duration of cell to cell object
            if 'metadata' in cell and 'duration' in cell['metadata']:
                all_cells[-1].duration = cell['metadata']['duration']
            else:
                all_cells[-1].duration = None

            # add exception of cell
            if 'metadata' in cell and 'exception' in cell['metadata']:
                all_cells[-1].exception = cell['metadata']['exception']
            else:
                all_cells[-1].exception = None

        # add cell objects to notebook object    
        nb.all_cells = all_cells
        nb.code_cells = all_code_cells
        nb.markdown_cells = all_markdown_cells

        notebooks_cell_obj.append(nb)

    return notebooks_cell_obj