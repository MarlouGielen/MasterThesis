import time
from .hardcoded_keywords import get_keywords, classify_by_keywords   


def classify_nb_cells(notebooks, method="all"):
    """
    Function to classify a cell based on the keywords.

    :param notebooks (lst): list of dictionaries, each dictionary representing a notebook
    :param method (str): thh classification method to be used, default "all"

    :return max_classification (str): class_probability
    :return class_probability (dict): dictionary with the class probabilities
    :return detailed_scores (dict): dictionary with the detailed scores
    """
    t0 = time.time()
    class_count = {
        "Environment": 0,
        "Data_Extraction": 0,
        "Exploratory_Data_Analysis": 0,
        "Data_Transform": 0,
        "Model_Train": 0,
        "Model_Evaluation": 0,
        "Model_Interpretation": 0,
        "Hyperparameter_Tuning": 0,
        "Visualization": 0,
        "Debug": 0,
        "Data_Export": 0,
        "Other": 0,
    }

    # Classification by keywords
    if method.upper() in ["HARDCODED", "SKLEARN", "NUMPY", "PANDAS", "TENSORFLOW", "TORCH", "ALL", "ALL_NO_HARDCODED"]:
        empty_classes = class_count.copy()
        empty_detailed_scores = [ {key: [] for key in empty_classes.keys()} ]
        keywords = get_keywords(class_count, method=method) 
        for nb in notebooks:
            class_count_nb = {
                "Environment": 0,
                "Data_Extraction": 0,
                "Exploratory_Data_Analysis": 0,
                "Data_Transform": 0,
                "Model_Train": 0,
                "Model_Evaluation": 0,
                "Model_Interpretation": 0,
                "Hyperparameter_Tuning": 0,
                "Visualization": 0,
                "Debug": 0,
                "Data_Export": 0,
                "Other": 0,
            }
            for cell in nb.all_cells:
                if cell.cell_type == "code":
                    max_classification, class_probability, detailed_scores = classify_by_keywords(cell.source, keywords)
                    class_count[max_classification] += 1
                    class_count_nb[max_classification] += 1
                else:
                    max_classification = "Markdown"
                    class_probability = empty_classes
                    detailed_scores = empty_detailed_scores

                cell.classification = max_classification
                cell.class_probability = class_probability
                cell.detailed_scores = detailed_scores

                # store all keywords in detailed_scores into 'keywords

            nb.class_count = class_count_nb
                
        return class_count
        
    # Classification by ML model
    elif method.upper() == "ML":
        print("ML model not implemented yet")
        pass

    # Classification by local LLM model
    elif method.upper() == "LLM":
        print("LLM model not implemented yet")
        pass

    else:
        raise ValueError("Method not recognized. Please use 'hardcoded', 'sklearn', 'all', 'ML' or 'LLM'.")
        pass
    
    print(f"Total classification time: {time.time()-t0:.2f} seconds")

    return class_count


