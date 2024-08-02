import os
import base64

from PIL import Image
from io import BytesIO


def save_image_from_data(data, output_folder="output_default", image_index="unknown_image_index", name="unknown_name"):
    """
    Function to save an image png from base64 encoded data
    
    :param data (str): base64 encoded image data
    :param output_folder (str): folder to save the image
    :param image_index (int): index of the image  (default: "unknown_image_index")
    :param name (str): name of the image (default: "unknown_name")
    
    :return (str): path to the saved image
    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        image_data = base64.b64decode(data)
        image = Image.open(BytesIO(image_data))
        image_path = os.path.join(output_folder, f"{name}_image_{image_index}.png")
        image.save(image_path, format='PNG')
        
        return image_path
    except Exception as e:
        print(f"An error occurred in save_image_from_data: {e}")
        return None
    

