import os
import re
import numpy as np
import tensorflow as tf
from PIL import Image

def load_model(model_path):
    # Cargar el modelo de detección de objetos
    detection_model = tf.saved_model.load(model_path)
    return detection_model

def load_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # Inicializa una lista para almacenar los elementos
    items = []

    # Inicializa variables temporales para almacenar información de cada elemento
    current_item = {}
    for line in lines:
        # Busca el patrón de nombre, id y display_name
        name_match = re.match(r'  name:\s+"(.*)"', line)
        id_match = re.match(r'  id:\s+(\d+)', line)
        display_name_match = re.match(r'  display_name:\s+"(.*)"', line)

        if name_match:
            current_item['name'] = name_match.group(1)
        elif id_match:
            current_item['id'] = int(id_match.group(1))
        elif display_name_match:
            current_item['display_name'] = display_name_match.group(1)
            items.append(current_item)
            current_item = {}

    # Convierte la lista de elementos a formato JSON
    return items

def detect_objects(image_path, detection_model, labels_map):
    # Cargar la imagen
    image = Image.open(image_path)
    image_np = np.array(image)
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Detección de objetos
    detections = detection_model(input_tensor)

    # Filtrar y obtener nombres de objetos
    objects = []
    for score, label in zip(detections['detection_scores'][0], detections['detection_classes'][0]):
        if score > 0.25:  # Umbral de confianza
            label_id = int(label.numpy())
            label_name = "Desconocido"
            for item in labels_map:
                # Verificar si el valor de la clave 'id' es igual a 4
                if item["id"] == label_id:
                    label_name = item["display_name"]
                    break

            objects.append(label_name)

    return objects

def detect_objects_labels(image_path):
    tensorflow_model_path = os.path.join(os.getcwd(), "deepface", "api", "modules", "tensorflow_model")

    saved_model_path = os.path.join(tensorflow_model_path, "saved_model")
    detection_model = load_model(saved_model_path)
    
    label_path = os.path.join(tensorflow_model_path, "mscoco_label_map.pbtxt")
    labels_map = load_labels(label_path)

    objects_found = detect_objects(image_path, detection_model, labels_map)
    print("Objetos encontrados:", objects_found)

    return objects_found
