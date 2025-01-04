from email import encoders
from email.mime.base import MIMEBase
import threading
from flask import Blueprint, request
from modules import service
from deepface.commons import logger as log
from modules import helpers_cv2 as hcv2
from modules import helpers_tf as htf
from PIL import Image
from datetime import datetime
import base64
import io
import os
import smtplib
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = log.get_singletonish_logger()

blueprint = Blueprint("routes", __name__)

@blueprint.route("/")
def home():
    return "<h1>Bienvenido a la API</h1>"

@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)

    obj = service.represent(
        img_path=img_path,
        model_name=model_name,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(obj)

    return obj

@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img1_path = input_args.get("img1") or input_args.get("img1_path")
    img2_path = input_args.get("img2") or input_args.get("img2_path")

    if img1_path is None:
        return {"message": "you must pass img1_path input"}

    if img2_path is None:
        return {"message": "you must pass img2_path input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

    verification = service.verify(
        img1_path=img1_path,
        img2_path=img2_path,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )

    logger.debug(verification)

    return verification

@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    img_path = input_args.get("img") or input_args.get("img_path")
    if img_path is None:
        return {"message": "you must pass img_path input"}

    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    align = input_args.get("align", True)
    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])

    demographies = service.analyze(
        img_path=img_path,
        actions=actions,
        detector_backend=detector_backend,
        enforce_detection=enforce_detection,
        align=align,
    )

    logger.debug(demographies)

    return demographies

@blueprint.route("/verify_string_base64", methods=["POST"])
def verify_string_base64():
    input_args = request.get_json()

    if input_args is None:
        return {"message": "empty input set passed"}

    loLicencia = input_args.get("licencia")

    img1 = input_args.get("img1")
    img2 = input_args.get("img2")
    img3 = input_args.get("img3")
    data = input_args.get("data")

    if img1 is None:
        return {"message": "you must pass img1 input"}

    if img2 is None:
        return {"message": "you must pass img2 input"}

    model_name = input_args.get("model_name", "VGG-Face")
    detector_backend = input_args.get("detector_backend", "opencv")
    enforce_detection = input_args.get("enforce_detection", True)
    distance_metric = input_args.get("distance_metric", "cosine")
    align = input_args.get("align", True)

    pathImagen = os.path.join(os.getcwd(), "deepface_img")

    if loLicencia is not None:
        pathImagen = os.path.join(pathImagen, loLicencia)

    fecha_hora_actual = datetime.now()

    pathImagen = os.path.join(pathImagen, str(fecha_hora_actual.strftime("%Y-%m-%d-%H_%M_%S")))

    os.makedirs(pathImagen)

    bytes_imagen1 = base64.b64decode(img1)
    imagen1 = Image.open(io.BytesIO(bytes_imagen1))
    # imagen1 = imagen1.rotate(180)
    ruta_imagen1 = os.path.join(pathImagen, "imagen_original.jpg")
    imagen1.save(ruta_imagen1)
    imagen1.close()

    bytes_imagen2 = base64.b64decode(img2)
    imagen2 = Image.open(io.BytesIO(bytes_imagen2))
    ruta_imagen2 = os.path.join(pathImagen, "imagen_captura01.jpg")
    imagen2.save(ruta_imagen2)
    imagen2.close()

    if img3 is not None:
        bytes_imagen3 = base64.b64decode(img3)
        imagen3 = Image.open(io.BytesIO(bytes_imagen3))
        ruta_imagen3 = os.path.join(pathImagen, "imagen_captura02.jpg")
        imagen3.save(ruta_imagen3)
        imagen3.close()

    txt_subject = "DEALER APP - ACTIVIDAD SOSPECHOSA"
    if data is not None:
        diccionario = json.loads(data)
        txt_subject = f"DEALER APP - ACTIVIDAD SOSPECHOSA - {diccionario.get("movimiento_tipo")} [{diccionario.get("fecha_hora")}]"

    verification = service.verify(
        img1_path=ruta_imagen1,
        img2_path=ruta_imagen2,
        model_name=model_name,
        detector_backend=detector_backend,
        distance_metric=distance_metric,
        align=align,
        enforce_detection=enforce_detection,
    )

    if isinstance(verification, dict) and "verified" in verification:
        verification["verified"] = str(verification["verified"])
    else:
        verification["verified"] = "False"

    mi_hilo = threading.Thread(target=verify_img, args=(txt_subject, ruta_imagen2, "soporte@dealermovil.com",))
    mi_hilo.start()

    return verification

def verify_img(txt_subject, img_path, dir_email):
    # deteccion de reflejos en la imagen
    cantidad_reflejos = hcv2.detectar_reflejos(img_path)
    if cantidad_reflejos is not None:
        if cantidad_reflejos > 0:
            print("Reflejos detectados")
            send_email(txt_subject, img_path, dir_email)
            return

    # deteccion de objetos sospechosos en la imagen
    first_list = htf.detect_objects_labels(img_path)
    second_list = ["tv", "notebook", "phone", "celphone", "monitor"]

    for item in first_list:
        if item in second_list:
            print("Items detectados")
            send_email(txt_subject, img_path, dir_email)
            return

    # deteccion de rostros entre lineas verticales
    existe_imagen = hcv2.detectar_rostros_entre_lineas_verticales(img_path)
    if existe_imagen:
        print("Rosotros detectados entre lineas verticales")
        send_email(txt_subject, img_path, dir_email)
        return

def send_email(txt_subject, img_path, dir_email):
    # Email configuration
    sender_email = "dealer.no.responder@gmail.com"
    receiver_email = dir_email
    password = "qlohuvwghojxcele"

    # Create a multipart message
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = txt_subject

    # Add body to the email
    body = "Se detecto una actividad sospechosa en el login app movil. Verifique la imagen adjunta."
    message.attach(MIMEText(body, "plain"))

    # Attach the image
    with open(img_path, "rb") as attachment:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header(
            "Content-Disposition",
            f"attachment; filename= {os.path.basename(img_path)}",
        )
        message.attach(part)

    # SMTP server configuration
    smtp_server = "smtp.gmail.com"
    smtp_port = 587

    # Start the SMTP session
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()

    # Login to your Gmail account
    server.login(sender_email, password)

    # Send the email
    server.sendmail(sender_email, receiver_email, message.as_string())

    # Close the SMTP session
    server.quit()