import cv2
import numpy as np

def detectar_reflejos(img_path):
    # Leer la imagen
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: No se pudo cargar la imagen en {img_path}")
        return None
    
    # Convertir a escala de grises
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Aplicar el filtro de Sobel para detectar bordes
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    sobel = cv2.magnitude(sobelx, sobely)
    
    # Normalizar y convertir a 8-bit
    sobel = cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX)
    sobel = np.uint8(sobel)
    
    # Umbralizar la imagen para obtener las regiones brillantes
    _, binary_sobel = cv2.threshold(sobel, 150, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos en la imagen binaria
    contours, _ = cv2.findContours(binary_sobel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área mínima para eliminar ruido
    min_area = 200  # Ajusta según sea necesario
    large_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]
    
    # Dibujar los contornos detectados en la imagen original
    img_contours = img.copy()
    cv2.drawContours(img_contours, large_contours, -1, (0, 255, 0), 2)
    
    # Retornar la cantidad de reflejos detectados
    return len(large_contours)

def detectar_bandas_desplazamiento(img_path):
    # Leer la imagen
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Aplicar un filtro de Laplaciano
    laplacian = cv2.Laplacian(image, cv2.CV_64F)

    # Calcular el valor absoluto y convertir a uint8
    laplacian_abs = cv2.convertScaleAbs(laplacian)

    # Umbralizar la imagen
    _, binary_image = cv2.threshold(laplacian_abs, 30, 255, cv2.THRESH_BINARY)

    # Contar píxeles blancos (bandas de desplazamiento)
    white_pixels = cv2.countNonZero(binary_image)

    # Mostrar la imagen umbralizada
    cv2.imshow('Thresholded Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("detectar_bandas_desplazamiento:", white_pixels)

def detectar_moire(img_path):
    # Leer la imagen
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # Calcular la transformada de Fourier
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20*np.log(np.abs(fshift))

    # Calcular el umbral basado en la media y la desviación estándar del espectro de magnitud
    mean_value = np.mean(magnitude_spectrum)
    std_dev = np.std(magnitude_spectrum)
    umbral = mean_value + std_dev * 2  # Ajustar el factor según sea necesario

    # Contar puntos brillantes en el espectro de magnitud
    bright_points = np.sum(magnitude_spectrum > umbral)  # umbral ajustable

    print("detectar_moire:", bright_points)

    # Mostrar el espectro de magnitud
    cv2.imshow('Magnitude Spectrum', magnitude_spectrum)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectar_artefactos(img_path):
    # Leer la imagen
    image = cv2.imread(img_path)

    # Convertir a escala de grises
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Difuminar la imagen para suavizar los artefactos
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)

    # Calcular la diferencia entre la imagen original y la difuminada
    difference = cv2.absdiff(gray_image, blurred_image)

    # Umbralizar la diferencia para resaltar los artefactos
    _, binary_image = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)

    # Mostrar la imagen umbralizada
    cv2.imshow('detectar_artefactos Image', binary_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectar_bandas_de_colores(img_path):
    # Leer la imagen
    image = cv2.imread(img_path)

    # Convertir la imagen a espacio de color HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Definir el rango de colores para las bandas (ejemplo: azul)
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])
    
    # Filtrar los colores dentro del rango especificado
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Aplicar una operación morfológica para eliminar el ruido
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Encontrar contornos de las áreas coloreadas
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar contornos en la imagen original
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
# Mostrar el resultado
    cv2.imshow('Resultado', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

def detectar_patron_moire(img_path):
    # Leer la imagen
    image = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Suavizar la imagen para eliminar el ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Calcular la transformada de Fourier
    f = np.fft.fft2(blurred)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    
    # Buscar picos en el espectro de frecuencia
    _, magnitude_spectrum_thresholded = cv2.threshold(magnitude_spectrum, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(magnitude_spectrum_thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Verificar si hay patrones repetitivos que puedan indicar un patrón Moiré
    pattern_detected = False
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 100:  # Ajustar el umbral del área según sea necesario
            pattern_detected = True
            break

    if pattern_detected:
        print("Patrón Moiré detectado")
    else:
        print("No se detectó un patrón Moiré")
    
    # Mostrar el espectro de frecuencia
    cv2.imshow('Magnitude Spectrum', magnitude_spectrum.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return pattern_detected, magnitude_spectrum

def analizar_distribucion_frecuencias(img_path):
    # Leer la imagen
    image = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Calcular la transformada de Fourier
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = np.log(np.abs(fshift) + 1)  # Aplicar logaritmo para mejorar la visualización
    
    # Calcular el histograma del espectro de magnitudes
    hist, bins = np.histogram(magnitude_spectrum.ravel(), bins=256, range=(0, 255))
    
    # Normalizar el histograma
    hist_normalized = hist.astype(np.float32) / hist.max()
    
    # Calcular el centro de los intervalos de bins para el eje x
    bins_centers = (bins[:-1] + bins[1:]) / 2
    
    return bins_centers, hist_normalized

def mostrar_histograma(img_path):
    # Leer la imagen
    image = cv2.imread(img_path)

    # Convertir la imagen a escala de grises si es necesario
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Calcular el histograma
    hist = cv2.calcHist([gray], [0], None, [256], [0,256])
    
def detectar_contornos(img_path):
    # Leer la imagen
    image = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar un umbral binario
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    
    # Encontrar contornos en la imagen binarizada
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"Contornos detectados: {len(contours)}")

    # Dibujar contornos en la imagen original
    result = image.copy()
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    
    # Mostrar el resultado
    cv2.imshow('Resultado', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

def detectar_lineas(img_path):
    # Leer la imagen
    image = cv2.imread(img_path)

    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar detección de bordes utilizando Canny
    edges = cv2.Canny(gray, 10, 70, apertureSize=3)
    
    # Aplicar la transformada de Hough para detectar líneas
    lines = cv2.HoughLines(edges, 1, np.pi/180, 150)
    
    # Dibujar las líneas detectadas sobre la imagen original
    result = image.copy()
    if lines is not None:
        for rho, theta in lines[:, 0]:
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))
            cv2.line(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    
    print(f"Líneas detectadas: {len(lines)}")

    # Mostrar el resultado
    cv2.imshow('LINEAS', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return result

def detectar_rectangulos(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.IMREAD_GRAYSCALE)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # using the Canny edge detector
    wide = cv2.Canny(blurred, 10, 200)
    mid = cv2.Canny(blurred, 30, 150)
    tight = cv2.Canny(blurred, 240, 250)
    # show the output Canny edge maps
    cv2.imshow("Wide Edge Map", wide)
    cv2.imshow("Mid Edge Map", mid)
    cv2.imshow("Tight Edge Map", tight)
    cv2.waitKey(0)

    # Aplicar la detección de bordes utilizando Canny
    # edges = cv2.Canny(gray, 100, 200, apertureSize=3, L2gradient=True)
    edges = mid
    
    # Encontrar contornos en la imagen
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Dibujar contornos que son aproximadamente rectangulares
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.05 * perimeter, True)  # Aumento del umbral de aproximación
        if len(approx) == 4:
            cv2.drawContours(image, [contour], -1, (0, 255, 0), 2)
    
    # Mostrar la imagen con los rectángulos detectados
    cv2.imshow('Rectáculos Detectados', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectar_rectangulos_con_rostros(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Aplicar la detección de bordes utilizando Canny
    # edges = cv2.Canny(gray, 100, 200, apertureSize=3, L2gradient=True)
    edges = cv2.Canny(blurred, 30, 100)
    
    # Encontrar contornos en la imagen
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cargar el clasificador Haar Cascade para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Dibujar contornos que son aproximadamente rectangulares y detectar rostros dentro de ellos
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]
            rostros = face_cascade.detectMultiScale(roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            for (x_face, y_face, width_face, height_face) in rostros:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, 'Rostro detectado', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                break  # Salir del bucle si se detecta un rostro dentro del rectángulo
    
    # Mostrar la imagen con los rectángulos y rostros detectados
    cv2.imshow('Rectáculos y Rostros Detectados', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def punto_en_poligono(point, polygon):
    # Convertir el punto a una tupla
    point = (int(point[0]), int(point[1]))
    
    # Crear un path con los puntos del polígono
    path = np.array(polygon, dtype=np.int32)
    
    # Verificar si el punto está dentro del polígono
    result = cv2.pointPolygonTest(path, point, False)
    
    return result >= 0

def detectar_trapecios_con_rostros(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Aplicar la detección de bordes utilizando Canny
    # edges = cv2.Canny(gray, 30, 150, apertureSize=3)
    edges = cv2.Canny(blurred, 10, 150, apertureSize=3)
    
    # Encontrar contornos en la imagen
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cargar el clasificador Haar Cascade para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Dibujar contornos que son aproximadamente trapecios y detectar rostros dentro de ellos
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        if len(approx) == 4:  # Si el contorno tiene 4 lados, puede ser un trapecio
            # Obtener los puntos del trapecio
            trapecio = approx.reshape(4, 2)
            
            # Dibujar el trapecio encontrado
            cv2.polylines(image, [trapecio], True, (0, 255, 0), 2)

            # Detectar rostros en la imagen completa
            rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Verificar si algún rostro está dentro del trapecio
            for (x_face, y_face, width_face, height_face) in rostros:
                centro_rostro = (x_face + width_face // 2, y_face + height_face // 2)
                if punto_en_poligono(centro_rostro, trapecio):
                    cv2.polylines(image, [trapecio], True, (0, 255, 0), 2)
                    cv2.putText(image, 'Rostro detectado', (trapecio[0][0], trapecio[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    cv2.rectangle(image, (x_face, y_face), (x_face + width_face, y_face + height_face), (255, 0, 0), 2)
                    break  # Salir del bucle si se detecta un rostro dentro del trapecio
    
    # Mostrar la imagen con los trapecios y rostros detectados
    cv2.imshow('Trapecios y Rostros Detectados', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectar_formas_con_rostros(image_path):
    # Cargar la imagen
    image = cv2.imread(image_path)
    
    # Convertir la imagen a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar la detección de bordes utilizando Canny
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    
    # Encontrar contornos en la imagen
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Cargar el clasificador Haar Cascade para detección de rostros
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detectar rostros en la imagen completa
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Dibujar contornos que representan marcos y detectar rostros dentro de ellos
    for contour in contours:
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
        
        # Dibujar el marco encontrado
        # cv2.polylines(image, [approx], True, (0, 255, 0), 2)
        
        # Verificar si algún rostro está dentro del marco
        for (x_face, y_face, width_face, height_face) in rostros:
            centro_rostro = (x_face + width_face // 2, y_face + height_face // 2)
            if punto_en_poligono(centro_rostro, approx):
                cv2.putText(image, 'Rostro detectado', (x_face, y_face - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(image, (x_face, y_face), (x_face + width_face, y_face + height_face), (255, 0, 0), 2)
                break  # Salir del bucle si se detecta un rostro dentro del marco
    
    # Mostrar la imagen con los marcos y rostros detectados
    cv2.imshow('Formas y Rostros Detectados', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def detectar_lineas_verticales(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 20, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)
    
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(x2 - x1) < 10:  # Considerar la línea como vertical si la diferencia en x es pequeña
                vertical_lines.append((x1, y1, x2, y2))
    
    return vertical_lines

def detectar_rostros(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def detectar_rostros_entre_lineas_verticales(image_path):
    image = cv2.imread(image_path)
    
    # Detectar líneas verticales
    vertical_lines = detectar_lineas_verticales(image)
    if len(vertical_lines) < 2:
        print("No se detectaron suficientes líneas verticales.")
        return False
    
    # Ordenar líneas por coordenada x
    vertical_lines.sort(key=lambda line: line[0])
    line_x1 = vertical_lines[0][0]
    line_x2 = vertical_lines[-1][0]
    
    # Dibujar líneas en la imagen (opcional, para visualización)
    for x1, y1, x2, y2 in vertical_lines:
        cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Mostrar la imagen con los marcos y rostros detectados
    # cv2.imshow('Lineas verticales', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Detectar rostros
    faces = detectar_rostros(image)
    
    # Verificar si algún rostro está entre las líneas
    for (x, y, w, h) in faces:
        face_center_x = x + w // 2
        if line_x1 < face_center_x < line_x2:
            print(f"Rostro detectado entre las líneas en coordenada X: {face_center_x}")
            return True
    
    print("No se detectó ningún rostro entre las líneas.")
    return False

# Rutas de la imagen
# img_path = "D:\\REPOSITORIOS LOCALES\\DEALER_DEEPFACE\\deepface_img\\2024-05-24-20_26_13\\imagen_original.jpg"
# img_path = "D:\\REPOSITORIOS LOCALES\\DEALER_DEEPFACE\\deepface_img\\2024-05-24-20_26_13\\imagen_captura01.jpg"
# img_path = "D:\\REPOSITORIOS LOCALES\\DEALER_DEEPFACE\\deepface_img\\2024-05-24-20_09_32\\imagen_captura01.jpg"

# # Detectar reflejos y obtener la cantidad de reflejos detectados
# cantidad_reflejos = detectar_reflejos(img_path)
# if cantidad_reflejos is not None:
#     print(f"Cantidad de reflejos detectados: {cantidad_reflejos}")

# detectar_bandas_desplazamiento(img_path)

# detectar_moire(img_path)

# detectar_artefactos(img_path)

# detectar_bandas_de_colores(img_path)

# detectar_patron_moire(img_path)

# analizar_distribucion_frecuencias(img_path)

# mostrar_histograma(img_path)

# detectar_contornos(img_path)

# detectar_lineas(img_path)

# detectar_rectangulos(img_path)

# detectar_rectangulos_con_rostros(img_path)

# detectar_trapecios_con_rostros(img_path)

# detectar_formas_con_rostros(img_path)

# detectar_rostros_entre_lineas_verticales(img_path)