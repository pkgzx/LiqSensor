import cv2
import numpy as np

def preprocess_image(image):
    """
    Preprocesa la imagen para mejorar la detección
    
    Args:
        image: Imagen de entrada en formato BGR
        
    Returns:
        gray: Imagen en escala de grises mejorada
        blurred: Imagen con blur aplicado
        edges: Bordes detectados
    """
    # Convertir a escala de grises
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Aplicar ecualización de histograma para mejorar el contraste
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Aplicar blur bilateral para reducir ruido mientras preserva bordes
    blurred = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # Detección de bordes
    median = np.median(blurred)
    sigma = 0.33
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    edges_canny = cv2.Canny(blurred, lower, upper)
    
    # Sobel para detectar bordes verticales (importantes en botellas)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobel = np.sqrt(sobelx**2 + sobely**2).astype(np.uint8)
    _, edges_sobel = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Combinar resultados de detección de bordes
    edges = cv2.bitwise_or(edges_canny, edges_sobel)
    
    # Operaciones morfológicas para mejorar los bordes
    kernel = np.ones((3, 3), np.uint8)
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)
    
    return gray, blurred, edges

def create_guide_area(image, guide_width=None, guide_height=None):
    """
    Crea un área guía en el centro de la imagen
    
    Args:
        image: Imagen en la que dibujar la guía
        guide_width: Ancho opcional de la guía
        guide_height: Alto opcional de la guía
        
    Returns:
        guide_x, guide_y: Coordenadas de inicio de la guía
        guide_width, guide_height: Dimensiones de la guía
        guide_mask: Máscara del área guía
    """
    h, w = image.shape[:2]
    
    # Valores predeterminados si no se proporcionan
    if guide_width is None:
        guide_width = 130
    if guide_height is None:
        guide_height = 320
    
    guide_x = w // 2 - guide_width // 2
    guide_y = h // 2 - guide_height // 2
    
    # Crear máscara para el área guía
    guide_mask = np.zeros((h, w), dtype=np.uint8)
    guide_mask[guide_y:guide_y+guide_height, guide_x:guide_x+guide_width] = 255
    
    return guide_x, guide_y, guide_width, guide_height, guide_mask

def detect_bottle_contours(edges, guide_x, guide_y, guide_width, guide_height, min_area=500):
    """
    Detecta contornos que podrían ser botellas dentro o cerca del área guía
    
    Args:
        edges: Imagen de bordes
        guide_x, guide_y, guide_width, guide_height: Definición del área guía
        min_area: Área mínima para considerar un contorno
    
    Returns:
        bottle_contours: Lista de contornos de botellas candidatas
        contour_scores: Puntuación para cada contorno
    """
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    bottle_contours = []
    contour_scores = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Verificar si está dentro o cerca del área guía
            if (x + w > guide_x - 30 and x < guide_x + guide_width + 30 and
                y + h > guide_y - 30 and y < guide_y + guide_height + 30):
                
                # Análisis básico de forma para filtrar contornos no válidos
                aspect_ratio = h / (w + 0.01)
                
                if aspect_ratio > 1.0:  # Más alto que ancho (típico de botellas)
                    bottle_contours.append(contour)
                    contour_scores.append(area)  # Usar área como score inicial
    
    return bottle_contours, contour_scores

def analyze_contour_shape(contour, gray, guide_x, guide_y, guide_width, guide_height):
    """
    Analiza la forma de un contorno para determinar si es una botella
    
    Args:
        contour: Contorno a analizar
        gray: Imagen en escala de grises
        guide_x, guide_y, guide_width, guide_height: Definición del área guía
    
    Returns:
        score: Puntuación entre 0 y 1 (mayor = más probable que sea botella)
        features: Diccionario con características extraídas del contorno
    """
    # Características a extraer
    features = {}
    
    # 1. Rectángulo mínimo
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(np.int_)
    rect_w, rect_h = rect[1]
    
    # 2. Relación de aspecto
    aspect_ratio = max(rect_h, rect_w) / (min(rect_h, rect_w) + 0.01)
    features['aspect_ratio'] = aspect_ratio
    
    # 3. Convexidad
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    contour_area = cv2.contourArea(contour)
    convexity = contour_area / hull_area if hull_area > 0 else 0
    features['convexity'] = convexity
    
    # 4. Centroide
    moments = cv2.moments(contour)
    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        features['centroid'] = (cx, cy)
        
        # Distancia al centro de la guía
        center_x = guide_x + guide_width / 2
        center_y = guide_y + guide_height / 2
        distance_from_center = np.sqrt((cx - center_x)**2 + (cy - center_y)**2)
        normalized_distance = 1 - min(1, distance_from_center / (guide_width/2))
        features['center_distance'] = normalized_distance
    else:
        features['centroid'] = None
        features['center_distance'] = 0
    
    # 5. Análisis de perfil para detección de cuello
    x, y, w, h = cv2.boundingRect(contour)
    mask = np.zeros_like(gray)
    cv2.drawContours(mask, [contour], 0, 255, -1)
    
    # Dividir en secciones para analizar estrechamiento
    sections = 8
    section_heights = np.array_split(np.arange(y, y+h), sections)
    section_widths = []
    
    for section in section_heights:
        if len(section) > 0:
            section_mask = mask[section[0]:section[-1]+1, x:x+w]
            horizontal_projection = np.sum(section_mask, axis=0) > 0
            if np.any(horizontal_projection):
                non_zero_indices = np.where(horizontal_projection)[0]
                section_width = non_zero_indices[-1] - non_zero_indices[0]
                section_widths.append(section_width)
    
    if len(section_widths) >= 4:
        upper_width = np.mean(section_widths[:len(section_widths)//2])
        lower_width = np.mean(section_widths[len(section_widths)//2:])
        if upper_width > 0 and lower_width > 0:
            width_ratio = upper_width / lower_width
            neck_detection = 1 - min(1, width_ratio)
            features['neck_detection'] = neck_detection
        else:
            features['neck_detection'] = 0
    else:
        features['neck_detection'] = 0
    
    # 6. Verticalidad
    angle = abs(rect[2])
    if angle > 45:
        angle = 90 - angle
    verticality = 1 - min(1, angle / 30)
    features['verticality'] = verticality
    
    # Calcular puntuación compuesta
    score = (
        0.25 * (1 if 1.5 < aspect_ratio < 10 else 0) +  # Aspecto
        0.20 * (0.8 if 0.8 < convexity < 0.98 else 0) +  # Convexidad
        0.15 * features.get('center_distance', 0) +  # Cercanía al centro
        0.25 * features.get('neck_detection', 0) +  # Cuello
        0.15 * verticality  # Verticalidad
    )
    
    return score, features, box

def extract_roi(image, gray, mask, contour=None, guide_x=0, guide_y=0, guide_width=0, guide_height=0):
    """
    Extrae la región de interés, ya sea usando un contorno específico o el área guía
    
    Args:
        image: Imagen original
        gray: Imagen en escala de grises
        mask: Máscara de la botella
        contour: Contorno de la botella (opcional)
        guide_x, guide_y, guide_width, guide_height: Para usar si no hay contorno
    
    Returns:
        roi_color: ROI de la imagen a color
        roi_gray: ROI de la imagen en escala de grises
        roi_mask: ROI de la máscara
    """
    if contour is not None:
        x, y, w, h = cv2.boundingRect(contour)
        
        # Expandir el área ligeramente
        x_padding = int(w * 0.1)
        y_padding = int(h * 0.05)
        
        # Asegurar que está dentro de los límites
        height, width = image.shape[:2]
        x_start = max(0, x - x_padding)
        y_start = max(0, y - y_padding)
        x_end = min(width, x + w + x_padding)
        y_end = min(height, y + h + y_padding)
        
        # Recortar las imágenes
        roi_color = image[y_start:y_end, x_start:x_end]
        roi_gray = gray[y_start:y_end, x_start:x_end]
        roi_mask = mask[y_start:y_end, x_start:x_end]
        
        # Coordenadas para uso posterior
        roi_coords = (x_start, y_start, x_end-x_start, y_end-y_start)
    else:
        # Usar área guía
        roi_color = image[guide_y:guide_y+guide_height, guide_x:guide_x+guide_width]
        roi_gray = gray[guide_y:guide_y+guide_height, guide_x:guide_x+guide_width]
        roi_mask = mask[guide_y:guide_y+guide_height, guide_x:guide_x+guide_width]
        
        # Coordenadas para uso posterior
        roi_coords = (guide_x, guide_y, guide_width, guide_height)
    
    return roi_color, roi_gray, roi_mask, roi_coords

def image_to_base64(img):
    """Convierte una imagen de OpenCV a base64 para enviarla al frontend"""
    import base64
    import io
    from PIL import Image
    
    try:
        if img is None or img.size == 0:
            return ""
        
        # Asegurarse de que sea una imagen válida
        if len(img.shape) == 2:  # Si es grayscale
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 3:  # Si es BGR
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode()
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"Error en image_to_base64: {str(e)}")
        return ""