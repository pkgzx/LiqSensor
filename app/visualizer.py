import cv2
import numpy as np
from config import VISUALIZATION, LEVEL_THRESHOLDS

def draw_guide_area(frame, guide_x, guide_y, guide_width, guide_height):
    """
    Dibuja el área guía en la imagen
    
    Args:
        frame: Imagen donde dibujar
        guide_x, guide_y: Coordenadas de inicio
        guide_width, guide_height: Dimensiones
    
    Returns:
        frame: Imagen con área guía dibujada
    """
    cv2.rectangle(frame, (guide_x, guide_y), 
                 (guide_x + guide_width, guide_y + guide_height), (0, 255, 0), 2)
    cv2.putText(frame, "Área de análisis", (guide_x - 50, guide_y - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    return frame

def draw_bottle_contour(frame, contour, box, score):
    """
    Dibuja el contorno de la botella detectada
    
    Args:
        frame: Imagen donde dibujar
        contour: Contorno de la botella
        box: Puntos del rectángulo rotado
        score: Puntuación de confianza
    
    Returns:
        frame: Imagen con contorno dibujado
    """
    cv2.drawContours(frame, [box], 0, (0, 255, 255), 2)
    
    x, y, w, h = cv2.boundingRect(contour)
    confidence_text = f"Confianza: {score*100:.1f}%"
    cv2.putText(frame, confidence_text, (x, y-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return frame

def draw_level_indicator(frame, x, y, w, h, nivel_detectado):
    """
    Dibuja la línea indicadora del nivel de líquido
    
    Args:
        frame: Imagen donde dibujar
        x, y, w, h: Coordenadas y dimensiones del área
        nivel_detectado: Nivel de líquido (porcentaje)
    
    Returns:
        frame: Imagen con nivel dibujado
    """
    nivel_y = y + h - int(nivel_detectado * h / 100)
    cv2.line(frame, (x, nivel_y), (x + w, nivel_y), (0, 0, 255), 2)
    
    nivel_text = f"{round(nivel_detectado, 1)}%"
    cv2.putText(frame, nivel_text, (x + w + 10, nivel_y), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame

def draw_level_bar(frame, nivel_detectado):
    """
    Dibuja una barra vertical de nivel en el lateral de la imagen
    
    Args:
        frame: Imagen donde dibujar
        nivel_detectado: Nivel de líquido (porcentaje)
    
    Returns:
        frame: Imagen con barra de nivel
    """
    h, w = frame.shape[:2]
    
    # Parámetros de la barra
    bar_width = VISUALIZATION["bar_width"]
    bar_height = VISUALIZATION["bar_height"]
    bar_x = w - 50
    bar_y = (h - bar_height) // 2
    
    # Fondo de la barra
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
    
    # Nivel de llenado
    filled_height = int(bar_height * nivel_detectado / 100)
    fill_y = bar_y + bar_height - filled_height
    
    # Color según nivel
    if nivel_detectado < LEVEL_THRESHOLDS["low"]:
        color = VISUALIZATION["low_level_color"]  # Rojo
    elif nivel_detectado < LEVEL_THRESHOLDS["high"]:
        color = VISUALIZATION["mid_level_color"]  # Amarillo
    else:
        color = VISUALIZATION["high_level_color"]  # Verde
        
    cv2.rectangle(frame, (bar_x, fill_y), (bar_x + bar_width, bar_y + bar_height), color, -1)
    
    # Marcas de porcentaje
    for i in range(0, 101, 25):
        mark_y = bar_y + bar_height - int(i * bar_height / 100)
        cv2.line(frame, (bar_x - 5, mark_y), (bar_x, mark_y), (255, 255, 255), 1)
        cv2.putText(frame, f"{i}%", (bar_x - 35, mark_y + 5), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Texto con el nivel actual
    cv2.putText(frame, f"Nivel: {round(nivel_detectado, 1)}%", (bar_x - 80, bar_y - 20), 
              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    return frame

def create_debug_visualization(frame, roi, mask, nivel_detectado):
    """
    Crea una visualización con datos de debug
    
    Args:
        frame: Imagen principal
        roi: Región de interés
        mask: Máscara de detección
        nivel_detectado: Nivel detectado
    
    Returns:
        debug_frame: Imagen con visualizaciones de debug
    """
    debug_frame = frame.copy()
    h, w = debug_frame.shape[:2]
    
    # Si ROI y mask son válidas, mostrarlas en una esquina
    if roi.size > 10 and mask.size > 10:
        # Redimensionar
        roi_resized = cv2.resize(roi, (w//4, h//4))
        mask_resized = cv2.resize(mask, (w//4, h//4))
        
        # Convertir a BGR si es necesario
        if len(roi_resized.shape) == 2:
            roi_bgr = cv2.cvtColor(roi_resized, cv2.COLOR_GRAY2BGR)
        else:
            roi_bgr = roi_resized
            
        if len(mask_resized.shape) == 2:
            mask_bgr = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
        else:
            mask_bgr = mask_resized
        
        # Añadir a la imagen principal
        debug_frame[0:h//4, 0:w//4] = roi_bgr
        debug_frame[0:h//4, w//4:(w//4)*2] = mask_bgr
        
        # Etiquetas
        cv2.putText(debug_frame, "ROI", (10, 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(debug_frame, "Máscara", (w//4 + 10, 20), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # Añadir la barra de nivel
    debug_frame = draw_level_bar(debug_frame, nivel_detectado)
    
    return debug_frame