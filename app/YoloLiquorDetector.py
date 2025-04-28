import cv2
import numpy as np
import torch
from ultralytics import YOLO
import time
from pathlib import Path

from config import LIQUOR_CONFIGS, PROCESSING_PARAMS, VISUALIZATION
from visualizer import draw_level_indicator, draw_level_bar

class YoloLiquorDetector:
    def __init__(self, model_path=None, conf_threshold=0.3):
        """
        Inicializa el detector basado en YOLOv8
        
        Args:
            model_path: Ruta al modelo YOLO entrenado o None para usar YOLOv8n
            conf_threshold: Umbral de confianza para detecciones
        """
        print("Inicializando detector YOLOv8...")
        
        # Cargar modelo YOLO - usa modelo predeterminado o uno personalizado
        if model_path and Path(model_path).exists():
            self.model = YOLO(model_path)
            print(f"Modelo cargado desde {model_path}")
        else:
            self.model = YOLO("yolov8n.pt")  # Modelo pequeño por defecto
            print("Usando modelo YOLOv8n predeterminado")
        
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Usando dispositivo: {self.device}")
        
        # Clases relevantes para botellas en COCO
        self.bottle_classes = [39]  # 39 = bottle en COCO
        
    def detect_bottle(self, image):
        """
        Detecta botellas en la imagen usando YOLO
        
        Args:
            image: Imagen BGR de OpenCV
        
        Returns:
            box: Coordenadas [x1, y1, x2, y2] de la botella detectada o None
            confidence: Nivel de confianza de la detección
        """
        # Realizar predicción con YOLO
        results = self.model(image, conf=self.conf_threshold, classes=self.bottle_classes)
        
        # Si se encontraron botellas, devolver la más confiable
        if results and len(results) > 0 and len(results[0].boxes) > 0:
            # Obtener la caja con mayor confianza
            boxes = results[0].boxes
            confidences = boxes.conf.cpu().numpy()
            best_idx = np.argmax(confidences)
            
            # Obtener coordenadas [x1, y1, x2, y2]
            box = boxes.xyxy.cpu().numpy()[best_idx].astype(int)
            confidence = confidences[best_idx]
            
            return box, confidence
        
        return None, 0.0
        
    def analyze_liquid_level(self, image, bottle_box, liquor_type="whisky"):
        """
        Analiza el nivel de líquido en la botella detectada
        
        Args:
            image: Imagen original BGR
            bottle_box: Coordenadas [x1, y1, x2, y2] de la botella
            liquor_type: Tipo de licor para configuración específica
            
        Returns:
            level_percentage: Porcentaje de llenado (0-100)
            visualization: Imagen con visualizaciones
            roi: Región de interés (botella recortada)
            mask: Máscara de líquido detectado
        """
        if bottle_box is None:
            return 0, image, None, None
            
        # Configuración específica para el tipo de licor
        bottle_config = LIQUOR_CONFIGS.get(liquor_type, LIQUOR_CONFIGS["whisky"])
        
        # Extraer ROI (Región de Interés)
        x1, y1, x2, y2 = bottle_box
        roi = image[y1:y2, x1:x2].copy()
        
        if roi.size == 0:
            return 0, image, None, None
            
        # Convertir a escala de grises y HSV
        roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Crear máscara para detección de líquido
        liquid_mask = np.zeros_like(roi_gray)
        
        # Aplicar rangos de color específicos para el tipo de licor
        hsv_ranges = bottle_config.get("hsv_ranges", [])
        for hsv_range in hsv_ranges:
            mask = cv2.inRange(roi_hsv, hsv_range["lower"], hsv_range["upper"])
            liquid_mask = cv2.bitwise_or(liquid_mask, mask)
            
        # Operaciones morfológicas para limpiar la máscara
        morph_params = bottle_config.get("morphology", {"kernel_size": 5, "iterations": 2})
        kernel = np.ones((morph_params["kernel_size"], morph_params["kernel_size"]), np.uint8)
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_OPEN, kernel, iterations=morph_params["iterations"])
        liquid_mask = cv2.morphologyEx(liquid_mask, cv2.MORPH_CLOSE, kernel, iterations=morph_params["iterations"])
        
        # Calcular nivel de líquido mediante proyección vertical
        h, w = liquid_mask.shape
        vertical_projection = np.sum(liquid_mask, axis=1)
        
        # Normalizar la proyección
        max_proj = np.max(vertical_projection)
        if max_proj > 0:
            vertical_projection = vertical_projection / max_proj
            
        # Umbral para determinar presencia de líquido
        liquid_threshold = w * 0.25
        
        # Encontrar el nivel más alto con suficiente líquido
        liquid_line = None
        for y in range(h):
            if vertical_projection[y] > liquid_threshold / w:
                liquid_line = y
                break
                
        # Calcular porcentaje de nivel
        level_percentage = 0
        
        if liquid_line is not None:
            # Nivel básico (porcentaje desde abajo)
            raw_level = 100 - (liquid_line / h * 100)
            
            # Compensación para el cuello de la botella
            neck_detection = bottle_config.get("neck_detection", {"enabled": False})
            
            if neck_detection.get("enabled", False):
                # Ajustar cálculo para considerar que el cuello no se llena
                top_ratio = neck_detection.get("top_section_ratio", 0)
                fillable_height = h * (1 - top_ratio)
                
                if liquid_line < h * top_ratio:
                    # Si detectamos líquido en el cuello (probablemente un error)
                    level_percentage = 100
                else:
                    # Calcular porcentaje del área llenable
                    adjusted_line = liquid_line - (h * top_ratio)
                    level_percentage = 100 - (adjusted_line / fillable_height * 100)
                    level_percentage = min(100, level_percentage)
            else:
                level_percentage = raw_level
                
            # Ajustes específicos por tipo de licor
            if liquor_type == "vodka" and level_percentage > 0:
                level_percentage = min(100, level_percentage * 1.15)
            elif liquor_type == "rum" and level_percentage > 0:
                level_percentage = max(0, level_percentage * 0.95)
                
        # Crear visualización
        visualization = image.copy()
        
        # Dibujar caja de botella detectada
        cv2.rectangle(visualization, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Añadir texto de confianza
        cv2.putText(visualization, f"Botella: {liquor_type}", 
                    (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Dibujar línea de nivel si se detectó
        if liquid_line is not None:
            level_y = y1 + liquid_line
            cv2.line(visualization, (x1, level_y), (x2, level_y), (0, 0, 255), 2)
            
            # Añadir texto de nivel
            cv2.putText(visualization, f"{level_percentage:.1f}%", 
                       (x2 + 5, level_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                       
        # Añadir barra de nivel
        h_img, w_img = image.shape[:2]
        bar_x = w_img - 50
        bar_y = (h_img - 300) // 2
        
        # Fondo de la barra
        cv2.rectangle(visualization, (bar_x, bar_y), (bar_x + 30, bar_y + 300), (50, 50, 50), -1)
        
        # Nivel de llenado
        filled_height = int(300 * level_percentage / 100)
        fill_y = bar_y + 300 - filled_height
        
        # Color según nivel
        if level_percentage < 30:
            color = (0, 0, 255)  # Rojo (bajo)
        elif level_percentage < 70:
            color = (0, 255, 255)  # Amarillo (medio)
        else:
            color = (0, 255, 0)  # Verde (alto)
            
        cv2.rectangle(visualization, (bar_x, fill_y), (bar_x + 30, bar_y + 300), color, -1)
        
        # Crear máscara visualización colorida
        mask_viz = np.zeros((h, w, 3), dtype=np.uint8)
        mask_viz[:,:,2] = liquid_mask  # Canal rojo para líquido
        
        # Marcar línea de nivel en la visualización de máscara
        if liquid_line is not None:
            cv2.line(mask_viz, (0, liquid_line), (w, liquid_line), (0, 255, 255), 2)
        
        return level_percentage, visualization, roi, mask_viz
        
    def process_image(self, image, liquor_type="whisky"):
        """
        Procesa una imagen completa: detecta botella y nivel de líquido
        
        Args:
            image: Imagen BGR
            liquor_type: Tipo de licor
        
        Returns:
            result_image: Imagen con anotaciones
            level_percentage: Nivel de líquido detectado (0-100)
            roi: Región de interés (botella recortada)
            mask_viz: Visualización de la máscara de líquido
        """
        # Medir tiempo de ejecución
        start_time = time.time()
        
        # Paso 1: Detectar botella con YOLO
        bottle_box, confidence = self.detect_bottle(image)
        
        # Si no se detecta ninguna botella, devolver imagen original
        if bottle_box is None:
            print("No se detectó ninguna botella")
            return image, 0, None, None
            
        # Paso 2: Analizar nivel de líquido
        level_percentage, result_image, roi, mask_viz = self.analyze_liquid_level(
            image, bottle_box, liquor_type)
            
        # Mostrar tiempo de procesamiento
        process_time = time.time() - start_time
        cv2.putText(result_image, f"Tiempo: {process_time:.3f}s", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                   
        return result_image, level_percentage, roi, mask_viz