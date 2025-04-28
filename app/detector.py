from YoloLiquorDetector import YoloLiquorDetector
import numpy as np
# Instanciar el detector YOLO a nivel global para reutilizarlo
yolo_detector = None

def detectar_botella_y_nivel(frame, guide_params=None, liquor_type="whisky"):
    """
    Detecta una botella y su nivel de líquido en un frame usando YOLOv8.
    
    Args:
        frame: Imagen de entrada
        guide_params: Parámetros opcionales para la guía (no usado con YOLO)
        liquor_type: Tipo de licor (whisky, vodka, rum, tequila, gin)
    
    Returns:
        frame: Imagen procesada con visualizaciones
        roi: Región de interés (botella recortada)
        combined_mask: Máscara combinada para visualización
        nivel_detectado: Nivel de líquido detectado (porcentaje)
    """
    # Validar entrada
    if frame is None or frame.size == 0:
        print("Error: frame vacío o inválido")
        return None, None, None, 0
        
    try:
        # Inicializar detector YOLO si no existe
        global yolo_detector
        if yolo_detector is None:
            yolo_detector = YoloLiquorDetector(model_path="D:/model/yolov8m.pt")
            
        # Procesar imagen con YOLO
        result_image, nivel_detectado, roi, mask_viz = yolo_detector.process_image(
            frame, liquor_type=liquor_type)
            
        # Si no hay ROI, crear una vacía para mantener compatibilidad
        if roi is None:
            roi = np.zeros((10, 10), dtype=np.uint8)
            mask_viz = np.zeros((10, 10, 3), dtype=np.uint8)
            
        return result_image, roi, mask_viz, nivel_detectado
        
    except Exception as e:
        print(f"Error en detectar_botella_y_nivel con YOLO: {str(e)}")
        import traceback
        traceback.print_exc()
        
        # Retornar valores predeterminados en caso de error
        return frame, np.zeros((10, 10), dtype=np.uint8), np.zeros((10, 10, 3), dtype=np.uint8), 0