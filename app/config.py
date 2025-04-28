import numpy as np

# Configuraciones mejoradas para diferentes tipos de licores
LIQUOR_CONFIGS = {
    # Para licores ámbar (whisky, brandy, ron añejo)
    "whisky": {
        "hsv_ranges": [
            # Rango principal para ámbar - ampliado para mejor detección
            {"lower": np.array([8, 35, 40]), "upper": np.array([35, 255, 255])},
            # Rango secundario para ámbar oscuro - mejorado para whiskies añejos
            {"lower": np.array([0, 40, 10]), "upper": np.array([18, 255, 180])},
            # Rango adicional para caramelo oscuro (común en whisky)
            {"lower": np.array([0, 30, 0]), "upper": np.array([20, 200, 100])},
        ],
        "threshold_params": {"blockSize": 17, "C": 6},
        "morphology": {"kernel_size": 5, "iterations": 2},
        "bottle_detection": {
            "min_area": 500,
            "aspect_ratio_bounds": [1.5, 8.0],
            "convexity_bounds": [0.75, 0.98],
        },
        # Nuevo: Porcentaje de cuello típicamente no llenado en estas botellas
        "neck_percentage": 15,
        "neck_detection": {
            "enabled": True,
            "top_section_ratio": 0.25  # El 25% superior se considera cuello
        }
    },
    
    # Para licores claros (vodka, gin, tequila blanco)
    "vodka": {
        "hsv_ranges": [
            # Rango principal para líquidos transparentes - ajustado para reflejos
            {"lower": np.array([0, 0, 140]), "upper": np.array([180, 50, 255])},
            # Rango adicional para detectar reflejos y brillos en vodka
            {"lower": np.array([0, 0, 200]), "upper": np.array([180, 30, 255])},
            # Rango para sombras suaves que indican presencia de líquido
            {"lower": np.array([0, 0, 100]), "upper": np.array([180, 40, 200])},
        ],
        "threshold_params": {"blockSize": 13, "C": 3},
        "morphology": {"kernel_size": 3, "iterations": 2},
        "bottle_detection": {
            "min_area": 500,
            "aspect_ratio_bounds": [2.0, 10.0],
            "convexity_bounds": [0.8, 0.99],
        },
        "neck_percentage": 10,
        "neck_detection": {
            "enabled": True,
            "top_section_ratio": 0.15
        }
    },
    
    # Para licores oscuros (ron oscuro, whisky añejo)
    "rum": {
        "hsv_ranges": [
            # Rango para licores muy oscuros - ampliado
            {"lower": np.array([0, 0, 0]), "upper": np.array([180, 255, 90])},
            # Rango para marrones oscuros - común en ron añejo
            {"lower": np.array([0, 40, 0]), "upper": np.array([25, 255, 130])},
            # Rango para detectar reflejos rojizos en ron
            {"lower": np.array([170, 30, 30]), "upper": np.array([180, 255, 150])},
            # Rango adicional para tonos caoba
            {"lower": np.array([10, 50, 20]), "upper": np.array([30, 255, 120])},
        ],
        "threshold_params": {"blockSize": 21, "C": 8},
        "morphology": {"kernel_size": 7, "iterations": 3},
        "bottle_detection": {
            "min_area": 500,
            "aspect_ratio_bounds": [1.5, 7.0],
            "convexity_bounds": [0.7, 0.98],
        },
        "neck_percentage": 12,
        "neck_detection": {
            "enabled": True,
            "top_section_ratio": 0.20
        }
    },
    
    # Para tequila (que puede ser claro o reposado)
    "tequila": {
        "hsv_ranges": [
            # Rango para tequila blanco - ajustado para mayor sensibilidad
            {"lower": np.array([0, 0, 140]), "upper": np.array([180, 60, 255])},
            # Rango ampliado para tequila reposado
            {"lower": np.array([12, 25, 50]), "upper": np.array([38, 255, 255])},
            # Rango adicional para tequila añejo (más oscuro)
            {"lower": np.array([5, 30, 30]), "upper": np.array([40, 200, 180])},
        ],
        "threshold_params": {"blockSize": 15, "C": 4},
        "morphology": {"kernel_size": 5, "iterations": 2},
        "bottle_detection": {
            "min_area": 500,
            "aspect_ratio_bounds": [0.8, 5.0],
            "convexity_bounds": [0.7, 0.99],
        },
        "neck_percentage": 8,
        "neck_detection": {
            "enabled": True,
            "top_section_ratio": 0.15
        }
    },
    
    # Categoría para Ginebra
    "gin": {
        "hsv_ranges": [
            # Rango principal para ginebra clara
            {"lower": np.array([0, 0, 150]), "upper": np.array([180, 50, 255])},
            # Rango para detectar tonos azulados (común en algunas ginebras)
            {"lower": np.array([90, 20, 150]), "upper": np.array([120, 255, 255])},
            # Rango para ginebras levemente teñidas
            {"lower": np.array([0, 0, 130]), "upper": np.array([180, 70, 255])},
        ],
        "threshold_params": {"blockSize": 11, "C": 3},
        "morphology": {"kernel_size": 3, "iterations": 2},
        "bottle_detection": {
            "min_area": 500,
            "aspect_ratio_bounds": [1.5, 8.0],
            "convexity_bounds": [0.8, 0.99],
        },
        "neck_percentage": 10,
        "neck_detection": {
            "enabled": True, 
            "top_section_ratio": 0.15
        }
    }
}

# Parámetros de visualización mejorados
VISUALIZATION = {
    "guide_width": 130,
    "guide_height": 320,
    "bar_width": 30,
    "bar_height": 300,
    "low_level_color": (0, 0, 255),    # Rojo si es bajo
    "mid_level_color": (0, 255, 255),  # Amarillo si es medio
    "high_level_color": (0, 255, 0),   # Verde si es alto
    "overlay_opacity": 0.7,            # Opacidad para overlay de visualización
    "highlight_detected_area": True,   # Resaltar área de líquido detectada
    "show_confidence_score": True      # Mostrar puntuación de confianza
}

# Umbrales de nivel
LEVEL_THRESHOLDS = {
    "low": 30,  # Menor a 30% se considera bajo
    "high": 70  # Mayor a 70% se considera alto
}

# Parámetros de procesamiento avanzados
PROCESSING_PARAMS = {
    "edge_detection": {
        "canny_low": 50,
        "canny_high": 150,
        "blur_kernel": 5,
        "use_adaptive_threshold": True
    },
    "liquid_detection": {
        "use_combined_method": True,  # Usar combinación de color y gradiente
        "gradient_weight": 0.3,       # Peso del análisis de gradiente
        "color_weight": 0.7,          # Peso del análisis de color
        "min_level_score": 10.0       # Puntuación mínima para considerar líquido
    }
}