// LiquorLevelDetector.tsx
import { useState, useRef } from 'react';

const liquorTypes = {
  whisky: { name: "Whisky/Bourbon" },
  vodka: { name: "Vodka/Gin" },
  tequila: { name: "Tequila" },
  ron: { name: "Ron" }
};

type LiquorType = keyof typeof liquorTypes;

const LiquorLevelDetector = () => {
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [processedImageUrl, setProcessedImageUrl] = useState<string | null>(null);
  const [liquidLevel, setLiquidLevel] = useState<number | null>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [selectedLiquor, setSelectedLiquor] = useState<LiquorType>("whisky");
  const [debugViews, setDebugViews] = useState<Record<string, string>>({});
  
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleImageUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      const url = URL.createObjectURL(file);
      setImageUrl(url);
      setProcessedImageUrl(null);
      setLiquidLevel(null);
      setDebugViews({});
    }
  };

  const handleLiquorTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    setSelectedLiquor(e.target.value as LiquorType);
    setProcessedImageUrl(null);
    setLiquidLevel(null);
    setDebugViews({});
  };

  const processImage = async () => {
    if (!imageUrl || !fileInputRef.current || !fileInputRef.current.files) return;
    
    setIsLoading(true);
    
    try {
      const file = fileInputRef.current.files[0];
      
      // Crear un FormData para enviar la imagen y el tipo de licor
      const formData = new FormData();
      formData.append('file', file);
      formData.append('liquor_type', selectedLiquor);
      
      // Enviar al backend de FastAPI
      const response = await fetch('http://localhost:8000/analyze/', {
        method: 'POST',
        body: formData,
      });
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      const result = await response.json();
      
      // Actualizar UI con los resultados
      setProcessedImageUrl(result.processed_image);
      setLiquidLevel(result.liquid_level);
      setDebugViews(result.debug_images || {});
      
    } catch (error) {
      console.error("Error processing image:", error);
      alert("Error procesando la imagen. Por favor intente de nuevo.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="liquor-level-detector">
      <h2>Detector de Nivel de Líquido</h2>
      
      <div className="upload-section">
        <input 
          type="file" 
          ref={fileInputRef} 
          accept="image/*" 
          onChange={handleImageUpload}
          className="file-input" 
        />
        <button 
          onClick={() => fileInputRef.current?.click()}
          className="upload-button"
        >
          Seleccionar Imagen
        </button>
      </div>

      {imageUrl && (
        <>
          <div className="settings-panel">
            <div className="liquor-type-selector">
              <label htmlFor="liquor-type">Tipo de Licor:</label>
              <select 
                id="liquor-type" 
                value={selectedLiquor}
                onChange={handleLiquorTypeChange}
              >
                {Object.entries(liquorTypes).map(([key, value]) => (
                  <option key={key} value={key}>{value.name}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="image-preview">
            <img 
              src={imageUrl} 
              alt="Imagen de botella" 
              style={{ maxWidth: '300px', marginTop: '20px' }} 
            />
            
            <button 
              onClick={processImage} 
              disabled={isLoading}
              className="process-button"
            >
              {isLoading ? 'Procesando...' : 'Detectar Nivel de Líquido'}
            </button>
          </div>
        </>
      )}

      {isLoading && <p>Procesando imagen en el servidor...</p>}

      {processedImageUrl && (
        <div className="results">
          <h3>Resultado del Análisis</h3>
          <div className="result-image">
            <img src={processedImageUrl} alt="Imagen procesada" style={{ maxWidth: '300px' }} />
          </div>
          {liquidLevel !== null && (
            <div className="level-info">
              <p>Nivel de líquido estimado: <strong>{liquidLevel}%</strong></p>
              <div className="level-indicator">
                <div 
                  className="level-fill" 
                  style={{ height: `${liquidLevel}%` }}
                ></div>
              </div>
            </div>
          )}

          {Object.keys(debugViews).length > 0 && (
            <div className="debug-views">
              <h4>Pasos del proceso</h4>
              <div className="debug-grid">
                {Object.entries(debugViews).map(([name, url]) => (
                  <div key={name} className="debug-item">
                    <h5>{name}</h5>
                    <img src={url} alt={`Debug: ${name}`} style={{ maxWidth: '150px' }} />
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default LiquorLevelDetector;