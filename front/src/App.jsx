import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
    const [selectedImage, setSelectedImage] = useState(null);
    const [selectedImageName, setSelectedImageName] = useState("");
    const [selectedModel, setSelectedModel] = useState("sam2.1_hiera_tiny");
    const [logs, setLogs] = useState([]);
    const [points, setPoints] = useState([]);
    const [imageDimensions, setImageDimensions] = useState({ width: 0, height: 0 });
    const [masks, setMasks] = useState([]); // [{key, score, mask_base64}]
    const [selectedMaskKey, setSelectedMaskKey] = useState(null);
    const imageRef = useRef(null);
    const containerRef = useRef(null);
    const logsEndRef = useRef(null);

    useEffect(() => {
        if (logsEndRef.current) {
            logsEndRef.current.scrollIntoView({ behavior: "smooth" });
        }
    }, [logs]);

    const addLog = message => {
        const timestamp = new Date().toLocaleTimeString();
        setLogs(prevLogs => [...prevLogs, { message, timestamp, id: Date.now() }]);
    };

    const handleImageUpload = event => {
        const file = event.target.files[0];
        if (file && file.type.startsWith("image/")) {
            setSelectedImageName(file.name);
            const reader = new FileReader();
            reader.onload = e => {
                const img = new window.Image();
                img.onload = () => {
                    setImageDimensions({ width: img.width, height: img.height });
                    setSelectedImage(e.target.result);
                    setPoints([]);
                    addLog(
                        `Imagen cargada: ${file.name} (${(file.size / 1024).toFixed(2)} KB) - Dimensiones: ${
                            img.width
                        }x${img.height}px`
                    );
                };
                img.src = e.target.result;
            };
            reader.readAsDataURL(file);
        }
        event.target.value = "";
    };

    const handleButtonClick = () => {
        document.getElementById("imageInput").click();
    };

    const handleRemoveImage = () => {
        setSelectedImage(null);
        setSelectedImageName("");
        setPoints([]);
        setMasks([]);
        setSelectedMaskKey(null);
        addLog("Imagen eliminada");
        const input = document.getElementById("imageInput");
        if (input) input.value = "";
    };

    const handleClearPoints = () => {
        setPoints([]);
        addLog("Puntos limpiados");
    };

    const handleImageClick = event => {
        if (!imageRef.current) return;
        const containerRect = imageRef.current.parentElement.getBoundingClientRect();
        const containerW = containerRect.width;
        const containerH = containerRect.height;
        const imgW = imageDimensions.width;
        const imgH = imageDimensions.height;
        const scale = Math.min(containerW / imgW, containerH / imgH);
        const scaledW = imgW * scale;
        const scaledH = imgH * scale;
        const offsetX = (containerW - scaledW) / 2;
        const offsetY = (containerH - scaledH) / 2;
        const relX = event.clientX - containerRect.left;
        const relY = event.clientY - containerRect.top;
        if (relX < offsetX || relX > offsetX + scaledW || relY < offsetY || relY > offsetY + scaledH) return;
        const imgX = relX - offsetX;
        const imgY = scaledH - (relY - offsetY); // origen abajo-izquierda
        if (event.button === 1) {
            // Eliminar punto cercano
            const threshold = 14;
            const idx = points.findIndex(
                p => Math.abs(p.x - imgX) < threshold && Math.abs(p.y - (scaledH - imgY)) < threshold
            );
            if (idx !== -1) {
                const removed = points[idx];
                setPoints(points.filter((_, i) => i !== idx));
                addLog(`Punto eliminado (${removed.type}): (${removed.realX}, ${removed.realY})`);
            }
            return;
        }
        if (event.button === 0 || event.button === 2) {
            const type = event.button === 0 ? "positive" : "negative";
            const realX = Math.round(imgX / scale);
            // Y desde arriba: realY = Math.round((relY - offsetY) / scale)
            const realY = Math.round((relY - offsetY) / scale);
            setPoints([
                ...points,
                {
                    id: Date.now(),
                    x: imgX,
                    y: scaledH - imgY,
                    realX,
                    realY,
                    type,
                },
            ]);
            addLog(`Punto ${type === "positive" ? "verde" : "rojo"}: (${realX}, ${realY})`);
        }
    };

    const handleContextMenu = e => {
        e.preventDefault();
    };

    const handleClearLogs = () => {
        setLogs([]);
    };

    const handleAnalyze = async (realtime = false) => {
        if (!selectedImage) return;
        const positives = points.filter(p => p.type === "positive").map(p => ({ x: p.realX, y: p.realY }));
        const negatives = points.filter(p => p.type === "negative").map(p => ({ x: p.realX, y: p.realY }));
        const payload = {
            model: selectedModel,
            image_path: selectedImageName,
            points: { positive: positives, negative: negatives },
            realtime,
        };
        addLog(`Enviando petición: ${JSON.stringify(payload)}`);
        try {
            const response = await fetch("http://localhost:8000/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            const data = await response.json();
            // Parse masks from backend
            const maskArr = Object.entries(data)
                .filter(([k, v]) => v && v.mask_base64)
                .map(([k, v]) => ({ key: k, score: v.score, mask_base64: v.mask_base64 }));
            setMasks(maskArr);
            setSelectedMaskKey(maskArr.length > 0 ? maskArr[0].key : null);
            // Truncate deep log
            const truncateDeep = (obj, maxLength = 50) => {
                if (typeof obj === "string") {
                    return obj.length > maxLength ? obj.substring(0, maxLength) + "..." : obj;
                }
                if (Array.isArray(obj)) {
                    return obj.map(item => truncateDeep(item, maxLength));
                }
                if (obj && typeof obj === "object") {
                    const result = {};
                    for (const [key, value] of Object.entries(obj)) {
                        result[key] = truncateDeep(value, maxLength);
                    }
                    return result;
                }
                return obj;
            };
            const truncatedData = JSON.stringify(truncateDeep(data));
            addLog(`Respuesta del backend: ${truncatedData}`);
        } catch (err) {
            addLog(`Error al llamar backend: ${err.message}`);
        }
    };

    return (
        <div className="app-container">
            <h1>Cargador de Imágenes</h1>
            <div className="main-content">
                <div className="image-display-area" ref={containerRef}>
                    {selectedImage ? (
                        <div
                            className="image-container"
                            onMouseDown={handleImageClick}
                            onContextMenu={handleContextMenu}
                            role="presentation"
                            tabIndex={-1}
                        >
                            <img ref={imageRef} src={selectedImage} alt="Imagen cargada" className="uploaded-image" />
                            {/* Overlay mask if selected */}
                            {(() => {
                                const mask = masks.find(m => m.key === selectedMaskKey);
                                if (mask) {
                                    // Use same scaling/position as image
                                    const container = imageRef.current ? imageRef.current.parentElement : null;
                                    const containerW = container ? container.offsetWidth : 0;
                                    const containerH = container ? container.offsetHeight : 0;
                                    const imgW = imageDimensions.width;
                                    const imgH = imageDimensions.height;
                                    const scale = Math.min(containerW / imgW, containerH / imgH);
                                    const scaledW = imgW * scale;
                                    const scaledH = imgH * scale;
                                    const offsetX = (containerW - scaledW) / 2;
                                    const offsetY = (containerH - scaledH) / 2;
                                    return (
                                        <img
                                            src={mask.mask_base64}
                                            alt="Máscara"
                                            style={{
                                                position: 'absolute',
                                                left: offsetX,
                                                top: offsetY,
                                                width: scaledW,
                                                height: scaledH,
                                                pointerEvents: 'none',
                                                zIndex: 5,
                                            }}
                                        />
                                    );
                                }
                                return null;
                            })()}
                            {/* ...existing points rendering... */}
                            {points.map(point =>
                                (() => {
                                    // Calculate offsets and scale as in click handler
                                    const container = imageRef.current ? imageRef.current.parentElement : null;
                                    const containerW = container ? container.offsetWidth : 0;
                                    const containerH = container ? container.offsetHeight : 0;
                                    const imgW = imageDimensions.width;
                                    const imgH = imageDimensions.height;
                                    const scale = Math.min(containerW / imgW, containerH / imgH);
                                    const scaledW = imgW * scale;
                                    const scaledH = imgH * scale;
                                    const offsetX = (containerW - scaledW) / 2;
                                    const offsetY = (containerH - scaledH) / 2;
                                    return (
                                        <div
                                            key={point.id}
                                            className={`point ${point.type}`}
                                            style={{
                                                left: `${offsetX + point.x}px`,
                                                top: `${offsetY + point.y}px`,
                                            }}
                                        />
                                    );
                                })()
                            )}
                        </div>
                    ) : (
                        <div className="placeholder">
                            <p>No hay imagen cargada</p>
                            <p className="hint">Usa el panel de la derecha para cargar una imagen</p>
                        </div>
                    )}
                </div>
                <div className="actions-panel">
                    <h2>Acciones</h2>
                    <div className="model-selector-section">
                        <label htmlFor="modelSelect" className="selector-label">
                            Modelo:
                        </label>
                        <select
                            id="modelSelect"
                            value={selectedModel}
                            onChange={e => setSelectedModel(e.target.value)}
                            className="model-select"
                        >
                            <option value="sam2.1_hiera_tiny">Tiny</option>
                            <option value="sam2.1_hiera_small">Small</option>
                            <option value="sam2.1_hiera_base_plus">Base Plus</option>
                            <option value="sam2.1_hiera_large">Large</option>
                        </select>
                    </div>
                    {masks.length > 0 && (
                        <div className="mask-selector-section">
                            <label className="selector-label">Máscara:</label>
                            <div className="mask-radio-group">
                                {masks.map(mask => (
                                    <label key={mask.key} className="mask-radio-label">
                                        <input
                                            type="radio"
                                            name="mask"
                                            value={mask.key}
                                            checked={selectedMaskKey === mask.key}
                                            onChange={() => setSelectedMaskKey(mask.key)}
                                        />
                                        {mask.key} (score: {mask.score.toFixed(3)})
                                    </label>
                                ))}
                            </div>
                        </div>
                    )}
                    <div className="actions-buttons">
                        <input
                            type="file"
                            id="imageInput"
                            accept="image/*"
                            onChange={handleImageUpload}
                            style={{ display: "none" }}
                        />
                        <button onClick={handleButtonClick} className="action-button add-button">
                            <span className="button-icon" aria-hidden="true">
                                📁
                            </span>{" "}
                            Añadir Imagen
                        </button>
                        <button
                            onClick={handleRemoveImage}
                            className="action-button remove-button"
                            disabled={!selectedImage}
                        >
                            <span className="button-icon" aria-hidden="true">
                                🗑️
                            </span>{" "}
                            Quitar Imagen
                        </button>
                        <button
                            onClick={handleClearPoints}
                            className="action-button clear-button"
                            disabled={!selectedImage || points.length === 0}
                        >
                            <span className="button-icon" aria-hidden="true">
                                🧹
                            </span>{" "}
                            Limpiar Puntos
                        </button>
                        <button
                            onClick={() => handleAnalyze(false)}
                            className="action-button analyze-button"
                            disabled={!selectedImage}
                        >
                            Analizar
                        </button>
                        <button
                            onClick={() => handleAnalyze(true)}
                            className="action-button analyze-realtime-button"
                            disabled={!selectedImage}
                        >
                            Analizar en tiempo real
                        </button>
                    </div>
                </div>
            </div>
            <div className="logs-section">
                <h3 style={{ display: "flex", alignItems: "center", justifyContent: "space-between", gap: "0.5rem" }}>
                    <span>Logs</span>
                    <button
                        className="action-button clear-button"
                        style={{
                            padding: "2px 10px",
                            fontSize: "13px",
                            marginLeft: 0,
                            minWidth: "auto",
                            height: "28px",
                            width: "auto",
                            flex: "none",
                        }}
                        onClick={handleClearLogs}
                        disabled={logs.length === 0}
                    >
                        Limpiar Logs
                    </button>
                </h3>
                <div className="logs-container">
                    {logs.length === 0 ? (
                        <p className="no-logs">No hay actividad registrada</p>
                    ) : (
                        logs.map(log => (
                            <div key={log.id} className="log-entry">
                                <span className="log-timestamp">[{log.timestamp}]</span>
                                <span className="log-message">{log.message}</span>
                            </div>
                        ))
                    )}
                    <div ref={logsEndRef} />
                </div>
            </div>
        </div>
    );
}

export default App;
