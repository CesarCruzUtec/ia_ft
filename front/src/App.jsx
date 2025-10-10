import { useRef, useState } from "react";

export default function App() {
    const [imageSrc, setImageSrc] = useState(null);
    const [logs, setLogs] = useState([]);
    const inputRef = useRef(null);
    const prevUrlRef = useRef(null);
    const modelSelectId = "model-select";

    function addLog(level, text) {
        const time = new Date().toLocaleTimeString();
        setLogs(s => [...s, { time, level, text }]);
    }

    function onFileChange(e) {
        const f = e.target.files && e.target.files[0];
        if (!f) return;
        const url = URL.createObjectURL(f);
        // revoke previous url if any
        if (prevUrlRef.current) {
            try {
                URL.revokeObjectURL(prevUrlRef.current);
            } catch (err) {
                console.warn("Error revoking previous object URL", err);
            }
        }
        prevUrlRef.current = url;
        setImageSrc(url);
        addLog("info", `Imagen seleccionada: ${f.name}`);
    }

    function onUploadClick() {
        inputRef.current?.click();
    }

    function onClear() {
        if (prevUrlRef.current) {
            try {
                URL.revokeObjectURL(prevUrlRef.current);
            } catch (err) {
                console.warn("Error revoking object URL on clear", err);
            }
            prevUrlRef.current = null;
        }
        setImageSrc(null);
        if (inputRef.current) {
            inputRef.current.value = null;
        }
        addLog("warn", "Imagen limpiada por el usuario");
    }

    return (
        <div className="page">
            <div className="panel left">
                <div className="left-top">
                    {imageSrc ? (
                        <div className="image-wrapper">
                            <img src={imageSrc} alt="uploaded" className="image" />
                        </div>
                    ) : (
                        <div className="placeholder">Sube una imagen usando el panel derecho</div>
                    )}
                </div>

                <div className="left-bottom">
                    <div className="logs" role="log" aria-live="polite">
                        {logs.length === 0 ? (
                            <div className="log-empty">No hay logs todavía.</div>
                        ) : (
                            logs.map((l, i) => (
                                <div key={`${l.time}-${i}`} className={`log-line log-${l.level}`}>
                                    <span className="log-time">[{l.time}]</span> {l.text}
                                </div>
                            ))
                        )}
                    </div>
                </div>
            </div>

            <div className="panel right">
                <h2>Controles</h2>

                {/* Section 1: Upload */}
                <div className="controls section">
                    <h3>1. Subir imagen</h3>
                    <input
                        ref={inputRef}
                        type="file"
                        accept="image/*"
                        onChange={onFileChange}
                        style={{ display: "none" }}
                        aria-hidden={true}
                        tabIndex={-1}
                    />
                    <div style={{ display: "flex", gap: 8 }}>
                        <button onClick={onUploadClick} className="btn">
                            Subir imagen
                        </button>
                        <button onClick={onClear} className="btn secondary">
                            Limpiar
                        </button>
                    </div>
                </div>

                {/* Section 2: Obtener cuadros */}
                <div className="controls section">
                    <h3>2. Obtener cuadros</h3>
                    <label htmlFor={modelSelectId} style={{ fontSize: 13, color: "#555" }}>
                        Modelo
                    </label>
                    <select id={modelSelectId} className="model-select" defaultValue={"default_model"}>
                        <option value="default_model">default_model</option>
                        <option value="model_a">model_a</option>
                        <option value="model_b">model_b</option>
                    </select>
                    <div style={{ display: "flex", gap: 8 }}>
                        <button
                            className="btn"
                            onClick={() => addLog("info", "Obteniendo cuadros con el modelo seleccionado...")}
                        >
                            Obtener
                        </button>
                    </div>
                </div>

                {/* Section 3: Analizar (placeholder) */}
                <div className="controls section">
                    <h3>3. Analizar</h3>
                    <p className="note">Aquí irá la UI para analizar (por ahora placeholder)</p>
                    <button className="btn" disabled>
                        Analizar (pendiente)
                    </button>
                </div>

                {/* Section 4: Medir (placeholder) */}
                <div className="controls section">
                    <h3>4. Medir</h3>
                    <p className="note">Aquí irá la UI para medir (por ahora placeholder)</p>
                    <button className="btn" disabled>
                        Medir (pendiente)
                    </button>
                </div>

                <p className="note">La imagen se escala manteniendo su relación de aspecto (object-fit: contain)</p>
            </div>
        </div>
    );
}
