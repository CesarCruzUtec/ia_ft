import Box from "@mui/material/Box";
import Button from "@mui/material/Button";
import CircularProgress from "@mui/material/CircularProgress";
import Dialog from "@mui/material/Dialog";
import DialogContent from "@mui/material/DialogContent";
import DialogTitle from "@mui/material/DialogTitle";
import Divider from "@mui/material/Divider";
import FormControl from "@mui/material/FormControl";
import InputLabel from "@mui/material/InputLabel";
import MenuItem from "@mui/material/MenuItem";
import Select from "@mui/material/Select";
import Stack from "@mui/material/Stack";
import Typography from "@mui/material/Typography";
import { useRef, useState } from "react";
import { getBoxes } from "./api";
import ImageOverlay from "./components/ImageOverlay";
import JSONTree from "./components/JSONTree";

export default function App() {
    const [imageSrc, setImageSrc] = useState(null);
    const [logs, setLogs] = useState([]);
    const inputRef = useRef(null);
    const prevUrlRef = useRef(null);
    const modelSelectId = "model-select";
    const [model, setModel] = useState("default_model");
    const [busy, setBusy] = useState(false);
    const [fileName, setFileName] = useState(null);
    const [detections, setDetections] = useState([]);
    const imgRef = useRef(null);
    const wrapperRef = useRef(null);
    const [imageLayout, setImageLayout] = useState(null);
    const [jsonModalOpen, setJsonModalOpen] = useState(false);
    const [selectedJson, setSelectedJson] = useState(null);

    function addLog(level, text) {
        const time = new Date().toLocaleTimeString();
        setLogs(s => [...s, { time, level, text }]);
    }

    // helper to identify long base64-like strings (kept for JSON rendering)

    // JSONNode moved to module bottom to avoid redefining on every render

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
        setFileName(f.name);
        setDetections([]);
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
        setFileName(null);
        setImageSrc(null);
        setDetections([]);
        if (inputRef.current) {
            inputRef.current.value = null;
        }
        addLog("warn", "Imagen limpiada por el usuario");
    }

    return (
        <>
            <div className="page">
                <div className="panel left">
                    <div className="left-top">
                        {imageSrc ? (
                            <div className="image-wrapper" ref={wrapperRef} style={{ position: "relative" }}>
                                <img
                                    ref={imgRef}
                                    onLoad={e => {
                                        const img = e.target;
                                        const wrapperRect = wrapperRef.current?.getBoundingClientRect();
                                        const wW = wrapperRect?.width || 0;
                                        const wH = wrapperRect?.height || 0;
                                        const nW = img.naturalWidth;
                                        const nH = img.naturalHeight;
                                        // compute displayed size when object-fit: contain is used
                                        const wrapperRatio = wW / wH || 1;
                                        const imgRatio = nW / nH || 1;
                                        let displayWidth, displayHeight, offsetLeft, offsetTop;
                                        if (imgRatio > wrapperRatio) {
                                            // image is relatively wider -> fit by width
                                            displayWidth = wW;
                                            const scale = wW / nW;
                                            displayHeight = Math.round(nH * scale);
                                            offsetLeft = 0;
                                            offsetTop = Math.round((wH - displayHeight) / 2);
                                        } else {
                                            // image is relatively taller -> fit by height
                                            displayHeight = wH;
                                            const scale = wH / nH;
                                            displayWidth = Math.round(nW * scale);
                                            offsetTop = 0;
                                            offsetLeft = Math.round((wW - displayWidth) / 2);
                                        }
                                        setImageLayout({
                                            naturalWidth: nW,
                                            naturalHeight: nH,
                                            displayWidth,
                                            displayHeight,
                                            offsetLeft,
                                            offsetTop,
                                        });
                                    }}
                                    src={imageSrc}
                                    alt="uploaded"
                                    className="image"
                                />
                                <ImageOverlay detections={detections} imageLayout={imageLayout} />
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
                                        <span className="log-time">[{l.time}]</span>{" "}
                                        {typeof l.text === "object" ? (
                                            <Button
                                                size="small"
                                                onClick={() => {
                                                    setSelectedJson(l.text);
                                                    setJsonModalOpen(true);
                                                }}
                                            >
                                                Ver JSON
                                            </Button>
                                        ) : (
                                            <span>{l.text}</span>
                                        )}
                                    </div>
                                ))
                            )}
                        </div>
                    </div>
                </div>

                <div className="panel right">
                    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                        <Typography variant="h6" component="div">
                            Controles
                        </Typography>
                        {busy && <CircularProgress size={18} />}
                    </div>

                    <Box sx={{ display: "flex", flexDirection: "column", gap: 2, mt: 1 }}>
                        {/* Section 1: Upload */}
                        <Box>
                            <Typography variant="subtitle1">1. Subir imagen</Typography>
                            <input
                                ref={inputRef}
                                type="file"
                                accept="image/*"
                                onChange={onFileChange}
                                style={{ display: "none" }}
                                aria-hidden={true}
                                tabIndex={-1}
                            />
                            <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                                <Button variant="contained" size="small" onClick={onUploadClick} disabled={busy}>
                                    Subir imagen
                                </Button>
                                <Button variant="outlined" size="small" onClick={onClear} disabled={!imageSrc || busy}>
                                    Limpiar
                                </Button>
                            </Stack>
                        </Box>

                        <Divider />

                        {/* Section 2: Obtener cuadros */}
                        <Box>
                            <Typography variant="subtitle1">2. Obtener cuadros</Typography>
                            <FormControl fullWidth size="small" sx={{ mt: 1 }}>
                                <InputLabel id="model-select-label">Modelo</InputLabel>
                                <Select
                                    labelId="model-select-label"
                                    id={modelSelectId}
                                    value={model}
                                    label="Modelo"
                                    onChange={e => setModel(e.target.value)}
                                >
                                    <MenuItem value="detection_model">Detección de brotes</MenuItem>
                                    <MenuItem value="model_a">Modelo A</MenuItem>
                                    <MenuItem value="model_b">Modelo B</MenuItem>
                                </Select>
                            </FormControl>
                            <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                                <Button
                                    variant="contained"
                                    size="small"
                                    onClick={async () => {
                                        if (!fileName || busy) return;
                                        setBusy(true);
                                        addLog("info", `Llamando getBoxes con model=${model}, image_name=${fileName}`);
                                        try {
                                            const res = await getBoxes({ model, image_name: fileName });
                                            // log raw response object so it can be rendered as JSON tree
                                            addLog("debug", res);
                                            // if backend returned an array of detections, store them so overlays render
                                            if (Array.isArray(res)) {
                                                setDetections(res);
                                            } else if (res && Array.isArray(res.detections)) {
                                                setDetections(res.detections);
                                            }
                                        } catch (err) {
                                            const msg = err?.body || err.message || String(err);
                                            addLog("error", `Error getBoxes: ${msg}`);
                                        } finally {
                                            setBusy(false);
                                        }
                                    }}
                                    disabled={!imageSrc || busy}
                                >
                                    Obtener
                                </Button>
                            </Stack>
                        </Box>

                        <Divider />

                        {/* Section 3: Analizar (placeholder) */}
                        <Box>
                            <Typography variant="subtitle1">3. Analizar</Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                Aquí irá la UI para analizar (por ahora placeholder)
                            </Typography>
                            <Button variant="contained" size="small" disabled sx={{ mt: 1 }}>
                                Analizar (pendiente)
                            </Button>
                        </Box>

                        <Divider />

                        {/* Section 4: Medir (placeholder) */}
                        <Box>
                            <Typography variant="subtitle1">4. Medir</Typography>
                            <Typography variant="body2" color="text.secondary" sx={{ mt: 1 }}>
                                Aquí irá la UI para medir (por ahora placeholder)
                            </Typography>
                            <Button variant="contained" size="small" disabled sx={{ mt: 1 }}>
                                Medir (pendiente)
                            </Button>
                        </Box>

                        <Typography variant="caption" color="text.secondary">
                            La imagen se escala manteniendo su relación de aspecto (object-fit: contain)
                        </Typography>
                    </Box>
                </div>
            </div>
            <JSONModal
                open={jsonModalOpen}
                data={selectedJson}
                onClose={() => {
                    setJsonModalOpen(false);
                    setSelectedJson(null);
                }}
            />
        </>
    );
}

/* eslint-disable react/prop-types */
function JSONModal({ open, data, onClose }) {
    return (
        <Dialog open={open} onClose={onClose} maxWidth="md" fullWidth>
            <DialogTitle>JSON</DialogTitle>
            <DialogContent>{data ? <JSONTree data={data} /> : <div>No data</div>}</DialogContent>
        </Dialog>
    );
}
