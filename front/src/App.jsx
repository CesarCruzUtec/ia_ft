import { useRef, useState } from 'react'

export default function App() {
  const [imageSrc, setImageSrc] = useState(null)
  const [logs, setLogs] = useState([])
  const inputRef = useRef(null)
  const prevUrlRef = useRef(null)

  function addLog(level, text) {
    const time = new Date().toLocaleTimeString()
    setLogs((s) => [...s, { time, level, text }])
  }

  function onFileChange(e) {
    const f = e.target.files && e.target.files[0]
    if (!f) return
    const url = URL.createObjectURL(f)
    // revoke previous url if any
    if (prevUrlRef.current) {
      try {
        URL.revokeObjectURL(prevUrlRef.current)
      } catch (err) {
        console.warn('Error revoking previous object URL', err)
      }
    }
    prevUrlRef.current = url
    setImageSrc(url)
    addLog('info', `Imagen seleccionada: ${f.name}`)
  }

  function onUploadClick() {
    inputRef.current?.click()
  }

  function onClear() {
    if (prevUrlRef.current) {
      try {
        URL.revokeObjectURL(prevUrlRef.current)
      } catch (err) {
        console.warn('Error revoking object URL on clear', err)
      }
      prevUrlRef.current = null
    }
    setImageSrc(null)
    if (inputRef.current) {
      inputRef.current.value = null
    }
    addLog('warn', 'Imagen limpiada por el usuario')
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
        <div className="controls">
          <input
            ref={inputRef}
            type="file"
            accept="image/*"
            onChange={onFileChange}
            style={{ display: 'none' }}
            aria-hidden={true}
            tabIndex={-1}
          />
          <button onClick={onUploadClick} className="btn">
            Subir imagen
          </button>
          <button onClick={onClear} className="btn secondary">
            Limpiar
          </button>
        </div>
        <p className="note">La imagen se escala manteniendo su relación de aspecto (object-fit: contain)</p>
      </div>
    </div>
  )
}
