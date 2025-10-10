/* eslint-disable react/prop-types */

export default function ImageOverlay({ detections = [], imageLayout }) {
    if (!imageLayout || !Array.isArray(detections)) return null;

    const scaleX = (imageLayout.displayWidth || imageLayout.width) / imageLayout.naturalWidth;
    const scaleY = (imageLayout.displayHeight || imageLayout.height) / imageLayout.naturalHeight;
    const offsetLeft = imageLayout.offsetLeft || 0;
    const offsetTop = imageLayout.offsetTop || 0;

    return (
        <div className="image-overlay" aria-hidden>
            {detections.map((d, i) => {
                const x1 = d.x1 ?? d[0] ?? 0;
                const y1 = d.y1 ?? d[1] ?? 0;
                const x2 = d.x2 ?? d[2] ?? 0;
                const y2 = d.y2 ?? d[3] ?? 0;
                const left = offsetLeft + x1 * scaleX;
                const top = offsetTop + y1 * scaleY;
                const width = (x2 - x1) * scaleX;
                const height = (y2 - y1) * scaleY;
                const key = `${d.label || "box"}_${i}_${Math.round(x1)}_${Math.round(y1)}`;
                const mask = d.mask_base64;
                let maskSrc = null;
                if (mask) {
                    if (mask.startsWith && mask.startsWith("data:")) maskSrc = mask;
                    else maskSrc = `data:image/png;base64,${mask}`;
                }

                // compute full-image overlay placement (masks come in image-sized)
                const imgLeft = offsetLeft;
                const imgTop = offsetTop;
                const imgWidth = imageLayout.displayWidth || imageLayout.width || 0;
                const imgHeight = imageLayout.displayHeight || imageLayout.height || 0;

                return (
                    <div key={key} style={{ position: "absolute", left: 0, top: 0, pointerEvents: "none" }}>
                        {maskSrc ? (
                            <img
                                className="overlay-mask"
                                src={maskSrc}
                                alt={d.label || "mask"}
                                style={{
                                    left: `${imgLeft}px`,
                                    top: `${imgTop}px`,
                                    width: `${imgWidth}px`,
                                    height: `${imgHeight}px`,
                                    position: "absolute",
                                }}
                            />
                        ) : null}
                        <div
                            className="overlay-rect"
                            style={{ left: `${left}px`, top: `${top}px`, width: `${width}px`, height: `${height}px` }}
                        />
                    </div>
                );
            })}
        </div>
    );
}
