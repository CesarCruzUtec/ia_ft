const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:8000";

/**
 * @typedef {Object} Detection
 * @property {string} label
 * @property {number} confidence
 * @property {number} x1
 * @property {number} y1
 * @property {number} x2
 * @property {number} y2
 *
 * @typedef {Detection & { mask_base64: string, score: number }} AnalyzedDetection
 */

async function postJson(path, body, fetchOpts = {}) {
    const url = `${API_BASE}${path}`;
    const res = await fetch(url, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        ...fetchOpts,
    });

    if (!res.ok) {
        const text = await res.text().catch(() => "");
        const err = new Error(`Request failed ${res.status} ${res.statusText}`);
        err.status = res.status;
        err.body = text;
        throw err;
    }

    // try parse json, but return text if not json
    const ct = res.headers.get("content-type") || "";
    if (ct.includes("application/json")) return res.json();
    return res.text();
}

/**
 * Llama al endpoint POST /get_boxes
 * @param {{model:string, image_name:string}} params
 * @returns {Promise<Detection[]>} Lista de detecciones: [{label, confidence, x1, y1, x2, y2}]
 */
export async function getBoxes(params) {
    return postJson("/get_boxes", params);
}

/**
 * Llama al endpoint POST /analyze
 * @param {{model:string, image_name:string, boxes:Detection[]}} params
 * @returns {Promise<AnalyzedDetection[]>} Lista de detecciones extendida con mask_base64 y score
 */
export async function analyze(params) {
    return postJson("/analyze", params);
}

/**
 * Llama al endpoint POST /measure
 * @param {{image_name:string, boxes:AnalyzedDetection[]}} params
 * @returns {Promise<any>} La forma exacta de la respuesta de measure aún no está definida
 */
export async function measure(params) {
    return postJson("/measure", params);
}

export default { API_BASE, getBoxes, analyze, measure };
