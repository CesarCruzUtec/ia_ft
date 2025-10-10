/* eslint-disable react/prop-types */
import { useState } from "react";

// minimal JSON tree renderer
function isLikelyBase64(str) {
    return typeof str === "string" && str.length > 15 && /^[A-Za-z0-9+/=\n\r]+$/.test(str);
}

export default function JSONTree({ data, rootKey }) {
    return (
        <div className="json-node-root">
            {typeof data === "object" ? <JSONNode k={rootKey} v={data} /> : String(data)}
        </div>
    );
}

function JSONNode({ k, v, depth = 0 }) {
    const [open, setOpen] = useState(false);
    const isObject = v && typeof v === "object" && !Array.isArray(v);
    const isArray = Array.isArray(v);

    if (isObject || isArray) {
        const entries = isArray ? v : Object.entries(v);
        return (
            <div className="json-node" style={{ marginLeft: depth * 12 }}>
                <button
                    className="json-key"
                    onClick={() => setOpen(!open)}
                    style={{ cursor: "pointer", background: "transparent", border: "none", padding: 0 }}
                >
                    {k !== undefined ? <strong>{k}: </strong> : null}
                    <em>{isArray ? `[${v.length}]` : "{...}"}</em>
                </button>
                {open && (
                    <div className="json-children">
                        {isArray
                            ? v.map(item => {
                                  const childKey = JSON.stringify(item).slice(0, 20);
                                  return <JSONNode key={childKey} k={childKey} v={item} depth={depth + 1} />;
                              })
                            : entries.map(([kk, vv]) => <JSONNode key={kk} k={kk} v={vv} depth={depth + 1} />)}
                    </div>
                )}
            </div>
        );
    }

    const display = typeof v === "string" && isLikelyBase64(v) ? v.slice(0, 15) + "..." : String(v);
    return (
        <div className="json-primitive" style={{ marginLeft: depth * 12 }}>
            {k !== undefined ? <span className="json-key">{k}: </span> : null}
            <span className="json-value" title={String(v)}>
                {display}
            </span>
        </div>
    );
}
