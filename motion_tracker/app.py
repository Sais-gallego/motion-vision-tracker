"""Real-time Flask Web Dashboard for Motion Vision Tracker."""

import time
import cv2
from flask import Flask, Response, render_template_string, jsonify, request
from .engine import MotionTrackingEngine
from .simulator import SyntheticVideoGenerator

app = Flask(__name__)
engine = MotionTrackingEngine()
engine.setup_default_zones(640, 480)
generator = SyntheticVideoGenerator(640, 480, num_targets=3)

# Global view settings
settings = {
    "show_boxes": True,
    "show_flow": False,
    "show_trails": True,
    "show_heatmap": False,
    "show_zones": True
}

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Motion Vision Tracker | Sentinel Telemetry</title>
    <style>
        :root {
            --bg: #09090b;
            --card: #18181b;
            --border: #27272a;
            --primary: #06b6d4;
            --accent: #10b981;
            --danger: #ef4444;
            --text: #f4f4f5;
            --muted: #a1a1aa;
        }
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: var(--bg); color: var(--text); padding: 1.5rem;
            min-height: 100vh;
        }
        .container { max-width: 1080px; margin: 0 auto; }
        header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1.5rem; border-bottom: 1px solid var(--border); padding-bottom: 1rem; }
        h1 { font-size: 1.8rem; font-weight: 700; color: var(--primary); }
        .badge-live { background: var(--accent); color: black; font-weight: 700; padding: 0.25rem 0.75rem; border-radius: 999px; font-size: 0.75rem; }
        .main-grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
        @media (max-width: 800px) { .main-grid { grid-template-columns: 1fr; } }
        .video-box {
            background: black; border-radius: 12px; overflow: hidden; border: 1px solid var(--border);
            box-shadow: 0 4px 20px rgba(0,0,0,0.5);
        }
        .video-box img { width: 100%; height: auto; display: block; }
        .sidebar { display: flex; flex-direction: column; gap: 1rem; }
        .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.25rem; }
        .card h2 { font-size: 1rem; color: var(--muted); text-transform: uppercase; margin-bottom: 1rem; font-weight: 600; }
        .toggle-row { display: flex; justify-content: space-between; align-items: center; padding: 0.5rem 0; border-bottom: 1px solid rgba(255,255,255,0.05); }
        .toggle-row:last-child { border-bottom: none; }
        .switch { position: relative; display: inline-block; width: 44px; height: 24px; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #3f3f46; transition: .3s; border-radius: 24px; }
        .slider:before { position: absolute; content: ""; height: 18px; width: 18px; left: 3px; bottom: 3px; background-color: white; transition: .3s; border-radius: 50%; }
        input:checked + .slider { background-color: var(--primary); }
        input:checked + .slider:before { transform: translateX(20px); }
        .stat-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.75rem; }
        .stat-item { background: rgba(255,255,255,0.03); padding: 0.75rem; border-radius: 8px; text-align: center; }
        .stat-val { font-size: 1.4rem; font-weight: 800; color: var(--primary); }
        .stat-lbl { font-size: 0.75rem; color: var(--muted); margin-top: 0.2rem; }
        .alert-box { max-height: 140px; overflow-y: auto; font-family: monospace; font-size: 0.8rem; }
        .alert-entry { padding: 0.35rem 0; color: var(--danger); border-bottom: 1px dashed rgba(239,68,68,0.2); }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🛰️ Motion Vision Tracker</h1>
                <p style="color: var(--muted); font-size: 0.9rem;">Telemetría de Detección de Movimiento, Flujo Óptico y Zonas de Seguridad</p>
            </div>
            <div class="badge-live">● EN VIVO</div>
        </header>

        <div class="main-grid">
            <div class="video-box">
                <img src="/video_feed" alt="Live Camera Stream">
            </div>

            <div class="sidebar">
                <div class="card">
                    <h2>Controles de Visualización</h2>
                    <div class="toggle-row">
                        <span>Bounding Boxes</span>
                        <label class="switch"><input type="checkbox" id="tBoxes" checked onchange="updateToggle('show_boxes', this.checked)"><span class="slider"></span></label>
                    </div>
                    <div class="toggle-row">
                        <span>Estelas de Trayectoria</span>
                        <label class="switch"><input type="checkbox" id="tTrails" checked onchange="updateToggle('show_trails', this.checked)"><span class="slider"></span></label>
                    </div>
                    <div class="toggle-row">
                        <span>Zonas de Seguridad</span>
                        <label class="switch"><input type="checkbox" id="tZones" checked onchange="updateToggle('show_zones', this.checked)"><span class="slider"></span></label>
                    </div>
                    <div class="toggle-row">
                        <span>Flujo Óptico Denso (HSV)</span>
                        <label class="switch"><input type="checkbox" id="tFlow" onchange="updateToggle('show_flow', this.checked)"><span class="slider"></span></label>
                    </div>
                    <div class="toggle-row">
                        <span>Mapa Térmico Acumulativo</span>
                        <label class="switch"><input type="checkbox" id="tHeatmap" onchange="updateToggle('show_heatmap', this.checked)"><span class="slider"></span></label>
                    </div>
                </div>

                <div class="card">
                    <h2>Telemetría en Tiempo Real</h2>
                    <div class="stat-grid">
                        <div class="stat-item">
                            <div class="stat-val" id="statFps">--</div>
                            <div class="stat-lbl">FPS</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-val" id="statTargets">--</div>
                            <div class="stat-lbl">Objetos Activos</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-val" id="statActiveArea">--%</div>
                            <div class="stat-lbl">Área Activa</div>
                        </div>
                        <div class="stat-item">
                            <div class="stat-val" id="statFrame">0</div>
                            <div class="stat-lbl">Cuadros</div>
                        </div>
                    </div>
                </div>

                <div class="card">
                    <h2>Registro de Intrusiones</h2>
                    <div class="alert-box" id="alertLog">
                        <div style="color: var(--muted);">Sin alertas recientes.</div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function updateToggle(key, val) {
            await fetch('/api/settings', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ [key]: val })
            });
        }

        async function refreshStats() {
            try {
                const res = await fetch('/api/stats');
                const data = await res.json();
                document.getElementById('statFps').innerText = data.fps;
                document.getElementById('statTargets').innerText = data.tracked_count;
                document.getElementById('statActiveArea').innerText = (data.heatmap_stats.active_area_percent || 0) + '%';
                document.getElementById('statFrame').innerText = data.frame_count;

                if (data.alerts && data.alerts.length > 0) {
                    const log = document.getElementById('alertLog');
                    log.innerHTML = data.alerts.map(a => `<div class="alert-entry">⚠️ [${new Date(a.timestamp * 1000).toLocaleTimeString()}] Intrusión en ${a.zone_name} (ID: ${a.target_id})</div>`).join('');
                }
            } catch (e) {}
        }
        setInterval(refreshStats, 800);
    </script>
</body>
</html>
"""

def generate_frames():
    while True:
        frame = generator.next_frame()
        res = engine.process_frame(
            frame,
            show_boxes=settings["show_boxes"],
            show_flow=settings["show_flow"],
            show_trails=settings["show_trails"],
            show_heatmap=settings["show_heatmap"],
            show_zones=settings["show_zones"]
        )
        _, buffer = cv2.imencode(".jpg", res["composite_frame"], [cv2.IMWRITE_JPEG_QUALITY, 80])
        frame_bytes = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n")
        time.sleep(0.03)

@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/video_feed")
def video_feed():
    return Response(generate_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/api/settings", methods=["POST"])
def api_settings():
    data = request.get_json() or {}
    for k, v in data.items():
        if k in settings:
            settings[k] = bool(v)
    return jsonify(settings)

@app.route("/api/stats")
def api_stats():
    return jsonify({
        "fps": round(engine.fps, 1),
        "frame_count": engine.frame_count,
        "tracked_count": len(engine.trajectory_tracker.targets) if engine.trajectory_tracker else 0,
        "heatmap_stats": engine.heatmap.get_activity_stats() if engine.heatmap else {},
        "alerts": [
            {"zone_name": a.zone_name, "target_id": a.target_id, "timestamp": a.timestamp}
            for a in (engine.tripwire_engine.active_alerts if engine.tripwire_engine else [])
        ]
    })

def run_app(host="127.0.0.1", port=5055):
    print(f"[*] Starting Motion Vision Tracker Stream on http://{host}:{port}")
    app.run(host=host, port=port, debug=False)
