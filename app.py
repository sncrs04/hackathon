import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

CSV_PATH = Path("scramble-data.csv")
_df: pd.DataFrame | None = None


def load_data() -> pd.DataFrame:
    global _df
    if _df is None:
        _df = pd.read_csv(CSV_PATH, encoding="utf-8")
    return _df


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/stats")
def api_stats():
    df = load_data()
    return jsonify({
        "total": len(df),
        "pending": int((df["status"] == "pending").sum()),
        "completed": int((df["status"] == "completed").sum()),
        "rejected": int((df["status"] == "rejected").sum()),
        "payment_required": int((df["payment_service"] == 1).sum()),
    })


@app.route("/api/records")
def api_records():
    df = load_data()
    page = max(1, int(request.args.get("page", 1)))
    per_page = min(100, max(5, int(request.args.get("per_page", 20))))
    search = request.args.get("search", "").strip()
    status_filter = request.args.get("status", "")
    org_filter = request.args.get("org", "")

    data = df

    if status_filter:
        data = data[data["status"] == status_filter]
    if org_filter:
        data = data[data["org_name"] == org_filter]
    if search:
        mask = (
            data["service_request_id"].astype(str).str.contains(search, case=False, na=False)
            | data["org_name"].astype(str).str.contains(search, case=False, na=False)
            | data["name"].astype(str).str.contains(search, case=False, na=False)
            | data["status"].astype(str).str.contains(search, case=False, na=False)
        )
        data = data[mask]

    total = len(data)
    start = (page - 1) * per_page
    rows = []
    for _, row in data.iloc[start: start + per_page].iterrows():
        rows.append({
            "id": int(row["service_request_id"]),
            "service_id": int(row["service_id"]),
            "name": str(row["name"]) if pd.notna(row["name"]) else "",
            "org_name": str(row["org_name"]) if pd.notna(row["org_name"]) else "",
            "payment_service": int(row["payment_service"]),
            "status": str(row["status"]),
            "created_at": str(row["created_at"]) if pd.notna(row["created_at"]) else "",
            "finish_time": str(row["finish_time"]) if pd.notna(row["finish_time"]) else "",
        })

    return jsonify({
        "records": rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "total_pages": max(1, (total + per_page - 1) // per_page),
    })


@app.route("/api/record/<int:record_id>")
def api_record(record_id):
    df = load_data()
    rows = df[df["service_request_id"] == record_id]
    if rows.empty:
        return jsonify({"error": "Not found"}), 404
    row = rows.iloc[0]

    pred_input = {
        "service_id": int(row["service_id"]),
        "name": str(row["name"]) if pd.notna(row["name"]) else "",
        "org_name": str(row["org_name"]) if pd.notna(row["org_name"]) else "",
        "payment_service": int(row["payment_service"]),
        "created_at": str(row["created_at"]) if pd.notna(row["created_at"]) else None,
        "birth_date": str(row["birth_date"]) if pd.notna(row["birth_date"]) else None,
    }

    try:
        from src.predictor import predict_service_outcome
        predicted, confidence = predict_service_outcome(pred_input)
    except Exception as exc:
        predicted, confidence = "unknown", 0.0

    nodes = []
    raw_nodes = row.get("nodes") if hasattr(row, "get") else row["nodes"]
    if pd.notna(raw_nodes):
        try:
            nodes = json.loads(raw_nodes)
        except Exception:
            nodes = []

    def safe(v):
        return str(v) if pd.notna(v) else ""

    return jsonify({
        "service_request_id": int(row["service_request_id"]),
        "user_id": safe(row["user_id"]),
        "service_id": int(row["service_id"]),
        "name": safe(row["name"]),
        "org_name": safe(row["org_name"]),
        "payment_service": int(row["payment_service"]),
        "status": safe(row["status"]),
        "created_at": safe(row["created_at"]),
        "finish_time": safe(row["finish_time"]),
        "response": safe(row["response"]),
        "first_name": safe(row["first_name"]),
        "middle_name": safe(row["middle_name"]),
        "last_name": safe(row["last_name"]),
        "phone_num": safe(row["phone_num"]),
        "birth_date": safe(row["birth_date"]),
        "nodes": nodes,
        "prediction": {"outcome": predicted, "confidence": confidence},
    })


@app.route("/api/charts/status")
def chart_status():
    df = load_data()
    counts = df["status"].value_counts()
    return jsonify({"labels": counts.index.tolist(), "values": [int(v) for v in counts.values]})


@app.route("/api/charts/timeline")
def chart_timeline():
    df = load_data()
    tmp = df.copy()
    tmp["_dt"] = pd.to_datetime(tmp["created_at"], errors="coerce")
    monthly = tmp.groupby(tmp["_dt"].dt.to_period("M")).size().sort_index()
    return jsonify({"labels": [str(p) for p in monthly.index], "values": [int(v) for v in monthly.values]})


@app.route("/api/charts/top_orgs")
def chart_top_orgs():
    df = load_data()
    top = df["org_name"].value_counts().head(10)
    return jsonify({"labels": top.index.tolist(), "values": [int(v) for v in top.values]})


@app.route("/api/orgs")
def api_orgs():
    df = load_data()
    orgs = sorted(df["org_name"].dropna().unique().tolist())
    return jsonify(orgs)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json()
    try:
        from src.predictor import predict_service_outcome
        outcome, confidence = predict_service_outcome(data)
        return jsonify({"outcome": outcome, "confidence": confidence})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    try:
        result = subprocess.run(
            [sys.executable, "train_model.py"],
            capture_output=True,
            text=True,
            timeout=600,
            cwd=str(Path(__file__).parent),
        )
        # Bust the cached model so the predictor reloads the freshly trained one.
        import src.predictor as _pred
        _pred._ARTIFACT = None
        return jsonify({
            "success": result.returncode == 0,
            "stdout": result.stdout[-4000:] if result.stdout else "",
            "stderr": result.stderr[-2000:] if result.stderr else "",
        })
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    load_data()
    print("\n  Dashboard running at http://localhost:5000\n")
    app.run(debug=False, port=5000)
