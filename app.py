import math
import os
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


# =========================
# CONFIG
# =========================
PASSI_ABACO = [500, 625, 750, 875, 1000]

COL_B = "B [mm]"
COL_H = "H [mm]"
COL_PASSO = "PASSI [mm]"

VALUE_COLS = [
    "Ke,T [N/mm]",
    "Fy,T [N]",
    "δy,T [mm]",
    "δu,T [mm]",
    "Ke,C [N/mm]",
    "Fy,C [N]",
    "δy,C [mm]",
    "δu,C [mm]",
]

IDW_POWER = 2.0
ABACO_XLSX_PATH = os.getenv("ABACO_XLSX_PATH", "Abaco.xlsx")


MAP = {
    "Ke_T": "Ke,T [N/mm]",
    "Fy_T": "Fy,T [N]",
    "dy_T": "δy,T [mm]",
    "du_T": "δu,T [mm]",
    "Ke_C": "Ke,C [N/mm]",
    "Fy_C": "Fy,C [N]",
    "dy_C": "δy,C [mm]",
    "du_C": "δu,C [mm]",
}


# =========================
# FASTAPI
# =========================
app = FastAPI(title="Tamponamenti Abaco API", version="1.0.0")

DF_ABACO: Optional[pd.DataFrame] = None


# =========================
# Pydantic models (minimi)
# =========================
class PanelIn(BaseModel):
    id: str
    B_mm: float
    H_mm: float
    avg_passo_mm: float
    ratio: float = Field(default=1.0)

    # campi extra presenti nel tuo JSON: li accettiamo senza usarli
    keq_hole_N_per_mm: Optional[float] = None
    keq_full_N_per_mm: Optional[float] = None
    avg_passo_x_mm: Optional[float] = None
    avg_passo_y_mm: Optional[float] = None
    Aeq_univoca_mm2: Optional[float] = None
    Aeq_univoca_full_mm2: Optional[float] = None
    ratio_open_full_univoco: Optional[float] = None


class BodyIn(BaseModel):
    panels: List[PanelIn]


class RootItemIn(BaseModel):
    body: BodyIn


# il tuo payload è una LISTA di RootItemIn
PayloadIn = List[RootItemIn]


# =========================
# Helpers: abaco load
# =========================
def load_abaco(path: str) -> pd.DataFrame:
    df_raw = pd.read_excel(path)
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    needed = [COL_B, COL_H, COL_PASSO] + VALUE_COLS
    missing = [c for c in needed if c not in df_raw.columns]
    if missing:
        raise RuntimeError(
            f"Abaco.xlsx missing columns: {missing}. Found: {list(df_raw.columns)}"
        )

    df = df_raw.copy()

    for c in needed:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # drop righe sporche
    df = df.dropna(subset=[COL_B, COL_H, COL_PASSO])
    df = df.dropna(subset=VALUE_COLS)

    # filtra passi ammessi
    df = df[df[COL_PASSO].isin([float(p) for p in PASSI_ABACO])].copy()

    # cast
    df[COL_B] = df[COL_B].astype(float)
    df[COL_H] = df[COL_H].astype(float)
    df[COL_PASSO] = df[COL_PASSO].astype(float)
    for c in VALUE_COLS:
        df[c] = df[c].astype(float)

    return df


@app.on_event("startup")
def startup():
    global DF_ABACO
    if not os.path.exists(ABACO_XLSX_PATH):
        raise RuntimeError(f"Cannot find {ABACO_XLSX_PATH} in working dir.")
    DF_ABACO = load_abaco(ABACO_XLSX_PATH)


# =========================
# Helpers: interpolation
# =========================
def find_bracketing_passi(passo: float, passi: List[int] = PASSI_ABACO) -> Tuple[int, int]:
    passi_sorted = sorted(passi)
    if passo <= passi_sorted[0]:
        return passi_sorted[0], passi_sorted[0]
    if passo >= passi_sorted[-1]:
        return passi_sorted[-1], passi_sorted[-1]
    for i in range(len(passi_sorted) - 1):
        a, b = passi_sorted[i], passi_sorted[i + 1]
        if a <= passo <= b:
            return a, b
    return passi_sorted[0], passi_sorted[-1]


def lerp(a: float, b: float, t: float) -> float:
    return a * (1 - t) + b * t


def idw_2d(points: List[Tuple[float, float]], values: List[float], x: float, y: float, power: float = 2.0) -> float:
    eps = 1e-12
    num = 0.0
    den = 0.0
    for (px, py), v in zip(points, values):
        d = math.hypot(px - x, py - y)
        if d < eps:
            return float(v)
        w = 1.0 / (d ** power)
        num += w * float(v)
        den += w
    return num / den if den else float("nan")


def nearest_4_points(df_abaco: pd.DataFrame, passo_ref: int, B: float, H: float) -> pd.DataFrame:
    # usa TUTTE le righe del passo e ordina deterministicamente
    sub = df_abaco[df_abaco[COL_PASSO] == float(passo_ref)].copy()
    if sub.empty:
        raise ValueError(f"No abaco rows for passo={passo_ref}")
    sub["dist"] = ((sub[COL_B] - B) ** 2 + (sub[COL_H] - H) ** 2) ** 0.5
    sub = sub.sort_values(["dist", COL_B, COL_H]).head(4).copy()
    return sub


def interpolate_BH_at_step(df_abaco: pd.DataFrame, passo_ref: int, B: float, H: float, power: float = 2.0) -> Dict[str, float]:
    nn = nearest_4_points(df_abaco, passo_ref, B, H)
    pts = list(zip(nn[COL_B].tolist(), nn[COL_H].tolist()))
    return {c: idw_2d(pts, nn[c].tolist(), B, H, power=power) for c in VALUE_COLS}


# =========================
# Helpers: curves + ratio
# =========================
def make_curve_5pts(vals: Dict[str, float]) -> Dict[str, Any]:
    """
    Crea curva 5 punti (compressione negativa, trazione positiva)
    e impone Fu = Fy (come tua regola).
    """
    Ke_T = float(vals[MAP["Ke_T"]])
    Fy_T = float(vals[MAP["Fy_T"]])
    dy_T = float(vals[MAP["dy_T"]])
    du_T = float(vals[MAP["du_T"]])

    Ke_C = float(vals[MAP["Ke_C"]])
    Fy_C = float(vals[MAP["Fy_C"]])
    dy_C = float(vals[MAP["dy_C"]])
    du_C = float(vals[MAP["du_C"]])

    Fu_T = Fy_T
    Fu_C = Fy_C

    pts = [
        {"d": -du_C, "F": -Fu_C},
        {"d": -dy_C, "F": -Fy_C},
        {"d": 0.0, "F": 0.0},
        {"d": dy_T, "F": Fy_T},
        {"d": du_T, "F": Fu_T},
    ]

    return {
        "Ke_T": Ke_T, "Fy_T": Fy_T, "dy_T": dy_T, "du_T": du_T, "Fu_T": Fu_T,
        "Ke_C": Ke_C, "Fy_C": Fy_C, "dy_C": dy_C, "du_C": du_C, "Fu_C": Fu_C,
        "points_5": pts
    }


def apply_ratio_reduction(curve: Dict[str, Any], ratio: float) -> Dict[str, Any]:
    """
    Applica ratio SOLO a K (T e C). Mantiene dy/du.
    Ricalcola:
      Fy' = K' * dy
      Fu' = Fy'
    """
    r = float(ratio)

    Ke_T_new = r * float(curve["Ke_T"])
    Ke_C_new = r * float(curve["Ke_C"])

    dy_T = float(curve["dy_T"])
    dy_C = float(curve["dy_C"])
    du_T = float(curve["du_T"])
    du_C = float(curve["du_C"])

    Fy_T_new = Ke_T_new * dy_T
    Fy_C_new = Ke_C_new * dy_C

    Fu_T_new = Fy_T_new
    Fu_C_new = Fy_C_new

    pts = [
        {"d": -du_C, "F": -Fu_C_new},
        {"d": -dy_C, "F": -Fy_C_new},
        {"d": 0.0, "F": 0.0},
        {"d": dy_T, "F": Fy_T_new},
        {"d": du_T, "F": Fu_T_new},
    ]

    out = dict(curve)
    out.update({
        "ratio_applied": r,
        "Ke_T": Ke_T_new, "Fy_T": Fy_T_new, "Fu_T": Fu_T_new,
        "Ke_C": Ke_C_new, "Fy_C": Fy_C_new, "Fu_C": Fu_C_new,
        "points_5": pts
    })
    return out


# =========================
# API endpoint
# =========================
@app.get("/health")
def health():
    return {"ok": True}


@app.post("/compute")
def compute(payload: PayloadIn):
    """
    Input:  [ { "body": { "panels": [ ... ] } } ]
    Output: { "panels": [ ... ] }  (frontend-ready)
    """
    global DF_ABACO
    if DF_ABACO is None:
        raise HTTPException(status_code=500, detail="Abaco not loaded")

    if not payload or not payload[0].body.panels:
        raise HTTPException(status_code=400, detail="No panels provided")

    df_abaco = DF_ABACO

    panels_out: List[Dict[str, Any]] = []

    for panel in payload[0].body.panels:
        pid = panel.id
        B = float(panel.B_mm)
        H = float(panel.H_mm)
        passo = float(panel.avg_passo_mm)
        ratio = float(panel.ratio)

        passo_low, passo_high = find_bracketing_passi(passo)

        try:
            vals_low = interpolate_BH_at_step(df_abaco, passo_low, B, H, power=IDW_POWER)
            vals_high = interpolate_BH_at_step(df_abaco, passo_high, B, H, power=IDW_POWER)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Panel {pid}: {str(e)}")

        t = 0.0 if passo_low == passo_high else (passo - passo_low) / (passo_high - passo_low)
        vals_interp = {c: lerp(vals_low[c], vals_high[c], t) for c in VALUE_COLS}

        # curve low/high/interp + final reduced
        curve_low = make_curve_5pts(vals_low)
        curve_high = make_curve_5pts(vals_high)
        curve_interp = make_curve_5pts(vals_interp)
        curve_reduced = apply_ratio_reduction(curve_interp, ratio)

        panels_out.append({
            "panel_id": pid,
            "B_mm": B,
            "H_mm": H,
            "avg_passo_mm": passo,
            "ratio": ratio,
            "passo_low": passo_low,
            "passo_high": passo_high,
            "t_passo": t,

            # se vuoi anche i valori numerici “abaco” (utile debugging/front)
            "vals_at_passo_low": vals_low,
            "vals_at_passo_high": vals_high,
            "vals_at_avg_passo": vals_interp,

            # curve frontend-ready
            "curves": {
                "abaco_low": curve_low,
                "abaco_high": curve_high,
                "interp_step": curve_interp,
                "final_reduced": curve_reduced,
            }
        })

    return {"panels": panels_out}
