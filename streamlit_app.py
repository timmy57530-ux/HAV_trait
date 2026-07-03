# -*- coding: utf-8 -*-
"""
WBV / HAV — Dépouillement Streamlit

Portage web de l'application PyQt :
- import par fichier téléversé, détection skip/colonnes/Fe ;
- timeline interactive Plotly ;
- calculs aHV Wh, aHV(wp) Wp, pF avec pondération Flat_h ;
- résultats lisibles, exports TSV/CSV français ;
- multi-découpe ;
- aucune modification volontaire des formules métier.

À déposer avec fonctionTCi.py dans le même dépôt Streamlit.
"""

from __future__ import annotations

import hashlib
import html
import io
import math
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time as dtime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

try:
    from scipy.signal import resample as scipy_resample
except Exception:  # pragma: no cover - dépendance gérée par requirements.txt
    scipy_resample = None

try:
    import fonctionTCi as TC
except Exception as e:  # l'app reste affichable, mais les calculs sont bloqués
    TC = None
    _TC_IMPORT_ERROR = str(e)


# ══════════════════════════════════════════════════════════════════════════════
# Configuration Streamlit
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="WBV / HAV — Dépouillement",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

PALETTE = [
    "#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
    "#9467bd", "#8c564b", "#17becf", "#7f7f7f",
]
DEFAULT_LABELS = {1: "x1", 2: "y1", 3: "z1", 4: "x2", 5: "y2", 6: "z2"}
FILTERS = ["wk", "wd", "wf", "wc", "we", "wj", "wsl", "wsv", "wp", "flat", "wh"]

THEMES: dict[str, dict[str, str]] = {
    "Clair": {
        "page": "#f4f7fb", "panel": "#ffffff", "panel2": "#f7f9fd", "border": "#d3deea",
        "text": "#132335", "muted": "#5c728b", "muted2": "#7f93aa",
        "chip": "#e8eef6", "chip_text": "#2d4056", "grid": "#e7edf5",
        "ahv_bg": "#e8f5ff", "ahv_soft": "#d7edff", "ahv_border": "#5ba8dc", "ahv_title": "#07598e", "ahv_badge": "#b8def8",
        "wp_bg": "#e6f8ee", "wp_soft": "#d6f2e3", "wp_border": "#52b879", "wp_title": "#0b6534", "wp_badge": "#b9ebd0",
        "pf_bg": "#fff2df", "pf_soft": "#ffe5c2", "pf_border": "#d7924d", "pf_title": "#8d4310", "pf_badge": "#ffd59f",
        "plot_template": "plotly_white",
    },
    "Sombre": {
        "page": "#0f1724", "panel": "#172234", "panel2": "#111a28", "border": "#2f4564",
        "text": "#e8eef7", "muted": "#9db0c8", "muted2": "#7488a3",
        "chip": "#223149", "chip_text": "#d7e4f6", "grid": "#253348",
        "ahv_bg": "#102a43", "ahv_soft": "#123a5d", "ahv_border": "#2d7db8", "ahv_title": "#58c4ff", "ahv_badge": "#0b4b78",
        "wp_bg": "#102e22", "wp_soft": "#123d2d", "wp_border": "#2da56b", "wp_title": "#66f0aa", "wp_badge": "#0d6b42",
        "pf_bg": "#382313", "pf_soft": "#4a2c12", "pf_border": "#d48332", "pf_title": "#ffb56b", "pf_badge": "#8a4a12",
        "plot_template": "plotly_dark",
    },
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    title: str
    status: str
    badge: str
    filter_txt: str
    filter_code: str
    final_label: str
    comp_label: str
    bg: str
    soft: str
    border: str
    title_color: str
    badge_bg: str
    unit: str = "m/s²"


# ══════════════════════════════════════════════════════════════════════════════
# CSS / thème résultats
# ══════════════════════════════════════════════════════════════════════════════


def inject_css(theme_name: str) -> None:
    t = THEMES[theme_name]
    st.markdown(
        f"""
        <style>
        :root {{
            --hav-page: {t['page']}; --hav-panel: {t['panel']}; --hav-panel2: {t['panel2']};
            --hav-border: {t['border']}; --hav-text: {t['text']}; --hav-muted: {t['muted']};
            --hav-muted2: {t['muted2']}; --hav-chip: {t['chip']}; --hav-chip-text: {t['chip_text']};
            --hav-grid: {t['grid']};
        }}
        .main .block-container {{ padding-top: 1.2rem; max-width: 100%; }}
        .hav-title {{
            padding: 0.85rem 1rem; border: 1px solid var(--hav-border); border-radius: 16px;
            background: linear-gradient(135deg, var(--hav-panel), var(--hav-panel2)); color: var(--hav-text);
            margin-bottom: 0.8rem;
        }}
        .hav-title h1 {{ margin: 0; font-size: 1.55rem; }}
        .hav-title p {{ margin: .25rem 0 0; color: var(--hav-muted); }}
        .hav-card {{
            background: var(--hav-panel); border: 1px solid var(--hav-border); border-radius: 14px;
            padding: .85rem; color: var(--hav-text); box-shadow: 0 1px 1px rgba(0,0,0,.04);
        }}
        .hav-note {{ color: var(--hav-muted); font-size: .86rem; }}
        .hav-chip {{
            display: inline-block; padding: .20rem .55rem; border-radius: 999px; background: var(--hav-chip);
            color: var(--hav-chip-text); border: 1px solid var(--hav-border); font-size: .74rem; font-weight: 700;
            margin: .08rem .2rem .08rem 0;
        }}
        .results-wrap {{
            background: var(--hav-panel); border: 1px solid var(--hav-border); border-radius: 18px;
            padding: .9rem; margin-top: .5rem; color: var(--hav-text);
        }}
        .results-legend {{
            display: flex; align-items: center; gap: .45rem; flex-wrap: wrap;
            background: var(--hav-panel2); border: 1px solid var(--hav-border); border-radius: 12px;
            padding: .6rem .75rem; margin-bottom: .8rem;
        }}
        .result-group {{
            background: var(--hav-panel2); border: 1px solid var(--hav-border); border-radius: 14px;
            padding: .75rem; margin: .75rem 0;
        }}
        .result-head {{
            display:flex; justify-content:space-between; gap:.8rem; flex-wrap:wrap; align-items:center;
            border-bottom: 1px solid var(--hav-border); padding-bottom: .55rem; margin-bottom: .7rem;
        }}
        .result-head b {{ color: var(--hav-text); }}
        .result-head span {{ color: var(--hav-muted); font-size: .82rem; }}
        .summary-grid {{ display:grid; grid-template-columns: repeat(3, minmax(230px, 1fr)); gap:.65rem; }}
        @media (max-width: 1100px) {{ .summary-grid {{ grid-template-columns: 1fr; }} }}
        .metric-card {{ border-radius: 14px; padding: .8rem; border: 1px solid; border-left-width: 6px; }}
        .metric-top {{ display:flex; align-items:center; gap:.45rem; flex-wrap:wrap; }}
        .metric-title {{ font-weight:800; font-size:1rem; }}
        .metric-badge {{ border-radius:999px; padding:.16rem .5rem; font-size:.72rem; font-weight:800; }}
        .metric-label {{ color: var(--hav-muted); font-size: .78rem; margin-top:.35rem; }}
        .metric-value {{ font-size: 2.1rem; line-height: 1.1; font-weight: 850; margin-top:.25rem; }}
        .metric-unit {{ color: var(--hav-muted2); font-size: .9rem; font-weight:600; }}
        .metric-meta {{ color: var(--hav-muted2); font-size:.78rem; margin-top:.15rem; }}
        .detail-table {{ width:100%; border-collapse: separate; border-spacing: 0 .35rem; margin-top:.7rem; }}
        .detail-table th {{
            background: var(--hav-chip); color: var(--hav-chip-text); text-align:left; border: 1px solid var(--hav-border);
            padding:.45rem .5rem; font-size:.78rem;
        }}
        .detail-table td {{
            background: var(--hav-panel); color: var(--hav-text); border: 1px solid var(--hav-border);
            padding:.48rem .52rem; vertical-align: top; font-size:.83rem;
        }}
        .detail-table td small {{ color: var(--hav-muted); }}
        .empty-results {{
            background: var(--hav-panel2); border: 1px dashed var(--hav-border); border-radius: 14px;
            padding: 1rem; color: var(--hav-muted);
        }}
        .stDownloadButton button {{ width: 100%; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# Parsing / fichiers
# ══════════════════════════════════════════════════════════════════════════════


def try_parse_float(x: Any) -> float | None:
    try:
        return float(x)
    except Exception:
        return None


def parse_num_robust(s: str) -> float | None:
    s = str(s).strip()
    if not s:
        return None
    attempts = (s, s.replace(",", "."), s.replace(".", "").replace(",", "."))
    for attempt in attempts:
        try:
            return float(attempt)
        except ValueError:
            continue
    return None


def guess_delimiter(line: str) -> str:
    line = line.strip()
    if "\t" in line and len(line.split("\t")) >= 2:
        return "\t"
    if ";" in line and len(line.split(";")) >= 2:
        return ";"
    if "," in line:
        parts = line.split(",")
        if len(parts) >= 3:
            num_ok = sum(1 for p in parts if try_parse_float(p.strip()) is not None)
            if num_ok / len(parts) >= 0.6:
                return ","
    return " "


def split_line(line: str, delimiter: str) -> list[str]:
    if delimiter == " ":
        return line.strip().split()
    return [p.strip() for p in line.strip().split(delimiter)]


def try_parse_datetime_like(s: str):
    s0 = str(s).strip()
    if not s0:
        return None
    s1 = " ".join(s0.replace("T", " ").replace(",", ".").split())
    try:
        return datetime.fromisoformat(s1)
    except Exception:
        pass
    patterns = [
        "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
        "%Y/%m/%d %H:%M:%S.%f", "%Y/%m/%d %H:%M:%S",
        "%d/%m/%Y %H:%M:%S.%f", "%d/%m/%Y %H:%M:%S",
        "%d-%m-%Y %H:%M:%S.%f", "%d-%m-%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S,%f", "%d/%m/%Y %H:%M:%S,%f",
    ]
    for p in patterns:
        try:
            return datetime.strptime(s0, p)
        except Exception:
            continue
    return None


def try_parse_time_only(s: str) -> float | None:
    s0 = str(s).strip().replace(",", ".")
    parts = s0.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60.0 + float(parts[1])
    except Exception:
        return None
    return None


def parse_time_vector_to_seconds(time_col: Iterable[Any] | None) -> np.ndarray | None:
    if time_col is None:
        return None
    arr = np.asarray(list(time_col), dtype=object)
    if arr.size == 0:
        return None
    try:
        out = np.asarray(arr, dtype=float)
        return out - float(out[0])
    except Exception:
        pass

    secs = np.empty(arr.size, dtype=float)
    base_dt = None
    base_sec = None
    for i, v in enumerate(arr):
        if isinstance(v, (int, float, np.integer, np.floating)):
            fv = float(v)
            if base_sec is None:
                base_sec = fv
            secs[i] = fv
            continue
        if isinstance(v, datetime):
            if base_dt is None:
                base_dt = v
            secs[i] = (v - base_dt).total_seconds()
            continue
        if isinstance(v, date):
            dv = datetime.combine(v, dtime(0, 0, 0))
            if base_dt is None:
                base_dt = dv
            secs[i] = (dv - base_dt).total_seconds()
            continue
        s = str(v).strip()
        fv = try_parse_float(s)
        if fv is not None:
            if base_sec is None:
                base_sec = fv
            secs[i] = fv
            continue
        t_only = try_parse_time_only(s)
        if t_only is not None:
            if base_sec is None:
                base_sec = t_only
            secs[i] = t_only
            continue
        dtv = try_parse_datetime_like(s)
        if dtv is not None:
            if base_dt is None:
                base_dt = dtv
            secs[i] = (dtv - base_dt).total_seconds()
            continue
        return None
    return secs - float(secs[0])


def parse_timecode(txt: Any) -> float:
    s = str(txt).strip().replace(",", ".")
    if not s:
        raise ValueError("timecode vide")
    parts = s.split(":")
    if len(parts) == 3:
        return int(parts[0]) * 3600.0 + int(parts[1]) * 60.0 + float(parts[2])
    if len(parts) == 2:
        return int(parts[0]) * 60.0 + float(parts[1])
    return float(s)


def parse_columns_list(raw: str) -> list[int]:
    raw = (raw or "").replace(" ", "")
    if not raw:
        return []
    return [int(x) for x in raw.split(",") if x != ""]


@st.cache_data(show_spinner=False, max_entries=32)
def scan_file_structure_bytes(file_bytes: bytes) -> dict[str, Any]:
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()[:2000]
    result = {
        "skip": 0,
        "delimiter": " ",
        "n_cols": 7,
        "col_str": "0,1,2,3,4,5,6",
        "probably_time0": True,
    }
    delimiter = " "
    for line in lines:
        if line.strip():
            delimiter = guess_delimiter(line.strip())
            break
    result["delimiter"] = delimiter

    first_data_idx = 0
    candidate_idx = None
    consecutive = 0
    col_counts: list[int] = []
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            consecutive = 0
            candidate_idx = None
            continue
        parts = split_line(stripped, delimiter)
        if not parts:
            consecutive = 0
            continue
        num_parts = [p for p in parts if parse_num_robust(p) is not None]
        ratio = len(num_parts) / max(len(parts), 1)
        if ratio >= 0.75 and len(num_parts) >= 2:
            consecutive += 1
            if consecutive == 1:
                candidate_idx = i
            if consecutive >= 3 and candidate_idx is not None:
                first_data_idx = candidate_idx
                col_counts.append(len(parts))
                for j in range(i + 1, min(i + 50, len(lines))):
                    l2 = lines[j].strip()
                    if l2:
                        p2 = split_line(l2, delimiter)
                        n2 = sum(1 for p in p2 if parse_num_robust(p) is not None)
                        if n2 >= 2:
                            col_counts.append(len(p2))
                break
        else:
            consecutive = 0
            candidate_idx = None

    result["skip"] = int(first_data_idx)
    if col_counts:
        n_cols = Counter(col_counts).most_common(1)[0][0]
        result["n_cols"] = int(n_cols)
        result["col_str"] = ",".join(str(i) for i in range(n_cols))
    else:
        result["n_cols"] = 1
        result["col_str"] = "0"

    times: list[float] = []
    for line in lines[first_data_idx:first_data_idx + 30]:
        parts = split_line(line.strip(), delimiter)
        if parts:
            v = parse_num_robust(parts[0])
            if v is not None:
                times.append(v)
    if len(times) >= 5:
        diffs = np.diff(times)
        diffs_std = float(np.std(diffs))
        diffs_mean = float(np.mean(np.abs(diffs)))
        result["probably_time0"] = bool(
            np.all(diffs > 0)
            and diffs_mean > 0
            and diffs_std / (diffs_mean + 1e-12) < 0.05
            and abs(times[0]) < 1e9
        )
    return result


def safe_filename(name: str) -> str:
    base = os.path.basename(name or "mesure.txt")
    base = re.sub(r"[^A-Za-z0-9_.-]+", "_", base)
    return base or "mesure.txt"


def materialize_upload(file_bytes: bytes, filename: str) -> str:
    digest = hashlib.sha256(file_bytes).hexdigest()[:16]
    folder = Path(tempfile.gettempdir()) / "hav_streamlit_uploads"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{digest}_{safe_filename(filename)}"
    if not path.exists() or path.stat().st_size != len(file_bytes):
        path.write_bytes(file_bytes)
    return str(path)


@st.cache_data(show_spinner=False, max_entries=16)
def load_data_cached(file_bytes: bytes, filename: str, skip: int, cols_tuple: tuple[int, ...]):
    if TC is None:
        raise RuntimeError(
            "Module fonctionTCi introuvable. Ajoute fonctionTCi.py dans le même dossier que streamlit_app.py. "
            f"Erreur import : {_TC_IMPORT_ERROR}"
        )
    filepath = materialize_upload(file_bytes, filename)
    data = TC.Recup(filepath, int(skip), list(cols_tuple))
    if not data or not len(data):
        raise ValueError("Aucune donnée valide — vérifie skip, colonnes et format du fichier.")
    return data, list(cols_tuple), filepath


# ══════════════════════════════════════════════════════════════════════════════
# Calculs métier portés sans changement de formule
# ══════════════════════════════════════════════════════════════════════════════


def get_by_col(data: list[Any], cols: list[int], col: int) -> Any:
    if col not in cols:
        raise ValueError(f"Colonne {col} absente de la liste {cols}.")
    return data[cols.index(col)]


def rms(signal: Iterable[float]) -> float:
    arr = np.asarray(signal, dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.square(arr))))


def cut_signal(sig: Iterable[Any], fe: float, start: float, end: float):
    if TC is not None:
        return TC.Cut(sig, fe, start, end, False)
    arr = np.asarray(sig)
    i0 = max(0, int(round(start * fe)))
    i1 = min(len(arr), int(round(end * fe)))
    return arr[i0:i1]


def resampling_allowed(ctx: str, enabled: bool, ahv_only: bool) -> bool:
    if not enabled:
        return False
    if not ahv_only:
        return True
    return ctx in ("ahv", "ahvwp")


def effective_fs(fe: float, ctx: str, enabled: bool, ahv_only: bool, fs_new: float) -> float:
    if not resampling_allowed(ctx, enabled, ahv_only):
        return float(fe)
    return float(fs_new)


def resample_signal(sig: np.ndarray, fe: float, t: np.ndarray | None, ctx: str,
                    enabled: bool, ahv_only: bool, fs_new: float):
    if not resampling_allowed(ctx, enabled, ahv_only) or scipy_resample is None:
        return sig, float(fe), t, len(sig)
    fe_new = float(fs_new)
    n0 = len(sig)
    if n0 < 2:
        return sig, float(fe), t, n0
    dur = (float(t[-1] - t[0]) if t is not None and len(t) == n0 else (n0 - 1) / float(fe))
    if dur <= 0:
        dur = (n0 - 1) / float(fe)
    n_new = int(max(2, round(dur * fe_new) + 1))
    sig_rs = np.asarray(scipy_resample(np.asarray(sig, dtype=float), n_new), dtype=float)
    t_rs = (
        np.linspace(float(t[0]), float(t[-1]), n_new)
        if t is not None and len(t) == n0 else np.arange(n_new, dtype=float) / fe_new
    )
    return sig_rs, float(fe_new), t_rs, n_new


def cut_filter(data, cols: list[int], col: int, fe: float, start: float, end: float,
               filt_code: str, ctx: str, resample_enabled: bool, resample_ahv_only: bool,
               fs_new: float) -> tuple[np.ndarray, float]:
    if TC is None:
        raise RuntimeError("fonctionTCi est requis pour appliquer les filtres de pondération.")
    sig = np.asarray(TC.Cut(get_by_col(data, cols, col), fe, start, end, False), dtype=float)
    sig_rs, fe_eff, _, _ = resample_signal(
        sig, fe, None, ctx, resample_enabled, resample_ahv_only, fs_new
    )
    arr = np.asarray(TC.Filtre(sig_rs, fe_eff, filt_code), dtype=float)
    return arr, fe_eff


def compute_ahv3(cols3: list[int], data, cols: list[int], fe: float, start: float, end: float,
                 resample_enabled: bool, resample_ahv_only: bool, fs_new: float):
    sx, fe_eff = cut_filter(data, cols, cols3[0], fe, start, end, "wh", "ahv", resample_enabled, resample_ahv_only, fs_new)
    sy, _ = cut_filter(data, cols, cols3[1], fe, start, end, "wh", "ahv", resample_enabled, resample_ahv_only, fs_new)
    sz, _ = cut_filter(data, cols, cols3[2], fe, start, end, "wh", "ahv", resample_enabled, resample_ahv_only, fs_new)
    ax, ay, az = rms(sx), rms(sy), rms(sz)
    ahv = float(np.sqrt(ax ** 2 + ay ** 2 + az ** 2))
    return ax, ay, az, ahv, fe_eff


def compute_ahvwp3(cols3: list[int], data, cols: list[int], fe: float, start: float, end: float,
                   resample_enabled: bool, resample_ahv_only: bool, fs_new: float):
    sx, fe_eff = cut_filter(data, cols, cols3[0], fe, start, end, "wp", "ahvwp", resample_enabled, resample_ahv_only, fs_new)
    sy, _ = cut_filter(data, cols, cols3[1], fe, start, end, "wp", "ahvwp", resample_enabled, resample_ahv_only, fs_new)
    sz, _ = cut_filter(data, cols, cols3[2], fe, start, end, "wp", "ahvwp", resample_enabled, resample_ahv_only, fs_new)
    ax, ay, az = rms(sx), rms(sy), rms(sz)
    ahvwp = float(np.sqrt(ax ** 2 + ay ** 2 + az ** 2))
    return ax, ay, az, ahvwp, fe_eff


def compute_pf3(cols3: list[int], data, cols: list[int], fe: float, start: float, end: float,
                resample_enabled: bool, resample_ahv_only: bool, fs_new: float):
    # Code filtre conservé : fonctionTCi attend généralement "flat".
    # Libellé interface : pondération Flat_h.
    x, fe_eff = cut_filter(data, cols, cols3[0], fe, start, end, "flat", "pf", resample_enabled, resample_ahv_only, fs_new)
    y, _ = cut_filter(data, cols, cols3[1], fe, start, end, "flat", "pf", resample_enabled, resample_ahv_only, fs_new)
    z, _ = cut_filter(data, cols, cols3[2], fe, start, end, "flat", "pf", resample_enabled, resample_ahv_only, fs_new)
    AFx, AFy, AFz = rms(x), rms(y), rms(z)
    afv = np.sqrt(x ** 2 + y ** 2 + z ** 2)
    AFv = rms(afv)
    den = np.sum(afv ** 4)
    pf = float(np.sqrt(np.sum(afv ** 6) / den)) if den != 0 else float("nan")
    return AFx, AFy, AFz, AFv, pf, fe_eff


def compute_1_axis(col: int, metric: str, data, cols: list[int], fe: float, start: float, end: float,
                   resample_enabled: bool, resample_ahv_only: bool, fs_new: float) -> dict[str, Any]:
    duration = end - start
    axes_lbl = f"[{col}]"
    if metric == "aHV":
        sig, fe_eff = cut_filter(data, cols, col, fe, start, end, "wh", "ahv", resample_enabled, resample_ahv_only, fs_new)
        val = float(abs(rms(sig)))
        return dict(type="aHV", axes=axes_lbl, start=start, end=end, duree=duration, x=val, y=np.nan, z=np.nan, res=val, fs=fe_eff)
    if metric == "aHVwp":
        sig, fe_eff = cut_filter(data, cols, col, fe, start, end, "wp", "ahvwp", resample_enabled, resample_ahv_only, fs_new)
        val = float(abs(rms(sig)))
        return dict(type="aHVwp", axes=axes_lbl, start=start, end=end, duree=duration, x=val, y=np.nan, z=np.nan, res=val, fs=fe_eff)
    sig, fe_eff = cut_filter(data, cols, col, fe, start, end, "flat", "pf", resample_enabled, resample_ahv_only, fs_new)
    afv = np.abs(sig)
    dt = 1.0 / fe_eff
    den = np.trapz(afv ** 4, dx=dt)
    pf = float(np.sqrt(np.trapz(afv ** 6, dx=dt) / den)) if den != 0 else float("nan")
    return dict(type="pF", axes=axes_lbl, start=start, end=end, duree=duration, x=pf, y=np.nan, z=np.nan, res=pf, fs=fe_eff)


def run_group(cols3: list[int], data, cols: list[int], fe: float, start: float, end: float,
              resample_enabled: bool, resample_ahv_only: bool, fs_new: float,
              do_ahv: bool = True, do_wp: bool = True, do_pf: bool = True) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    duration = end - start
    axes_lbl = f"[{cols3[0]},{cols3[1]},{cols3[2]}]"
    if do_ahv:
        ax, ay, az, ahv, fe_ahv = compute_ahv3(cols3, data, cols, fe, start, end, resample_enabled, resample_ahv_only, fs_new)
        rows.append(dict(type="aHV", axes=axes_lbl, start=start, end=end, duree=duration, x=ax, y=ay, z=az, res=ahv, fs=fe_ahv))
    if do_wp:
        wx, wy, wz, ahvwp, fe_wp = compute_ahvwp3(cols3, data, cols, fe, start, end, resample_enabled, resample_ahv_only, fs_new)
        rows.append(dict(type="aHVwp", axes=axes_lbl, start=start, end=end, duree=duration, x=wx, y=wy, z=wz, res=ahvwp, fs=fe_wp))
    if do_pf:
        AFx, AFy, AFz, AFv, pf, fe_pf = compute_pf3(cols3, data, cols, fe, start, end, resample_enabled, resample_ahv_only, fs_new)
        rows.append(dict(type="pF", axes=axes_lbl, start=start, end=end, duree=duration, x=AFx, y=AFy, z=AFz, afv=AFv, res=pf, fs=fe_pf))
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Graphiques / auto-découpe
# ══════════════════════════════════════════════════════════════════════════════


def downsample_xy(x: np.ndarray, y: np.ndarray, max_points: int = 14_000) -> tuple[np.ndarray, np.ndarray]:
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]
    if n <= max_points:
        return x, y
    step = int(math.ceil(n / max_points))
    return x[::step], y[::step]


def make_timeline_figure(t: np.ndarray, sigs: dict[int, np.ndarray], labels: dict[int, str],
                         start: float, end: float, theme_name: str, synthetic: bool = True) -> go.Figure:
    theme = THEMES[theme_name]
    active_cols = list(sigs.keys())
    if not active_cols:
        fig = go.Figure()
        fig.update_layout(template=theme["plot_template"], height=320, title="Aucune courbe sélectionnée")
        return fig

    if synthetic:
        fig = make_subplots(
            rows=len(active_cols), cols=1, shared_xaxes=True, vertical_spacing=0.025,
            subplot_titles=[labels.get(c, str(c)) for c in active_cols],
        )
        for i, col in enumerate(active_cols, start=1):
            y = np.asarray(sigs[col], dtype=float)
            xd, yd = downsample_xy(np.asarray(t, dtype=float), y)
            fig.add_trace(
                go.Scattergl(x=xd, y=yd, mode="lines", name=labels.get(col, f"Col {col}"), line=dict(color=PALETTE[(i - 1) % len(PALETTE)], width=1)),
                row=i, col=1,
            )
            if y.size:
                fig.add_annotation(
                    x=float(xd[-1]) if len(xd) else 0,
                    y=float(np.nanmax(yd)) if len(yd) else 0,
                    text=f"RMS={rms(y):.4f} · DC={float(np.nanmean(y)):.4f}",
                    showarrow=False, xanchor="right", yanchor="top",
                    font=dict(size=10, color=theme["muted"]), row=i, col=1,
                )
            fig.update_yaxes(title_text="m/s²", showgrid=True, gridcolor=theme["grid"], row=i, col=1)
    else:
        fig = go.Figure()
        for i, col in enumerate(active_cols):
            y = np.asarray(sigs[col], dtype=float)
            xd, yd = downsample_xy(np.asarray(t, dtype=float), y)
            fig.add_trace(go.Scattergl(x=xd, y=yd, mode="lines", name=labels.get(col, f"Col {col}"), line=dict(color=PALETTE[i % len(PALETTE)], width=1)))
        fig.update_yaxes(title_text="Signal (m/s²)", showgrid=True, gridcolor=theme["grid"])

    fig.update_xaxes(title_text="Temps (s)", showgrid=True, gridcolor=theme["grid"])
    fill = "rgba(0, 180, 220, 0.18)" if theme_name == "Sombre" else "rgba(30, 120, 200, 0.14)"
    try:
        fig.add_vrect(x0=start, x1=end, fillcolor=fill, line_width=1, line_color="#00b8d8", row="all", col=1)
    except Exception:
        fig.add_vrect(x0=start, x1=end, fillcolor=fill, line_width=1, line_color="#00b8d8")
    fig.update_layout(
        template=theme["plot_template"], height=max(320, 220 * len(active_cols)) if synthetic else 520,
        margin=dict(l=30, r=20, t=35, b=35), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def make_spectrum_figure(signals: list[np.ndarray], fe: float, labels: list[str], theme_name: str) -> go.Figure:
    theme = THEMES[theme_name]
    fig = go.Figure()
    for i, sig in enumerate(signals):
        arr = np.asarray(sig, dtype=float)
        if arr.size < 2:
            continue
        arr = arr - float(np.mean(arr))
        freq = np.fft.rfftfreq(arr.size, d=1.0 / fe)
        amp = np.abs(np.fft.rfft(arr)) / max(arr.size, 1)
        xd, yd = downsample_xy(freq, amp, 18_000)
        fig.add_trace(go.Scattergl(x=xd, y=yd, mode="lines", name=labels[i], line=dict(color=PALETTE[i % len(PALETTE)], width=1)))
    fig.update_layout(template=theme["plot_template"], height=430, margin=dict(l=30, r=20, t=30, b=35))
    fig.update_xaxes(title="Fréquence (Hz)", showgrid=True, gridcolor=theme["grid"])
    fig.update_yaxes(title="Amplitude", showgrid=True, gridcolor=theme["grid"])
    return fig


def detect_vibration_zones(sig: np.ndarray, t: np.ndarray, threshold: float, min_dur_s: float = 0.1, gap_s: float = 0.5) -> list[tuple[float, float]]:
    n = len(sig)
    if n < 2 or len(t) != n:
        return []
    dt = (float(t[-1]) - float(t[0])) / max(n - 1, 1)
    if dt <= 0:
        dt = 1.0 / 20_000.0
    win = max(1, int(0.05 / dt))
    abs_sig = np.abs(sig.astype(float))
    if win > 1:
        kernel = np.ones(win, dtype=float) / win
        envelope = np.convolve(abs_sig, kernel, mode="same")
    else:
        envelope = abs_sig
    above = envelope > threshold
    min_samples = max(1, int(min_dur_s / dt))
    zones: list[tuple[float, float]] = []
    in_zone = False
    start_idx = 0
    for i in range(n):
        if above[i] and not in_zone:
            in_zone = True
            start_idx = i
        elif not above[i] and in_zone:
            if i - start_idx >= min_samples:
                zones.append((float(t[start_idx]), float(t[i - 1])))
            in_zone = False
    if in_zone and n - start_idx >= min_samples:
        zones.append((float(t[start_idx]), float(t[-1])))
    if len(zones) > 1 and gap_s > 0:
        merged = [list(zones[0])]
        for z in zones[1:]:
            if z[0] - merged[-1][1] < gap_s:
                merged[-1][1] = z[1]
            else:
                merged.append(list(z))
        zones = [tuple(z) for z in merged]
    return zones


# ══════════════════════════════════════════════════════════════════════════════
# Résultats / export
# ══════════════════════════════════════════════════════════════════════════════


def esc(x: Any) -> str:
    return html.escape(str(x), quote=True)


def fmt(v: Any, dec: int = 3) -> str:
    try:
        fv = float(v)
        if np.isnan(fv):
            return "—"
        return f"{fv:.{dec}f}"
    except Exception:
        return "—"


def fmt_int(v: Any) -> str:
    try:
        fv = float(v)
        if np.isnan(fv):
            return "—"
        return f"{fv:.0f}"
    except Exception:
        return "—"


def csv_num(v: Any, dec: int = 4) -> str:
    try:
        fv = float(v)
        if np.isnan(fv):
            return ""
        return f"{fv:.{dec}f}".replace(".", ",")
    except Exception:
        return "" if v is None else str(v)


def csv_int(v: Any) -> str:
    try:
        return str(int(round(float(v))))
    except Exception:
        return ""


def parse_axes(axes_lbl: str) -> list[int]:
    try:
        return [int(x) for x in axes_lbl.strip("[]").split(",") if x.strip()]
    except Exception:
        return []


def metric_specs(theme_name: str, mtype: str) -> MetricSpec:
    t = THEMES[theme_name]
    if mtype == "aHV":
        return MetricSpec("aHV", "a<sub>hv</sub>", "pondéré Wh", "PONDÉRÉ · Wh", "Wh", "wh", "Valeur finale pondérée Wh", "Composantes RMS pondérées Wh", t["ahv_bg"], t["ahv_soft"], t["ahv_border"], t["ahv_title"], t["ahv_badge"])
    if mtype == "aHVwp":
        return MetricSpec("aHVwp", "a<sub>hv</sub>(wp)", "pondéré Wp", "PONDÉRÉ · Wp", "Wp", "wp", "Valeur finale pondérée Wp", "Composantes RMS pondérées Wp", t["wp_bg"], t["wp_soft"], t["wp_border"], t["wp_title"], t["wp_badge"])
    return MetricSpec("pF", "p<sub>F</sub>", "pondéré Flat_h", "PONDÉRÉ · Flat_h", "Flat_h", "flat", "Valeur finale pondérée Flat_h", "Composantes RMS pondérées Flat_h", t["pf_bg"], t["pf_soft"], t["pf_border"], t["pf_title"], t["pf_badge"])


def components_for(row: dict[str, Any], mtype: str, axes_lbl: str) -> list[tuple[str, str, Any]]:
    axes = parse_axes(axes_lbl)
    s = "2" if axes and axes[0] >= 4 else "1"
    if len(axes) == 1:
        axis = f"colonne {axes[0]}"
        if mtype == "aHV":
            return [(f"{axis} · RMS Wh", "x", row.get("x"))]
        if mtype == "aHVwp":
            return [(f"{axis} · RMS Wp", "x", row.get("x"))]
        return [(f"{axis} · pF Flat_h", "x", row.get("x"))]
    if mtype == "aHV":
        return [(f"x{s} · RMS Wh", "x", row.get("x")), (f"y{s} · RMS Wh", "y", row.get("y")), (f"z{s} · RMS Wh", "z", row.get("z"))]
    if mtype == "aHVwp":
        return [(f"x{s} · RMS Wp", "x", row.get("x")), (f"y{s} · RMS Wp", "y", row.get("y")), (f"z{s} · RMS Wp", "z", row.get("z"))]
    return [
        (f"AFx{s} · RMS Flat_h", "x", row.get("x")),
        (f"AFy{s} · RMS Flat_h", "y", row.get("y")),
        (f"AFz{s} · RMS Flat_h", "z", row.get("z")),
        (f"AFv{s} · RMS vectoriel Flat_h", "afv", row.get("afv")),
    ]


def render_results(rows: list[dict[str, Any]], theme_name: str) -> None:
    t = THEMES[theme_name]
    if not rows:
        st.markdown(
            """
            <div class="results-wrap">
              <div class="empty-results">
                <b>Aucun résultat affiché</b><br>
                Choisis une plage Start/End puis lance un calcul. Les cartes préciseront : pondéré Wh, pondéré Wp ou pondéré Flat_h.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for row in rows:
        axes = row.get("axes", "")
        groups.setdefault(axes, {})[row.get("type", "")] = row

    chunks = [
        '<div class="results-wrap">',
        '<div class="results-legend"><b>Synthèse lisible</b><span class="hav-note">valeurs finales en haut, détail complet par indicateur juste en dessous.</span>',
        '<span class="hav-chip">PONDÉRÉ Wh</span><span class="hav-chip">PONDÉRÉ Wp</span><span class="hav-chip">PONDÉRÉ Flat_h</span></div>',
    ]
    for axes_lbl, grp in groups.items():
        ref = grp.get("aHV") or grp.get("aHVwp") or grp.get("pF") or {}
        fs_values = sorted({fmt_int(r.get("fs")) for r in grp.values() if fmt_int(r.get("fs")) != "—"})
        fs_txt = f"{fs_values[0]} Hz" if len(fs_values) == 1 else "variable — voir détail"
        chunks.append('<div class="result-group">')
        chunks.append(
            '<div class="result-head">'
            f'<b>Axes analysés : {esc(axes_lbl)}</b>'
            f'<span>Start <b>{fmt(ref.get("start"), 4)}</b> s · End <b>{fmt(ref.get("end"), 4)}</b> s · '
            f'Durée <b>{fmt(ref.get("duree"), 4)}</b> s · Fs eff. <b>{esc(fs_txt)}</b></span>'
            '</div>'
        )
        chunks.append('<div class="summary-grid">')
        for mtype in ("aHV", "aHVwp", "pF"):
            row = grp.get(mtype)
            if not row:
                continue
            sp = metric_specs(theme_name, mtype)
            chunks.append(
                f'<div class="metric-card" style="background:{sp.bg}; border-color:{sp.border};">'
                '<div class="metric-top">'
                f'<span class="metric-title" style="color:{sp.title_color};">{sp.title}</span>'
                f'<span class="metric-badge" style="background:{sp.badge_bg}; color:{t["text"]};">{esc(sp.badge)}</span>'
                '</div>'
                f'<div class="metric-label">{esc(sp.final_label)} · filtre {esc(sp.filter_txt)}</div>'
                f'<div class="metric-value" style="color:{sp.title_color};">{fmt(row.get("res"), 3)} <span class="metric-unit">{sp.unit}</span></div>'
                f'<div class="metric-meta">Statut : <b>{esc(sp.status)}</b> · Fs : <b>{fmt_int(row.get("fs"))}</b> Hz</div>'
                '</div>'
            )
        chunks.append('</div>')
        chunks.append('<table class="detail-table"><thead><tr><th>Indicateur</th><th>Statut</th><th>Filtre</th><th>Composantes RMS / détail</th><th>Valeur finale</th><th>Fs eff.</th></tr></thead><tbody>')
        for mtype in ("aHV", "aHVwp", "pF"):
            row = grp.get(mtype)
            if not row:
                continue
            sp = metric_specs(theme_name, mtype)
            comp_lines = []
            for label, _, value in components_for(row, mtype, axes_lbl):
                if fmt(value) != "—":
                    comp_lines.append(f"<small>{esc(label)}</small> <b>{fmt(value, 3)}</b> <small>m/s²</small>")
            if mtype == "pF":
                formula = "√(Σ AFv⁶ / Σ AFv⁴) sur le module vectoriel pondéré Flat_h"
            elif mtype == "aHVwp":
                formula = "√(x² + y² + z²) sur les composantes pondérées Wp"
            else:
                formula = "√(x² + y² + z²) sur les composantes pondérées Wh"
            comp_html = "<br>".join(comp_lines) + f"<br><small>{esc(formula)}</small>"
            chunks.append(
                "<tr>"
                f"<td><b style='color:{sp.title_color};'>{sp.title}</b><br><small>{esc(sp.final_label)}</small></td>"
                f"<td>{esc(sp.status)}</td>"
                f"<td><b>{esc(sp.filter_txt)}</b><br><small>filtre de pondération {esc(sp.filter_txt)}</small></td>"
                f"<td>{comp_html}</td>"
                f"<td style='text-align:right;'><b style='color:{sp.title_color}; font-size:1.15rem;'>{fmt(row.get('res'), 3)}</b> <small>{sp.unit}</small></td>"
                f"<td style='text-align:center;'><b>{fmt_int(row.get('fs'))}</b> Hz</td>"
                "</tr>"
            )
        chunks.append('</tbody></table></div>')
    chunks.append('</div>')
    st.markdown("\n".join(chunks), unsafe_allow_html=True)


def build_tsv(rows: list[dict[str, Any]], file_name: str, selected_types: list[str] | None = None, include_header: bool = True) -> str:
    if not rows:
        return ""
    selected_types = selected_types or ["aHV", "aHVwp", "pF"]
    has_zone = any("zone" in row for row in rows)
    groups: dict[tuple[Any, str], dict[str, dict[str, Any]]] = {}
    for row in rows:
        axes = row.get("axes", "")
        group_key = (row.get("zone"), axes) if has_zone else (None, axes)
        groups.setdefault(group_key, {})[row.get("type", "")] = row

    def sfx(axes_lbl: str) -> str:
        axes = parse_axes(axes_lbl)
        return "2" if axes and axes[0] >= 4 else "1"

    lines: list[str] = []
    if include_header:
        first_ax = next(iter(groups))[1]
        s = sfx(first_ax)
        hdr = ["Zone", "Axes"] if has_zone else ["Axes"]
        if "aHV" in selected_types:
            hdr += ["x Wh (m/s²)", "y Wh (m/s²)", "z Wh (m/s²)", "aHV Wh (m/s²)"]
        if "aHVwp" in selected_types:
            hdr += ["x Wp (m/s²)", "y Wp (m/s²)", "z Wp (m/s²)", "aHV(wp) Wp (m/s²)"]
        if "pF" in selected_types:
            hdr += ["AFx Flat_h (m/s²)", "AFy Flat_h (m/s²)", "AFz Flat_h (m/s²)", "AFv Flat_h (m/s²)", "pF Flat_h"]
        hdr += ["Fs (Hz)", "Start (s)", "End (s)", "Durée (s)", "Fichier"]
        lines.append("\t".join(hdr))

    for (zone_lbl, axes_lbl), grp in groups.items():
        ref = grp.get("aHV") or grp.get("aHVwp") or grp.get("pF")
        if not ref:
            continue
        row_out = [str(zone_lbl), axes_lbl] if has_zone else [axes_lbl]
        if "aHV" in selected_types:
            r = grp.get("aHV")
            row_out += [csv_num(r.get("x")), csv_num(r.get("y")), csv_num(r.get("z")), csv_num(r.get("res"))] if r else ["", "", "", ""]
        if "aHVwp" in selected_types:
            r = grp.get("aHVwp")
            row_out += [csv_num(r.get("x")), csv_num(r.get("y")), csv_num(r.get("z")), csv_num(r.get("res"))] if r else ["", "", "", ""]
        if "pF" in selected_types:
            r = grp.get("pF")
            row_out += [csv_num(r.get("x")), csv_num(r.get("y")), csv_num(r.get("z")), csv_num(r.get("afv")), csv_num(r.get("res"))] if r else ["", "", "", "", ""]
        row_out += [csv_int(ref.get("fs")), csv_num(ref.get("start")), csv_num(ref.get("end")), csv_num(ref.get("duree")), file_name]
        lines.append("\t".join(row_out))
    return "\n".join(lines)


def rows_to_detail_df(rows: list[dict[str, Any]], theme_name: str) -> pd.DataFrame:
    out = []
    for row in rows:
        sp = metric_specs(theme_name, row.get("type", "pF"))
        comps = "; ".join(f"{label}={fmt(value, 3)}" for label, _, value in components_for(row, row.get("type", "pF"), row.get("axes", "")) if fmt(value) != "—")
        item = {}
        if "zone" in row:
            item["Zone"] = row.get("zone")
        item.update({
            "Axes": row.get("axes", ""),
            "Indicateur": re.sub("<.*?>", "", sp.title),
            "Statut": sp.status,
            "Filtre / pondération": sp.filter_txt,
            "Composantes": comps,
            "Valeur finale": row.get("res"),
            "Unité": sp.unit,
            "Fs eff. (Hz)": row.get("fs"),
            "Start (s)": row.get("start"),
            "End (s)": row.get("end"),
            "Durée (s)": row.get("duree"),
        })
        out.append(item)
    return pd.DataFrame(out)


# ══════════════════════════════════════════════════════════════════════════════
# UI principale
# ══════════════════════════════════════════════════════════════════════════════


def initialize_state() -> None:
    defaults = {
        "skip": 15,
        "cols_txt": "0,1,2,3,4,5,6",
        "time0": True,
        "fe": 20_000.0,
        "start": 0.0,
        "end": 0.0,
        "zones_df": pd.DataFrame(columns=["Zone", "Start (s)", "End (s)"]),
        "result_rows": [],
        "multi_rows": pd.DataFrame(),
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def fs_options(fe: float) -> list[int]:
    if fe <= 0:
        return []
    max_pow = 2 ** int(np.floor(np.log2(fe)))
    opts = []
    v = max_pow
    while v >= 128:
        opts.append(int(v))
        v //= 2
    return opts or [int(round(fe))]


def apply_detection(file_bytes: bytes) -> None:
    info = scan_file_structure_bytes(file_bytes)
    st.session_state.skip = int(info["skip"])
    st.session_state.cols_txt = str(info["col_str"])
    st.session_state.time0 = bool(info["probably_time0"])


def estimate_fe_from_data(data, cols: list[int], time0: bool, current_fe: float) -> float:
    if not time0 or 0 not in cols:
        return current_fe
    raw_t = get_by_col(data, cols, 0)
    try:
        if TC is not None and hasattr(TC, "fs_detect"):
            fe = float(TC.fs_detect(raw_t))
        else:
            t = parse_time_vector_to_seconds(raw_t)
            if t is None or len(t) < 2:
                return current_fe
            dt = np.diff(t)
            fe = 1.0 / float(np.median(dt[dt > 0]))
        return float(round(fe, -1)) if fe >= 100 else float(round(fe, 2))
    except Exception:
        return current_fe


def main() -> None:
    initialize_state()

    with st.sidebar:
        st.header("Paramètres")
        theme_name = st.selectbox("Thème visuel", ["Clair", "Sombre"], index=0, help="La zone Résultats reprend exactement ce mode.")
        inject_css(theme_name)
        uploaded = st.file_uploader("Fichier de mesure", type=["txt", "csv", "tsv", "dat", "asc"])
        st.caption("Sur Streamlit, l'application lit un fichier téléversé : elle ne peut pas accéder à un chemin Windows local.")

        if uploaded is not None:
            file_bytes = uploaded.getvalue()
            info = scan_file_structure_bytes(file_bytes)
            st.info(
                f"Détection : skip={info['skip']} · colonnes={info['col_str']} · "
                f"délimiteur={repr(info['delimiter'])} · temps col.0={'oui' if info['probably_time0'] else 'non'}"
            )
            if st.button("↺ Utiliser la détection", use_container_width=True):
                apply_detection(file_bytes)
                st.rerun()
        else:
            file_bytes = b""

        st.number_input("Lignes d'en-tête à ignorer", min_value=0, max_value=100_000, key="skip")
        st.text_input("Colonnes à lire", key="cols_txt", help="Exemple : 0,1,2,3,4,5,6")
        st.checkbox("Colonne 0 = Temps", key="time0")
        st.number_input("Fe entrée (Hz)", min_value=0.001, max_value=10_000_000.0, step=100.0, key="fe", format="%.4f")

        st.divider()
        st.subheader("Rééchantillonnage")
        resample_enabled = st.checkbox("Activer", value=True)
        resample_ahv_only = st.checkbox("Uniquement aHV / aHV(wp)", value=False)
        opts = fs_options(float(st.session_state.fe))
        default_idx = opts.index(4096) if 4096 in opts else 0
        fs_new = st.selectbox("Fs_new", opts, index=default_idx) if opts else st.session_state.fe

        st.divider()
        st.subheader("Calculs")
        do_ahv = st.checkbox("aHV — pondéré Wh", value=True)
        do_wp = st.checkbox("aHV(wp) — pondéré Wp", value=True)
        do_pf = st.checkbox("pF — pondéré Flat_h", value=True)
        include_header = st.checkbox("En-tête dans les exports", value=True)

    st.markdown(
        """
        <div class="hav-title">
            <h1>WBV / HAV — Dépouillement web</h1>
            <p>Lecture fichier, timeline interactive, calculs aHV Wh / aHV(wp) Wp / pF Flat_h et exports Excel FR.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if TC is None:
        st.error(
            "Le module `fonctionTCi.py` n'est pas disponible. L'interface peut s'ouvrir, mais les calculs Wh/Wp/Flat_h ne peuvent pas être conservés. "
            f"Erreur import : {_TC_IMPORT_ERROR}"
        )

    if uploaded is None:
        st.markdown(
            """
            <div class="hav-card">
              <b>Charge un fichier de mesure pour commencer.</b><br>
              <span class="hav-note">Le portage Streamlit remplace les fenêtres PyQt par des composants web : upload, graphiques Plotly, formulaires et boutons de téléchargement.</span>
            </div>
            """,
            unsafe_allow_html=True,
        )
        render_results([], theme_name)
        return

    try:
        col_list = parse_columns_list(st.session_state.cols_txt)
    except Exception as ex:
        st.error(f"Liste de colonnes invalide : {ex}")
        return
    if not col_list:
        st.error("Renseigne au moins une colonne.")
        return

    try:
        with st.spinner("Lecture du fichier…"):
            data, cols, filepath = load_data_cached(file_bytes, uploaded.name, int(st.session_state.skip), tuple(col_list))
    except Exception as ex:
        st.error(str(ex))
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Fichier", uploaded.name)
    c2.metric("Échantillons", f"{len(data[0]) if data else 0:,}".replace(",", " "))
    c3.metric("Colonnes lues", ", ".join(str(c) for c in cols))
    c4.metric("Fe", f"{float(st.session_state.fe):.2f} Hz")

    if st.button("🔍 Estimer Fe depuis la colonne temps", disabled=not (st.session_state.time0 and 0 in cols)):
        st.session_state.fe = estimate_fe_from_data(data, cols, bool(st.session_state.time0), float(st.session_state.fe))
        st.rerun()

    fe = float(st.session_state.fe)
    time_sec = parse_time_vector_to_seconds(get_by_col(data, cols, 0)) if st.session_state.time0 and 0 in cols else None
    data_cols = [c for c in cols if not (st.session_state.time0 and c == 0)]
    if not data_cols:
        st.error("Aucune colonne de signal disponible. Si la colonne 0 est le temps, ajoute aussi des colonnes de vibration.")
        return

    if time_sec is not None and len(time_sec) > 1:
        end_auto = float(time_sec[-1])
    else:
        end_auto = len(get_by_col(data, cols, data_cols[0])) / fe
    if st.session_state.end == 0.0 or st.session_state.end > end_auto:
        st.session_state.end = float(end_auto)

    st.subheader("Timeline")
    tl_ctrl = st.columns([1.1, 1.1, 1, 1, 1])
    mode = tl_ctrl[0].selectbox("Affichage", ["Courbes 1/2/3", "Courbes 4/5/6", "Courbes 1→6", "Colonne seule"], index=0)
    synthetic = tl_ctrl[1].checkbox("Vue synthétique", value=True)

    if mode == "Courbes 1/2/3":
        candidates = [c for c in data_cols if c in [1, 2, 3]] or data_cols[:3]
    elif mode == "Courbes 4/5/6":
        candidates = [c for c in data_cols if c in [4, 5, 6]] or data_cols[-3:]
    elif mode == "Courbes 1→6":
        candidates = [c for c in data_cols if c in [1, 2, 3, 4, 5, 6]] or data_cols
    else:
        one = tl_ctrl[2].selectbox("Colonne", data_cols)
        candidates = [one]

    selected_plot_cols = st.multiselect(
        "Courbes affichées",
        options=candidates,
        default=candidates,
        format_func=lambda c: f"{c} ({DEFAULT_LABELS.get(c, f'Col {c}')})",
    )

    st.session_state.start = float(np.clip(float(st.session_state.start), 0.0, float(end_auto)))
    st.session_state.end = float(np.clip(float(st.session_state.end), 0.0, float(end_auto)))
    start = st.number_input("Start (s)", min_value=0.0, max_value=float(end_auto), step=0.01, format="%.4f", key="start")
    end = st.number_input("End (s)", min_value=0.0, max_value=float(end_auto), step=0.01, format="%.4f", key="end")
    if end < start:
        start, end = end, start
        st.session_state.start = float(start)
        st.session_state.end = float(end)
    if st.button("⇥ Plage complète"):
        st.session_state.start = 0.0
        st.session_state.end = float(end_auto)
        st.rerun()

    first_sig = np.asarray(get_by_col(data, cols, data_cols[0]), dtype=float)
    sig_ref = np.asarray(cut_signal(first_sig, fe, 0.0, end_auto), dtype=float)
    if time_sec is not None:
        t_cut = np.asarray(cut_signal(time_sec, fe, 0.0, end_auto), dtype=float)
    else:
        t_cut = np.arange(len(sig_ref), dtype=float) / fe

    sigs_plot: dict[int, np.ndarray] = {}
    labels: dict[int, str] = {}
    for c in selected_plot_cols:
        sigs_plot[c] = np.asarray(cut_signal(get_by_col(data, cols, c), fe, 0.0, end_auto), dtype=float)
        labels[c] = DEFAULT_LABELS.get(c, f"Col {c}")

    fig = make_timeline_figure(t_cut, sigs_plot, labels, start, end, theme_name, synthetic=synthetic)
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})

    with st.expander("Découpage automatique", expanded=False):
        if selected_plot_cols:
            ref_col = selected_plot_cols[0]
            sig_auto = np.asarray(sigs_plot.get(ref_col), dtype=float)
            abs_sig = np.abs(sig_auto)
            default_thr = float(np.mean(abs_sig) + 2.0 * np.std(abs_sig)) if abs_sig.size else 0.0
            auto_col1, auto_col2, auto_col3 = st.columns(3)
            threshold = auto_col1.number_input("Seuil", value=default_thr, format="%.6f")
            min_dur = auto_col2.number_input("Durée min. zone (s)", value=0.1, min_value=0.001, step=0.05, format="%.3f")
            gap = auto_col3.number_input("Fusion si écart < (s)", value=0.5, min_value=0.0, step=0.05, format="%.3f")
            if st.button("Détecter et ajouter aux zones"):
                zones = detect_vibration_zones(sig_auto, t_cut[:len(sig_auto)], threshold, min_dur, gap)
                if zones:
                    existing = st.session_state.zones_df.copy()
                    add = pd.DataFrame({"Zone": list(range(len(existing) + 1, len(existing) + len(zones) + 1)), "Start (s)": [z[0] for z in zones], "End (s)": [z[1] for z in zones]})
                    st.session_state.zones_df = pd.concat([existing, add], ignore_index=True)
                    st.success(f"{len(zones)} zone(s) ajoutée(s).")
                else:
                    st.warning("Aucune zone détectée avec ce seuil.")
        else:
            st.info("Sélectionne au moins une courbe sur la timeline.")

    st.subheader("Calcul simple")
    calc_cols = st.multiselect(
        "Colonnes de calcul",
        options=data_cols,
        default=[c for c in [1, 2, 3, 4, 5, 6] if c in data_cols][:6] or data_cols[:3],
        format_func=lambda c: f"{c} ({DEFAULT_LABELS.get(c, f'Col {c}')})",
        help="3 colonnes = un tri-axe ; 6 colonnes + mode dual = deux tri-axes.",
    )
    dual_mode = st.checkbox("Dual : calculer [1,2,3] ET [4,5,6] si 6 colonnes sont sélectionnées", value=True)

    button_cols = st.columns([1, 1, 1, 4])
    if button_cols[0].button("⚡ Calculer", type="primary", use_container_width=True):
        try:
            rows: list[dict[str, Any]] = []
            if len(calc_cols) == 1:
                if do_ahv:
                    rows.append(compute_1_axis(calc_cols[0], "aHV", data, cols, fe, start, end, resample_enabled, resample_ahv_only, float(fs_new)))
                if do_wp:
                    rows.append(compute_1_axis(calc_cols[0], "aHVwp", data, cols, fe, start, end, resample_enabled, resample_ahv_only, float(fs_new)))
                if do_pf:
                    rows.append(compute_1_axis(calc_cols[0], "pF", data, cols, fe, start, end, resample_enabled, resample_ahv_only, float(fs_new)))
            elif dual_mode and len(calc_cols) >= 6:
                rows += run_group(calc_cols[:3], data, cols, fe, start, end, resample_enabled, resample_ahv_only, float(fs_new), do_ahv, do_wp, do_pf)
                rows += run_group(calc_cols[3:6], data, cols, fe, start, end, resample_enabled, resample_ahv_only, float(fs_new), do_ahv, do_wp, do_pf)
            elif len(calc_cols) >= 3:
                rows += run_group(calc_cols[:3], data, cols, fe, start, end, resample_enabled, resample_ahv_only, float(fs_new), do_ahv, do_wp, do_pf)
            else:
                st.warning("Sélectionne 1 colonne pour le calcul mono-axe ou au moins 3 colonnes pour un tri-axe.")
            st.session_state.result_rows = rows
        except Exception as ex:
            st.error(f"Calcul impossible : {ex}")

    result_rows = st.session_state.get("result_rows", [])
    render_results(result_rows, theme_name)

    if result_rows:
        selected_export_types = []
        exp_cols = st.columns(6)
        if exp_cols[0].checkbox("Exporter aHV Wh", value=True):
            selected_export_types.append("aHV")
        if exp_cols[1].checkbox("Exporter aHV(wp) Wp", value=True):
            selected_export_types.append("aHVwp")
        if exp_cols[2].checkbox("Exporter pF Flat_h", value=True):
            selected_export_types.append("pF")
        tsv = build_tsv(result_rows, uploaded.name, selected_export_types, include_header)
        exp_cols[3].download_button("⬇ Télécharger TSV", data=tsv.encode("utf-8"), file_name="resultats_hav.tsv", mime="text/tab-separated-values", use_container_width=True)
        with st.expander("Table détail / copier-coller"):
            st.dataframe(rows_to_detail_df(result_rows, theme_name), use_container_width=True, hide_index=True)
            st.text_area("TSV prêt pour Excel FR", tsv, height=150)

    st.subheader("Multi-découpe")
    zc1, zc2, zc3 = st.columns([1, 1, 4])
    if zc1.button("➕ Ajouter plage courante", use_container_width=True):
        df = st.session_state.zones_df.copy()
        df.loc[len(df)] = [len(df) + 1, float(start), float(end)]
        st.session_state.zones_df = df
        st.rerun()
    if zc2.button("Vider", use_container_width=True):
        st.session_state.zones_df = pd.DataFrame(columns=["Zone", "Start (s)", "End (s)"])
        st.session_state.multi_rows = pd.DataFrame()
        st.rerun()

    zones_df = st.data_editor(
        st.session_state.zones_df,
        num_rows="dynamic",
        use_container_width=True,
        hide_index=True,
        column_config={
            "Zone": st.column_config.NumberColumn("Zone", step=1),
            "Start (s)": st.column_config.NumberColumn("Start (s)", format="%.4f"),
            "End (s)": st.column_config.NumberColumn("End (s)", format="%.4f"),
        },
        key="zones_editor",
    )
    st.session_state.zones_df = zones_df

    axes_choice = st.selectbox("Axes multi-découpe", ["[1,2,3]", "[4,5,6]", "[1,2,3] + [4,5,6]"], index=0)
    if st.button("⚡ Calculer toutes les zones", use_container_width=True):
        try:
            axes_sets: list[list[int]]
            if axes_choice == "[1,2,3]":
                axes_sets = [[1, 2, 3]]
            elif axes_choice == "[4,5,6]":
                axes_sets = [[4, 5, 6]]
            else:
                axes_sets = [[1, 2, 3], [4, 5, 6]]
            axes_sets = [[c for c in ax if c in data_cols] for ax in axes_sets]
            axes_sets = [ax for ax in axes_sets if len(ax) == 3]
            if not axes_sets:
                raise ValueError("Les colonnes nécessaires aux axes choisis ne sont pas présentes.")
            all_out_rows: list[dict[str, Any]] = []
            for i, z in zones_df.reset_index(drop=True).iterrows():
                zs = float(z.get("Start (s)", 0.0))
                ze = float(z.get("End (s)", 0.0))
                if ze <= zs:
                    continue
                for ax in axes_sets:
                    rows = run_group(ax, data, cols, fe, zs, ze, resample_enabled, resample_ahv_only, float(fs_new), do_ahv, do_wp, do_pf)
                    for r in rows:
                        r = dict(r)
                        r["zone"] = int(z.get("Zone", i + 1)) if not pd.isna(z.get("Zone", np.nan)) else i + 1
                        all_out_rows.append(r)
            if not all_out_rows:
                raise ValueError("Aucune zone valide à calculer.")
            st.session_state.multi_rows = rows_to_detail_df(all_out_rows, theme_name)
            st.session_state.multi_raw_rows = all_out_rows
        except Exception as ex:
            st.error(f"Multi-découpe impossible : {ex}")

    multi_df = st.session_state.get("multi_rows", pd.DataFrame())
    if isinstance(multi_df, pd.DataFrame) and not multi_df.empty:
        st.dataframe(multi_df, use_container_width=True, hide_index=True)
        raw_rows = st.session_state.get("multi_raw_rows", [])
        tsv_multi = build_tsv(raw_rows, uploaded.name, ["aHV", "aHVwp", "pF"], include_header)
        st.download_button("⬇ Télécharger multi-découpe TSV", data=tsv_multi.encode("utf-8"), file_name="multi_decoupe_hav.tsv", mime="text/tab-separated-values", use_container_width=True)

    with st.expander("Tracés avancés : brut / filtre / spectre", expanded=False):
        plot_cols = st.multiselect("Colonnes à tracer", data_cols, default=data_cols[:3], format_func=lambda c: f"{c} ({DEFAULT_LABELS.get(c, f'Col {c}')})", key="advanced_cols")
        f1 = st.selectbox("Filtre 1", FILTERS, index=FILTERS.index("wh"))
        f2 = st.selectbox("Filtre 2", FILTERS, index=FILTERS.index("flat"))
        if st.button("Tracer comparaison filtres") and plot_cols:
            try:
                raw_signals = [np.asarray(cut_signal(get_by_col(data, cols, c), fe, start, end), dtype=float) for c in plot_cols]
                t_adv = np.asarray(cut_signal(time_sec, fe, start, end), dtype=float) if time_sec is not None else np.arange(len(raw_signals[0])) / fe
                fig_cmp = go.Figure()
                sigs_for_spectrum = []
                labels_for_spectrum = []
                for i, c in enumerate(plot_cols):
                    raw = raw_signals[i]
                    filt1, fe1 = cut_filter(data, cols, c, fe, start, end, f1, "plot", resample_enabled, resample_ahv_only, float(fs_new))
                    filt2, fe2 = cut_filter(data, cols, c, fe, start, end, f2, "plot", resample_enabled, resample_ahv_only, float(fs_new))
                    xd, yd = downsample_xy(t_adv[:len(raw)], raw)
                    fig_cmp.add_trace(go.Scattergl(x=xd, y=yd, mode="lines", name=f"{DEFAULT_LABELS.get(c, c)} brut", line=dict(width=1)))
                    t_f = np.arange(len(filt1)) / fe1 + start
                    xd, yd = downsample_xy(t_f, filt1)
                    fig_cmp.add_trace(go.Scattergl(x=xd, y=yd, mode="lines", name=f"{DEFAULT_LABELS.get(c, c)} {f1}", line=dict(width=1)))
                    t_f2 = np.arange(len(filt2)) / fe2 + start
                    xd, yd = downsample_xy(t_f2, filt2)
                    fig_cmp.add_trace(go.Scattergl(x=xd, y=yd, mode="lines", name=f"{DEFAULT_LABELS.get(c, c)} {f2}", line=dict(width=1)))
                    sigs_for_spectrum.extend([raw, filt1, filt2])
                    labels_for_spectrum.extend([f"{DEFAULT_LABELS.get(c, c)} brut", f"{DEFAULT_LABELS.get(c, c)} {f1}", f"{DEFAULT_LABELS.get(c, c)} {f2}"])
                fig_cmp.update_layout(template=THEMES[theme_name]["plot_template"], height=450, margin=dict(l=30, r=20, t=30, b=35))
                fig_cmp.update_xaxes(title="Temps (s)")
                fig_cmp.update_yaxes(title="m/s²")
                st.plotly_chart(fig_cmp, use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
                st.plotly_chart(make_spectrum_figure(sigs_for_spectrum, fe, labels_for_spectrum, theme_name), use_container_width=True, config={"displaylogo": False, "scrollZoom": True})
            except Exception as ex:
                st.error(f"Tracé impossible : {ex}")


if __name__ == "__main__":
    main()
