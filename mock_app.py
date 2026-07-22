from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parent
RESULT_ROOT = ROOT / "result"
REPORT_ROOT = RESULT_ROOT / "reports"
LATEST_REPORT = RESULT_ROOT / "latest_report.json"
LATEST_MD = RESULT_ROOT / "latest_report.md"


@dataclass(frozen=True)
class ReportItem:
    label: str
    path: Path


def _safe_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except Exception:
        return None


def _normalize_path_hint(raw: str) -> str:
    s = str(raw).strip()
    s = s.replace("\\", "/")
    s = re.sub(r"/+", "/", s)
    return s


def _resolve_maybe_mounted_path(raw: Any) -> Path | None:
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    p = Path(s)
    if p.exists():
        return p

    hint = _normalize_path_hint(s).lower()
    for anchor in ("/similarity_api/", "/result/", "/img/", "/models/", "/configs/"):
        idx = hint.find(anchor)
        if idx >= 0:
            tail = _normalize_path_hint(s)[idx + 1 :]
            cand = ROOT / Path(tail)
            if cand.exists():
                return cand

    base = Path(_normalize_path_hint(s)).name
    if base:
        hits = list(RESULT_ROOT.rglob(base))
        if hits:
            return hits[0]
    return None


@st.cache_data(show_spinner=False)
def _list_report_items() -> list[ReportItem]:
    items: list[ReportItem] = []
    if LATEST_REPORT.exists():
        items.append(ReportItem(label="最新一次（latest_report.json）", path=LATEST_REPORT))
    if REPORT_ROOT.exists():
        for p in sorted(REPORT_ROOT.glob("*.json"), reverse=True):
            items.append(ReportItem(label=p.name, path=p))
    return items


@st.cache_data(show_spinner=False)
def _load_report(path: str) -> dict[str, Any]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    return payload


def _score_badge(score: float | None) -> str:
    if score is None:
        return "—"
    if score >= 85:
        return "高度相似"
    if score >= 70:
        return "局部相似"
    if score >= 55:
        return "差异明显"
    return "差异很大"


def _css() -> str:
    return """
<style>
  :root{
    --bg0:#06080b;
    --bg1:#0b0f16;
    --card:rgba(255,255,255,.06);
    --card2:rgba(255,255,255,.08);
    --stroke:rgba(255,255,255,.10);
    --muted:rgba(255,255,255,.70);
    --muted2:rgba(255,255,255,.55);
    --hi:#7dd3fc;
    --ok:#34d399;
    --warn:#fbbf24;
    --bad:#fb7185;
    --mono: ui-monospace, "Cascadia Mono", "JetBrains Mono", Consolas, monospace;
    --sans: "HarmonyOS Sans SC", "Source Han Sans SC", "Microsoft YaHei UI", "PingFang SC", system-ui, sans-serif;
    --r: 18px;
    --shadow: 0 16px 40px rgba(0,0,0,.35);
  }

  html, body, [data-testid="stAppViewContainer"]{
    background:
      radial-gradient(1200px 600px at 15% -10%, rgba(125,211,252,.18), transparent 60%),
      radial-gradient(900px 500px at 90% 10%, rgba(52,211,153,.12), transparent 55%),
      linear-gradient(180deg, var(--bg0), var(--bg1));
    color: rgba(255,255,255,.92);
    font-family: var(--sans);
  }

  [data-testid="stHeader"]{ background: transparent; }
  [data-testid="stSidebar"]{
    background: rgba(0,0,0,.35);
    border-right: 1px solid rgba(255,255,255,.08);
  }

  .card{
    background: var(--card);
    border: 1px solid var(--stroke);
    border-radius: var(--r);
    box-shadow: var(--shadow);
    padding: 14px 14px;
  }

  .kpi{
    display:flex;
    gap:12px;
    align-items:baseline;
    flex-wrap:wrap;
  }

  .kpi .label{
    color: var(--muted2);
    font-size: 12px;
    letter-spacing: .08em;
    text-transform: uppercase;
  }

  .kpi .value{
    font-family: var(--mono);
    font-size: 28px;
    line-height: 1;
  }

  .scorebar{
    height: 10px;
    width: 100%;
    border-radius: 999px;
    background: rgba(255,255,255,.08);
    border: 1px solid rgba(255,255,255,.10);
    overflow:hidden;
  }
  .scorebar > span{
    display:block;
    height:100%;
    width: var(--w);
    background: linear-gradient(90deg, rgba(125,211,252,.95), rgba(52,211,153,.95));
    box-shadow: 0 0 0 1px rgba(0,0,0,.35) inset;
  }

  .pill{
    display:inline-flex;
    align-items:center;
    gap:8px;
    padding:6px 10px;
    border-radius: 999px;
    background: rgba(255,255,255,.07);
    border: 1px solid rgba(255,255,255,.10);
    color: rgba(255,255,255,.86);
    font-size: 12px;
  }
  .pill b{ font-family: var(--mono); font-weight: 650; }

  .muted{ color: var(--muted2); }

  .grid2{
    display:grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
  }

  .grid3{
    display:grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
  }

  .tablelike{
    border: 1px solid rgba(255,255,255,.10);
    border-radius: 14px;
    overflow: hidden;
    background: rgba(0,0,0,.22);
  }
  .row{
    display:flex;
    gap: 12px;
    justify-content: space-between;
    align-items: center;
    padding: 10px 12px;
    border-top: 1px solid rgba(255,255,255,.06);
  }
  .row:first-child{ border-top:none; }
  .row .k{
    font-family: var(--mono);
    font-size: 12px;
    color: var(--muted2);
  }
  .row .v{
    font-family: var(--mono);
    font-size: 13px;
    color: rgba(255,255,255,.90);
    text-align:right;
  }
  .row .v .sub{
    display:block;
    font-family: var(--sans);
    font-size: 12px;
    color: rgba(255,255,255,.62);
    margin-top: 2px;
    text-align:right;
  }

  .hr{
    height: 1px;
    background: rgba(255,255,255,.10);
    margin: 12px 0;
  }
</style>
"""


def _render_scorebar(score: float | None) -> None:
    v = 0.0 if score is None else max(0.0, min(100.0, float(score)))
    st.markdown(f'<div class="scorebar" style="--w:{v:.1f}%"><span></span></div>', unsafe_allow_html=True)


def _render_path(path: Path | None) -> None:
    if path is None:
        st.caption("路径：—")
        return
    st.caption(f"路径：{path}")


def _render_image(path: Path | None, caption: str) -> None:
    if path is None or not path.exists():
        st.markdown(f'<span class="muted">{caption}: 未找到文件</span>', unsafe_allow_html=True)
        return
    st.image(str(path), caption=caption, use_container_width=True)


def _render_summary(report: dict[str, Any]) -> None:
    run_id = str(report.get("run_id") or "")
    query = report.get("query") or {}
    query_id = str(query.get("id") or "")
    gallery_count = int(report.get("gallery_count") or 0)
    parts = report.get("parts") or []
    features = report.get("features") or []

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="kpi">
  <div>
    <div class="label">RUN</div>
    <div class="value">{run_id}</div>
  </div>
  <span class="pill"><span>Query</span><b>{query_id}</b></span>
  <span class="pill"><span>Gallery</span><b>{gallery_count}</b></span>
  <span class="pill"><span>Parts</span><b>{len(parts)}</b></span>
  <span class="pill"><span>Features</span><b>{len(features)}</b></span>
</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="hr"></div>', unsafe_allow_html=True)
    st.write(
        {
            "parts": parts,
            "features": features,
        }
    )
    st.markdown("</div>", unsafe_allow_html=True)


def _render_rank_list(report: dict[str, Any], selected_id: str | None) -> str | None:
    rows = report.get("results") or []
    ids = [str(r.get("candidate_id") or "") for r in rows if str(r.get("candidate_id") or "")]
    if not ids:
        st.info("报告中没有 results。")
        return None

    default = selected_id if selected_id in ids else ids[0]
    picked = st.radio("候选结果", ids, index=ids.index(default), horizontal=False)

    row = next((x for x in rows if str(x.get("candidate_id") or "") == picked), None)
    if row is None:
        return picked

    score = _safe_float(row.get("final_score"))
    contour = _safe_float(row.get("contour_score"))
    part = _safe_float(row.get("part_score"))

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(
        f"""
<div class="grid3">
  <div>
    <div class="label">FINAL</div>
    <div class="value" style="font-family:var(--mono); font-size:28px">{'—' if score is None else f'{score:.1f}'}</div>
  </div>
  <div>
    <div class="label">CONTOUR</div>
    <div class="value" style="font-family:var(--mono); font-size:28px">{'—' if contour is None else f'{contour:.1f}'}</div>
  </div>
  <div>
    <div class="label">PARTS</div>
    <div class="value" style="font-family:var(--mono); font-size:28px">{'—' if part is None else f'{part:.1f}'}</div>
  </div>
</div>
""",
        unsafe_allow_html=True,
    )
    _render_scorebar(score)
    st.markdown(f'<span class="pill">判定：<b>{_score_badge(score)}</b></span>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    return picked


def _render_detail(report: dict[str, Any], candidate_id: str) -> None:
    rows = report.get("results") or []
    row = next((x for x in rows if str(x.get("candidate_id") or "") == candidate_id), None)
    if row is None:
        st.info("未找到候选项。")
        return

    query = report.get("query") or {}
    query_path = _resolve_maybe_mounted_path(query.get("path"))
    outputs = report.get("outputs") or {}
    view_name = str(report.get("view") or "front")
    query_label_dir = _resolve_maybe_mounted_path(
        outputs.get(f"{view_name}_label")
        or outputs.get("label_dir")
        or outputs.get("front_label")
    )

    candidate_path = _resolve_maybe_mounted_path(row.get("candidate_path"))
    contour_path = _resolve_maybe_mounted_path(row.get("contour_diff_image"))

    st.subheader("结果详情")

    c1, c2 = st.columns([1, 1], gap="large")
    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Query A**")
        _render_image(query_path, "Query 原图")
        if query_label_dir and query_label_dir.exists():
            hits = list(query_label_dir.glob(f"*{str(query.get('id') or '')}*.png"))
            _render_image(hits[0] if hits else None, "Query 标注图")
        st.markdown("</div>", unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Candidate**")
        _render_image(candidate_path, f"候选原图：{candidate_id}")
        _render_image(contour_path, "轮廓差异图")
        st.markdown("</div>", unsafe_allow_html=True)

    analysis = row.get("analysis") or []
    if analysis:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**判定要点**")
        for x in analysis:
            st.write(f"- {x}")
        st.markdown("</div>", unsafe_allow_html=True)

    part_scores: dict[str, Any] = row.get("part_scores") or {}
    if not part_scores:
        st.info("该候选没有部件相似度明细。")
        return

    st.subheader("部件明细")
    for part_name, detail in part_scores.items():
        q_part = _resolve_maybe_mounted_path((detail or {}).get("query_path"))
        c_part = _resolve_maybe_mounted_path((detail or {}).get("candidate_path"))
        weights = (detail or {}).get("weights_used") or {}

        fused = _safe_float((detail or {}).get("fused"))

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**{part_name}**  <span class='pill'>Fused <b>{'—' if fused is None else f'{fused:.1f}'}</b></span>", unsafe_allow_html=True)
        _render_scorebar(fused)

        pc1, pc2 = st.columns([1, 1], gap="large")
        with pc1:
            _render_image(q_part, f"{part_name} - Query")
        with pc2:
            _render_image(c_part, f"{part_name} - Candidate")

        features = [k for k in ("clip", "dino", "ssim", "edge") if k in detail]
        if features:
            st.markdown('<div class="tablelike">', unsafe_allow_html=True)
            for feat in features:
                sc = _safe_float((detail or {}).get(feat))
                w = _safe_float(weights.get(feat))
                st.markdown(
                    f"""
<div class="row">
  <div class="k">{feat}</div>
  <div class="v">
    {'—' if sc is None else f'{sc:.1f}'}
    <span class="sub">w={'—' if w is None else f'{w:.2f}'}</span>
  </div>
</div>
""",
                    unsafe_allow_html=True,
                )
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(page_title="相似度报告（Mock）", page_icon="report", layout="wide")
    st.markdown(_css(), unsafe_allow_html=True)
    st.title("相似度报告（Mock）")
    st.caption("只读取 ./result 历史报告，不触发任何模型推理。")

    items = _list_report_items()
    if not items:
        st.error("未找到历史报告。请先跑一次 pipeline，确保 ./result/reports 或 ./result/latest_report.json 存在。")
        return

    with st.sidebar:
        st.subheader("选择报告")
        choice = st.selectbox("Report", options=items, format_func=lambda x: x.label)
        st.markdown('<span class="muted">提示：CPU 环境下建议先用该页面做结果回放与 UI 调整。</span>', unsafe_allow_html=True)

    report = _load_report(str(choice.path))
    _render_summary(report)

    st.subheader("排名列表")
    selected = st.session_state.get("mock_selected_candidate")
    picked = _render_rank_list(report, selected_id=selected)
    if picked:
        st.session_state["mock_selected_candidate"] = picked
        _render_detail(report, picked)

    md_path = LATEST_MD if choice.path == LATEST_REPORT else (REPORT_ROOT / f"{str(report.get('run_id') or '')}.md")
    md_real = md_path if md_path.exists() else None
    with st.expander("查看 Markdown 报告"):
        if md_real is None:
            st.info("未找到对应 Markdown 报告。")
        else:
            st.markdown(md_real.read_text(encoding="utf-8"))
            st.caption(str(md_real))


if __name__ == "__main__":
    main()
