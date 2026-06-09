from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any


DEFAULT_ABLATION_DIR = Path(r"C:\Users\Lenovo\Desktop\ablation")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "index.html"


def _num(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _as_bool(value: str | bool | None) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"true", "1", "yes"}


def _file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def _load_rows(ablation_dir: Path) -> list[dict[str, Any]]:
    csv_path = ablation_dir / "all_pairs.csv"
    rows: list[dict[str, Any]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for raw in reader:
            is_self = _as_bool(raw.get("is_self"))
            baseline_rank = _num(raw.get("E0_baseline_rank"))
            query_id = str(raw.get("query_id", ""))
            candidate_id = str(raw.get("candidate_id", ""))
            visual_uri = ""
            if not is_self and baseline_rank is not None:
                visual_path = (
                    ablation_dir
                    / "visualizations"
                    / query_id
                    / f"rank_{int(baseline_rank):03d}_{query_id}_vs_{candidate_id}.png"
                )
                if visual_path.exists():
                    visual_uri = _file_uri(visual_path)

            rows.append(
                {
                    "q": query_id,
                    "c": candidate_id,
                    "self": is_self,
                    "fill": _num(raw.get("fill_iou")),
                    "top": _num(raw.get("top_edge_iou")),
                    "bottom": _num(raw.get("bottom_edge_iou")),
                    "edge": _num(raw.get("edge_score")),
                    "aspect": _num(raw.get("aspect_penalty")),
                    "baseBeforeGamma": _num(raw.get("baseline_before_gamma")),
                    "E0": _num(raw.get("E0_baseline")),
                    "E1": _num(raw.get("E1_no_fill_iou")),
                    "E2": _num(raw.get("E2_no_aspect")),
                    "E3": _num(raw.get("E3_no_gamma")),
                    "r0": baseline_rank,
                    "r1": _num(raw.get("E1_no_fill_iou_rank")),
                    "r2": _num(raw.get("E2_no_aspect_rank")),
                    "r3": _num(raw.get("E3_no_gamma_rank")),
                    "visual": visual_uri,
                }
            )
    return rows


def _load_manifest(ablation_dir: Path) -> dict[str, str]:
    manifest_path = ablation_dir / "image_manifest.csv"
    if not manifest_path.exists():
        return {}
    result: dict[str, str] = {}
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_id = str(row.get("image_id", ""))
            path = str(row.get("path", ""))
            if image_id:
                result[image_id] = path
    return result


def _load_summary(ablation_dir: Path) -> dict[str, Any]:
    summary_path = ablation_dir / "ablation_summary.json"
    return json.loads(summary_path.read_text(encoding="utf-8"))


def _build_html(ablation_dir: Path, rows: list[dict[str, Any]], summary: dict[str, Any], manifest: dict[str, str]) -> str:
    data = {
        "ablationDir": str(ablation_dir),
        "summaryChart": _file_uri(ablation_dir / "visualizations" / "ablation_summary.png"),
        "rows": rows,
        "summary": summary,
        "manifest": manifest,
    }
    data_json = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    title = "整车轮廓消融实验结果对比"
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      color-scheme: dark;
      --bg: #11161c;
      --panel: #1a222b;
      --panel-2: #202a34;
      --line: #34424f;
      --text: #edf4ff;
      --muted: #9fb0c1;
      --accent: #27b5e8;
      --green: #45d17c;
      --yellow: #f5c84b;
      --red: #f26b6b;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 3;
      padding: 20px 28px;
      background: rgba(17, 22, 28, .94);
      border-bottom: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }}
    h1 {{ margin: 0 0 8px; font-size: 26px; font-weight: 700; }}
    .sub {{ color: var(--muted); font-size: 13px; }}
    main {{ padding: 24px 28px 42px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(180px, 1fr));
      gap: 14px;
      margin-bottom: 20px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }}
    .card h2, .section h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .metric {{ font-size: 34px; font-weight: 800; line-height: 1; }}
    .metric small {{ color: var(--muted); font-size: 13px; font-weight: 500; }}
    .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      margin-top: 18px;
    }}
    .summary-layout {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(360px, 1.2fr);
      gap: 18px;
      align-items: start;
    }}
    .chart {{
      width: 100%;
      max-height: 520px;
      object-fit: contain;
      background: #151a20;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid #2d3843;
      text-align: right;
      white-space: nowrap;
    }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    th {{ color: #c4d2e1; background: #18202a; position: sticky; top: 0; z-index: 1; }}
    tr:hover {{ background: #202b36; }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: end;
      margin-bottom: 14px;
    }}
    label {{ display: grid; gap: 6px; color: var(--muted); font-size: 12px; }}
    select, input {{
      height: 38px;
      min-width: 150px;
      border: 1px solid #52606d;
      border-radius: 7px;
      background: #0f151b;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }}
    button {{
      height: 38px;
      border: 0;
      border-radius: 7px;
      background: var(--accent);
      color: white;
      padding: 0 16px;
      font-weight: 700;
      cursor: pointer;
    }}
    .table-wrap {{
      max-height: 520px;
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 8px;
    }}
    .pill {{
      display: inline-flex;
      align-items: center;
      min-width: 58px;
      justify-content: center;
      border-radius: 999px;
      padding: 3px 9px;
      background: #101820;
      border: 1px solid #364656;
      font-weight: 700;
    }}
    .high {{ color: var(--green); }}
    .mid {{ color: var(--yellow); }}
    .low {{ color: var(--red); }}
    .viewer {{
      display: grid;
      grid-template-columns: minmax(360px, 1fr) 360px;
      gap: 18px;
      align-items: start;
    }}
    .visual {{
      width: 100%;
      background: #080b0f;
      border: 1px solid var(--line);
      border-radius: 8px;
      min-height: 260px;
      object-fit: contain;
    }}
    .kv {{
      display: grid;
      grid-template-columns: 128px 1fr;
      gap: 8px 10px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }}
    .kv strong {{ color: var(--text); }}
    .hint {{ color: var(--muted); font-size: 13px; margin-top: 10px; }}
    @media (max-width: 1100px) {{
      .grid, .summary-layout, .viewer {{ grid-template-columns: 1fr; }}
      header {{ position: static; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(title)}</h1>
    <div class="sub" id="subtitle"></div>
  </header>
  <main>
    <section class="grid" id="experimentCards"></section>

    <section class="section summary-layout">
      <div>
        <h2>实验总体统计</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr><th>实验</th><th>均值</th><th>标准差</th><th>最小</th><th>最大</th><th>与基线排名相关</th></tr>
            </thead>
            <tbody id="summaryRows"></tbody>
          </table>
        </div>
      </div>
      <img class="chart" id="summaryChart" alt="ablation summary chart" />
    </section>

    <section class="section">
      <h2>配对结果浏览</h2>
      <div class="controls">
        <label>查询图 ID<select id="querySelect"></select></label>
        <label>候选图过滤<input id="candidateFilter" placeholder="例如 49" /></label>
        <label>排序指标<select id="sortSelect">
          <option value="r0">E0 基线排名</option>
          <option value="E0">E0 基线分</option>
          <option value="E1">E1 去主体 IoU</option>
          <option value="E2">E2 去长宽惩罚</option>
          <option value="E3">E3 去 gamma</option>
          <option value="fill">主体 IoU</option>
          <option value="top">非底部边缘 IoU</option>
          <option value="bottom">底部边缘 IoU</option>
        </select></label>
        <label>显示 Top-N<input id="topN" type="number" min="1" max="100" value="20" /></label>
        <button id="applyBtn">刷新</button>
      </div>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Query</th><th>Candidate</th><th>Rank</th>
              <th>E0</th><th>E1</th><th>ΔE1</th><th>E2</th><th>ΔE2</th><th>E3</th><th>ΔE3</th>
              <th>主体</th><th>上边缘</th><th>底边缘</th><th>长宽惩罚</th>
            </tr>
          </thead>
          <tbody id="pairRows"></tbody>
        </table>
      </div>
      <div class="hint">点击任意配对行可在下方查看对应可视化图片。</div>
    </section>

    <section class="section viewer">
      <img class="visual" id="pairVisual" alt="pair visualization" />
      <div>
        <h2>当前配对</h2>
        <div class="kv" id="detailBox"></div>
      </div>
    </section>
  </main>

  <script id="ablation-data" type="application/json">{data_json}</script>
  <script>
    const DATA = JSON.parse(document.getElementById('ablation-data').textContent);
    const experiments = [
      ['E0_baseline', 'E0', '基线'],
      ['E1_no_fill_iou', 'E1', '去主体 IoU'],
      ['E2_no_aspect', 'E2', '去长宽惩罚'],
      ['E3_no_gamma', 'E3', '去 gamma']
    ];
    const fmt = (v, d = 1) => v === null || v === undefined || Number.isNaN(Number(v)) ? '-' : Number(v).toFixed(d);
    const scoreClass = v => Number(v) >= 85 ? 'high' : Number(v) >= 70 ? 'mid' : 'low';
    const byId = id => document.getElementById(id);

    function init() {{
      byId('subtitle').textContent = `${{DATA.summary.image_count}} 张图片，${{DATA.summary.pair_count}} 个有序配对，结果目录：${{DATA.ablationDir}}`;
      byId('summaryChart').src = DATA.summaryChart;
      renderCards();
      renderSummaryRows();
      const queries = [...new Set(DATA.rows.map(r => r.q))].sort((a, b) => Number(a) - Number(b));
      byId('querySelect').innerHTML = queries.map(q => `<option value="${{q}}">${{q}}</option>`).join('');
      byId('applyBtn').addEventListener('click', renderPairs);
      byId('querySelect').addEventListener('change', renderPairs);
      byId('sortSelect').addEventListener('change', renderPairs);
      byId('candidateFilter').addEventListener('input', renderPairs);
      byId('topN').addEventListener('change', renderPairs);
      renderPairs();
    }}

    function renderCards() {{
      byId('experimentCards').innerHTML = experiments.map(([key, short, label]) => {{
        const s = DATA.summary.experiments[key];
        return `<article class="card">
          <h2>${{short}} · ${{label}}</h2>
          <div class="metric ${{scoreClass(s.mean_score)}}">${{fmt(s.mean_score, 2)}} <small>均值</small></div>
          <div class="hint">std ${{fmt(s.std_score, 2)}} · min ${{fmt(s.min_score)}} · max ${{fmt(s.max_score)}}</div>
        </article>`;
      }}).join('');
    }}

    function renderSummaryRows() {{
      byId('summaryRows').innerHTML = experiments.map(([key, short, label]) => {{
        const s = DATA.summary.experiments[key];
        return `<tr>
          <td>${{short}} · ${{label}}</td>
          <td><span class="pill ${{scoreClass(s.mean_score)}}">${{fmt(s.mean_score, 2)}}</span></td>
          <td>${{fmt(s.std_score, 2)}}</td>
          <td>${{fmt(s.min_score)}}</td>
          <td>${{fmt(s.max_score)}}</td>
          <td>${{fmt(s.spearman_vs_baseline, 4)}}</td>
        </tr>`;
      }}).join('');
    }}

    function renderPairs() {{
      const q = byId('querySelect').value;
      const filter = byId('candidateFilter').value.trim();
      const sortKey = byId('sortSelect').value;
      const topN = Math.max(1, Number(byId('topN').value || 20));
      let rows = DATA.rows.filter(r => r.q === q && !r.self);
      if (filter) rows = rows.filter(r => String(r.c).includes(filter));
      rows.sort((a, b) => {{
        if (sortKey === 'r0') return (a.r0 ?? 999999) - (b.r0 ?? 999999);
        return (b[sortKey] ?? -999999) - (a[sortKey] ?? -999999);
      }});
      rows = rows.slice(0, topN);
      byId('pairRows').innerHTML = rows.map((r, idx) => rowHtml(r, idx)).join('');
      document.querySelectorAll('[data-row-index]').forEach(el => {{
        el.addEventListener('click', () => showDetail(rows[Number(el.dataset.rowIndex)]));
      }});
      if (rows.length) showDetail(rows[0]);
    }}

    function rowHtml(r, idx) {{
      const d1 = r.E1 - r.E0;
      const d2 = r.E2 - r.E0;
      const d3 = r.E3 - r.E0;
      return `<tr data-row-index="${{idx}}">
        <td>${{r.q}}</td><td>${{r.c}}</td><td>${{fmt(r.r0, 0)}}</td>
        <td><span class="pill ${{scoreClass(r.E0)}}">${{fmt(r.E0)}}</span></td>
        <td>${{fmt(r.E1)}}</td><td>${{signed(d1)}}</td>
        <td>${{fmt(r.E2)}}</td><td>${{signed(d2)}}</td>
        <td>${{fmt(r.E3)}}</td><td>${{signed(d3)}}</td>
        <td>${{fmt(r.fill)}}</td><td>${{fmt(r.top)}}</td><td>${{fmt(r.bottom)}}</td><td>${{fmt((r.aspect ?? 0) * 100)}}</td>
      </tr>`;
    }}

    function signed(v) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
      const n = Number(v);
      const cls = n >= 0 ? 'high' : 'low';
      return `<span class="${{cls}}">${{n >= 0 ? '+' : ''}}${{n.toFixed(1)}}</span>`;
    }}

    function showDetail(r) {{
      byId('pairVisual').src = r.visual || '';
      byId('pairVisual').alt = r.visual ? `${{r.q}} vs ${{r.c}}` : '未找到可视化图片';
      byId('detailBox').innerHTML = `
        <strong>Query</strong><span>${{r.q}}</span>
        <strong>Candidate</strong><span>${{r.c}}</span>
        <strong>E0 基线分</strong><span>${{fmt(r.E0)}}，Rank ${{fmt(r.r0, 0)}}</span>
        <strong>E1 去主体</strong><span>${{fmt(r.E1)}}，变化 ${{plainSigned(r.E1 - r.E0)}}</span>
        <strong>E2 去长宽</strong><span>${{fmt(r.E2)}}，变化 ${{plainSigned(r.E2 - r.E0)}}</span>
        <strong>E3 去 gamma</strong><span>${{fmt(r.E3)}}，变化 ${{plainSigned(r.E3 - r.E0)}}</span>
        <strong>主体 IoU</strong><span>${{fmt(r.fill)}}</span>
        <strong>非底部边缘</strong><span>${{fmt(r.top)}}</span>
        <strong>底部边缘</strong><span>${{fmt(r.bottom)}}</span>
        <strong>图片路径</strong><span>${{r.visual || '未生成'}}</span>
      `;
    }}

    function plainSigned(v) {{
      if (v === null || v === undefined || Number.isNaN(Number(v))) return '-';
      return `${{v >= 0 ? '+' : ''}}${{Number(v).toFixed(1)}}`;
    }}

    init();
  </script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a static HTML viewer for contour ablation results.")
    parser.add_argument("--ablation-dir", default=str(DEFAULT_ABLATION_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    ablation_dir = Path(args.ablation_dir).resolve()
    output = Path(args.output).resolve()
    rows = _load_rows(ablation_dir)
    summary = _load_summary(ablation_dir)
    manifest = _load_manifest(ablation_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(_build_html(ablation_dir, rows, summary, manifest), encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
