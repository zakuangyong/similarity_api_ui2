from __future__ import annotations

from html import escape
from pathlib import Path
from urllib.parse import quote

import streamlit as st

from .config import DEFAULT_GALLERY_DIR, PART_LABELS
from .utils import (
    _find_label,
    _fmt_score,
    _gallery_image_paths,
    _html_block,
    _img_to_data_uri,
    _part_visuals,
    _thumb_uri_for_path,
)


PROTOTYPE_CSS = """
<style>
    :root {
        --bg: #07090c;
        --panel: #1f252e;
        --panel-2: #15191f;
        --card: #10161d;
        --card-2: #0c1117;
        --line: rgba(255,255,255,.10);
        --line-strong: rgba(255,255,255,.16);
        --text: #edf4ff;
        --muted: #9aa8b8;
        --muted-2: #6f7f91;
        --accent: #2ea8e5;
        --accent-dark: #1479b8;
        --good: #34d399;
        --warn: #fbbf24;
        --danger: #f87171;
        --radius: 10px;
        --shadow: 0 20px 60px rgba(0,0,0,.32);
        color-scheme: dark;
        font-family: "Microsoft YaHei UI", "PingFang SC", "Source Han Sans SC", system-ui, sans-serif;
    }
    * { box-sizing: border-box; }
    html, body, .stApp, [data-testid="stAppViewContainer"] {
        background: var(--bg) !important;
        color: var(--text) !important;
    }
    body { margin: 0; min-width: 320px; }
    #MainMenu, footer, header[data-testid="stHeader"], [data-testid="stToolbar"],
    [data-testid="stDecoration"], .stDeployButton { display: none !important; }
    [data-testid="stMainBlockContainer"] {
        max-width: none !important;
        padding: 0 !important;
    }
    [data-testid="stVerticalBlock"] { gap: 0 !important; }
    .block-container { padding: 0 !important; }
    div[data-testid="stColumn"] > div[data-testid="stVerticalBlock"] {
        min-height: calc(100dvh - 106px);
    }
    div[data-testid="stHorizontalBlock"] {
        padding: 14px 30px 16px;
        min-height: calc(100dvh - 76px);
        gap: 18px !important;
        flex-wrap: nowrap !important;
        align-items: stretch !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
        flex: 0 0 380px !important;
        width: 380px !important;
        max-width: 380px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
        flex: 1 1 0 !important;
        width: calc(100% - 398px) !important;
        max-width: calc(100% - 398px) !important;
        min-width: 0 !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child > div[data-testid="stVerticalBlock"] {
        background: var(--panel);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 10px;
        box-shadow: var(--shadow);
        overflow: hidden;
        padding: 28px 46px 18px;
        gap: 16px !important;
    }
    div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child > div[data-testid="stVerticalBlock"] > div:last-child {
        margin-top: auto;
    }
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: rgba(12,17,23,.25); }
    ::-webkit-scrollbar-thumb { background: rgba(154,168,184,.34); border-radius: 999px; }

    .app-header {
        position: sticky;
        top: 0;
        z-index: 30;
        height: 76px;
        display: grid;
        grid-template-columns: 260px minmax(0, 1fr);
        align-items: center;
        gap: 24px;
        padding: 0 32px;
        background: #1f252e;
        border-bottom: 1px solid rgba(255,255,255,.12);
        box-shadow: 0 12px 32px rgba(0,0,0,.22);
    }
    .brand {
        display: inline-flex;
        align-items: center;
        gap: 12px;
        min-width: 0;
        color: #f5fbff;
        font-size: 18px;
        font-weight: 1000;
        white-space: nowrap;
    }
    .bot-mark {
        width: 38px;
        height: 32px;
        position: relative;
        flex: 0 0 auto;
        border-radius: 15px 15px 12px 12px;
        background:
            radial-gradient(circle at 32% 48%, #243849 0 8%, transparent 9%),
            radial-gradient(circle at 68% 48%, #243849 0 8%, transparent 9%),
            linear-gradient(180deg, #f2f8ff, #a8bed3 58%, #748aa0);
        border: 1px solid rgba(255,255,255,.35);
        box-shadow: inset 0 -4px 8px rgba(0,0,0,.18), 0 4px 12px rgba(0,0,0,.25);
    }
    .bot-mark::before {
        content: "";
        position: absolute;
        left: 50%;
        top: -10px;
        width: 5px;
        height: 10px;
        border-radius: 999px;
        transform: translateX(-50%);
        background: linear-gradient(180deg, #ff9f43, #e66b19);
    }
    .bot-mark::after {
        content: "";
        position: absolute;
        left: 7px;
        right: 7px;
        bottom: 5px;
        height: 5px;
        border-radius: 999px;
        background: rgba(36,56,73,.28);
    }
    .main-nav {
        display: flex;
        align-items: center;
        gap: 22px;
        overflow-x: auto;
        scrollbar-width: none;
    }
    .main-nav::-webkit-scrollbar { display: none; }
    .main-nav a {
        color: #edf4ff;
        opacity: .92;
        text-decoration: none;
        font-size: 15px;
        font-weight: 900;
        line-height: 1;
        padding: 11px 10px;
        border-radius: 5px;
        white-space: nowrap;
    }
    .main-nav a:hover { background: rgba(255,255,255,.06); opacity: 1; }
    .main-nav a.active {
        color: #7fd0ff;
        background: rgba(97,111,132,.48);
        padding-inline: 20px;
    }

    .shell {
        min-height: 100dvh;
        height: auto;
        overflow: visible;
        padding: 14px 30px 16px;
    }
    .app-grid {
        display: grid;
        grid-template-columns: 380px minmax(760px, 1fr);
        gap: 18px;
        align-items: stretch;
        min-height: calc(100dvh - 30px);
        height: auto;
    }
    .panel {
        background: var(--panel);
        border: 1px solid rgba(255,255,255,.06);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        overflow: hidden;
        min-height: 0;
    }
    .sidebar {
        padding: 0;
        display: flex;
        flex-direction: column;
        gap: 18px;
        min-height: 0;
    }
    .app-grid > .sidebar {
        padding: 34px 46px 24px;
    }
    .sidebar-content {
        display: grid;
        gap: 16px;
    }
    h1, h2, h3, p { margin: 0; }
    h1.sidebar-title,
    .sidebar-title {
        font-size: 26px !important;
        line-height: 1.25 !important;
        letter-spacing: 0;
        font-weight: 1000 !important;
        color: var(--text) !important;
        margin: 0 !important;
    }
    .field-title {
        color: var(--text);
        font-size: 18px;
        font-weight: 900;
        margin-bottom: 10px;
    }
    .upload-box {
        border: 1px dashed rgba(255,255,255,.36);
        background: var(--card-2);
        border-radius: 8px;
        min-height: 188px;
        display: grid;
        place-items: center;
        color: var(--accent-dark);
        transition: border-color .16s ease, background .16s ease;
    }
    .upload-box:hover {
        border-color: rgba(46,168,229,.78);
        background: #101823;
    }
    .upload-icon {
        width: 58px;
        height: 48px;
        border: 5px solid rgba(20,121,184,.7);
        border-radius: 6px;
        position: relative;
        margin: 0 auto 18px;
    }
    .upload-icon::before {
        content: "";
        position: absolute;
        width: 14px;
        height: 14px;
        border-radius: 50%;
        background: rgba(20,121,184,.78);
        left: 10px;
        top: 10px;
    }
    .upload-icon::after {
        content: "";
        position: absolute;
        left: 9px;
        right: 7px;
        bottom: 8px;
        height: 24px;
        background:
            linear-gradient(135deg, transparent 0 28%, rgba(20,121,184,.8) 29% 52%, transparent 53%),
            linear-gradient(45deg, transparent 0 35%, rgba(20,121,184,.8) 36% 62%, transparent 63%);
    }
    .upload-text {
        text-align: center;
        color: #1687c6;
        font-size: 14px;
        font-weight: 800;
    }
    .select-row {
        display: grid;
        grid-template-columns: 86px minmax(0, 1fr);
        gap: 12px;
        align-items: center;
    }
    .select-label {
        color: #edf4ff;
        font-size: 16px;
        font-weight: 900;
        white-space: nowrap;
    }
    .view-select,
    .vehicle-select {
        width: 100%;
        height: 44px;
        border: 1px solid rgba(255,255,255,.42);
        background:
            linear-gradient(45deg, transparent 50%, rgba(255,255,255,.72) 50%) right 22px center / 9px 9px no-repeat,
            linear-gradient(135deg, rgba(255,255,255,.72) 50%, transparent 50%) right 14px center / 9px 9px no-repeat,
            var(--card-2);
        border-radius: 8px;
        color: #dce8f7;
        appearance: none;
        padding: 0 44px 0 16px;
        font-size: 16px;
        font-weight: 900;
        cursor: pointer;
    }
    .view-select:focus-visible,
    .vehicle-select:focus-visible {
        outline: 2px solid rgba(127,208,255,.8);
        outline-offset: 2px;
    }
    .topk-row {
        display: grid;
        grid-template-columns: 1fr 112px;
        gap: 12px;
        align-items: center;
    }
    .label {
        color: #c8d6e6;
        font-size: 18px;
        font-weight: 900;
    }
    .hint {
        color: var(--muted);
        font-size: 11px;
        font-weight: 700;
        margin-top: 4px;
        line-height: 1.5;
    }
    .sidebar .hint { display: none; }
    .number-like {
        height: 44px;
        border: 1px solid rgba(255,255,255,.42);
        background: var(--card-2);
        border-radius: 8px;
        color: #dce8f7;
        display: flex;
        align-items: center;
        justify-content: center;
        padding: 0;
        font-size: 18px;
        font-weight: 900;
    }
    .gallery-entry {
        width: 100%;
        min-width: 0;
        height: 54px;
        flex: 0 0 auto;
        border: 1px solid rgba(255,255,255,.14);
        border-radius: 8px;
        background: rgba(118,128,142,.34);
        color: #c8d0dc;
        color: #c8d0dc !important;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        padding: 0 20px;
        text-decoration: none;
        font-size: 18px;
        font-weight: 900;
    }
    .gallery-entry:visited,
    .gallery-entry:active,
    .gallery-entry *,
    .gallery-entry *:visited {
        color: #c8d0dc !important;
    }
    .gallery-entry:hover {
        border-color: rgba(255,255,255,.22);
        background: rgba(145,154,166,.42);
        color: #edf4ff;
        color: #edf4ff !important;
        text-decoration: none;
    }
    .gallery-entry:hover *,
    .gallery-entry:focus-visible {
        color: #edf4ff !important;
    }

    .result-panel {
        display: flex;
        flex-direction: column;
        min-height: 0;
    }
    .result-head {
        display: flex;
        align-items: center;
        justify-content: space-between;
        gap: 18px;
        margin: 24px 30px 14px 44px;
    }
    .result-head h2 {
        color: var(--text);
        font-size: 25px;
        line-height: 1.25;
        margin: 0;
        font-weight: 1000;
    }
    .result-stage {
        flex: 1;
        min-height: 0;
        margin: 0 30px 16px;
        background: var(--card-2);
        border: 1px solid rgba(255,255,255,.05);
        border-radius: 8px;
        padding: 18px 24px;
        display: grid;
        grid-template-columns: minmax(300px, 390px) auto;
        gap: 24px;
        align-items: center;
        justify-content: center;
    }
    .query-card,
    .candidate-card,
    .gallery-card,
    .score-card,
    .summary-points,
    .weight-card,
    .contour-card,
    .image-card,
    .part-card {
        background: rgba(21,28,37,.72);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 8px;
        overflow: hidden;
    }
    .query-card img,
    .candidate-card img,
    .gallery-card img {
        width: 100%;
        aspect-ratio: 4 / 3;
        object-fit: contain;
        background: #11181d;
        display: block;
    }
    .query-meta {
        border-top: 1px solid rgba(255,255,255,.08);
        padding: 11px 14px;
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: baseline;
    }
    .query-name { font-size: 15px; font-weight: 900; }
    .query-note { color: var(--muted); font-size: 12px; white-space: nowrap; }
    .candidate-grid {
        display: grid;
        grid-template-columns: repeat(2, 176px);
        gap: 10px 16px;
        justify-content: center;
    }
    .candidate-card {
        position: relative;
        display: block;
        text-decoration: none;
        color: inherit;
        transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }
    .candidate-card:hover {
        transform: translateY(-1px);
        border-color: rgba(46,168,229,.65);
        box-shadow: 0 18px 42px rgba(46,168,229,.12);
    }
    .candidate-card img {
        aspect-ratio: auto;
        height: 96px;
    }
    .badge {
        position: absolute;
        top: 5px;
        border-radius: 999px;
        background: rgba(0,0,0,.62);
        border: 1px solid rgba(255,255,255,.14);
        padding: 3px 5px;
        font-size: 9px;
        font-weight: 1000;
        backdrop-filter: blur(8px);
    }
    .rank { left: 5px; }
    .score { right: 5px; color: #bbf7d0; border-color: rgba(52,211,153,.5); }
    .score.mid { color: #bae6fd; border-color: rgba(46,168,229,.55); }
    .score.warn { color: #fde68a; border-color: rgba(251,191,36,.58); }
    .card-foot {
        min-height: 28px;
        padding: 6px 8px;
        border-top: 1px solid rgba(255,255,255,.08);
        display: flex;
        justify-content: space-between;
        gap: 10px;
        align-items: center;
    }
    .car-name {
        color: #edf4ff;
        font-size: 11px;
        font-weight: 900;
        overflow: hidden;
        white-space: nowrap;
        text-overflow: ellipsis;
    }
    .detail-link {
        color: var(--accent);
        font-size: 10px;
        font-weight: 900;
        white-space: nowrap;
    }
    .empty-result {
        min-height: 320px;
        display: grid;
        place-items: center;
        color: #8ab7d8;
        font-size: 18px;
        font-weight: 900;
        text-align: center;
    }
    .result-blank {
        min-height: 100%;
        width: 100%;
    }
    .result-loading {
        grid-column: 1 / -1;
        min-height: 320px;
        width: 100%;
        display: grid;
        place-items: center;
        text-align: center;
    }
    .result-loading-card {
        display: grid;
        justify-items: center;
        gap: 14px;
        color: #edf4ff;
        font-size: 18px;
        font-weight: 900;
    }
    .loading-ring {
        width: 48px;
        height: 48px;
        border-radius: 50%;
        border: 4px solid rgba(46,168,229,.18);
        border-top-color: #2ea8e5;
        animation: spin .9s linear infinite;
    }
    .loading-note {
        color: #9aa8b8;
        font-size: 13px;
        font-weight: 800;
    }
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    @media (prefers-reduced-motion: reduce) {
        .loading-ring { animation: none; }
    }

    .detail-stage {
        flex: 1;
        min-height: 0;
        margin: 0 30px 16px;
        background: var(--card-2);
        border: 1px solid rgba(255,255,255,.05);
        border-radius: 8px;
        overflow: auto;
    }
    .detail-canvas {
        min-width: 980px;
        padding: 0 8px 8px;
    }
    .topbar {
        position: sticky;
        top: 0;
        z-index: 10;
        background: rgba(5,8,12,.96);
        padding: 0 0 12px;
        display: grid;
        grid-template-columns: 1fr auto;
        gap: 18px;
        align-items: center;
    }
    .topbar h1 {
        color: var(--text);
        font-size: 26px;
        line-height: 1.25;
        font-weight: 1000;
    }
    .back {
        min-height: 44px;
        display: inline-flex;
        align-items: center;
        gap: 10px;
        border: 1px solid rgba(255,255,255,.12);
        background: var(--card);
        color: #c8d6e6;
        border-radius: 8px;
        padding: 0 18px;
        text-decoration: none;
        font-weight: 900;
    }
    .back:hover { border-color: rgba(46,168,229,.55); color: #fff; }
    .anchor-nav {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin: 0 0 14px;
    }
    .anchor-nav a {
        min-height: 36px;
        display: inline-flex;
        align-items: center;
        padding: 0 12px;
        border-radius: 999px;
        border: 1px solid rgba(255,255,255,.10);
        background: rgba(31,37,46,.76);
        color: #c8d6e6;
        text-decoration: none;
        font-size: 13px;
        font-weight: 900;
    }
    .summary {
        display: grid;
        grid-template-columns: 280px minmax(420px, 1fr) 300px;
        gap: 20px;
        padding: 20px;
        margin-bottom: 24px;
        background: var(--panel);
        border-radius: 12px;
    }
    .score-card,
    .summary-points,
    .weight-card,
    .contour-card,
    .image-card,
    .part-card,
    .part-score-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,.09);
        border-radius: 10px;
    }
    .score-card, .part-score-card {
        padding: 22px 24px;
        display: grid;
        align-content: center;
    }
    .detail-label {
        color: var(--muted);
        font-size: 13px;
        font-weight: 900;
        margin-bottom: 8px;
    }
    .big-score {
        font-size: 42px;
        font-weight: 1000;
        line-height: 1;
    }
    .tag {
        display: inline-flex;
        width: fit-content;
        align-items: center;
        border: 1px solid rgba(46,168,229,.48);
        color: #bae6fd;
        border-radius: 999px;
        padding: 6px 10px;
        margin-top: 14px;
        font-size: 13px;
        font-weight: 900;
    }
    .tag.good { border-color: rgba(52,211,153,.5); color: #bbf7d0; }
    .tag.warn { border-color: rgba(251,191,36,.58); color: #fde68a; }
    .summary-points { padding: 18px; }
    .summary-title {
        font-size: 20px;
        font-weight: 1000;
        margin-bottom: 14px;
    }
    .point-list {
        display: grid;
        gap: 10px;
        margin: 0;
        padding: 0;
    }
    .point-list li {
        list-style: none;
        background: rgba(12,17,23,.62);
        border: 1px solid rgba(255,255,255,.07);
        border-radius: 8px;
        color: #dce8f7;
        font-size: 15px;
        line-height: 1.55;
        padding: 10px 12px;
    }
    .weight-card {
        padding: 18px;
        display: grid;
        gap: 14px;
    }
    .weight-row { display: grid; gap: 8px; }
    .weight-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: baseline;
        font-weight: 900;
    }
    .weight-name { color: #c8d6e6; font-size: 13px; }
    .weight-score { font-size: 20px; }
    .bar {
        height: 7px;
        background: rgba(255,255,255,.08);
        border-radius: 999px;
        overflow: hidden;
    }
    .bar span {
        display: block;
        height: 100%;
        width: var(--value);
        background: linear-gradient(90deg, var(--accent), var(--good));
        border-radius: inherit;
    }
    .section {
        padding: 18px;
        margin-bottom: 24px;
        background: var(--panel);
        border-radius: 12px;
    }
    .section-head {
        display: flex;
        justify-content: space-between;
        gap: 16px;
        align-items: baseline;
        margin-bottom: 16px;
    }
    .section-head h2 {
        color: var(--text);
        font-size: 24px;
        line-height: 1.25;
        font-weight: 1000;
    }
    .section-note {
        color: var(--muted);
        font-size: 13px;
        font-weight: 800;
    }
    .contour-layout {
        display: grid;
        grid-template-columns: minmax(240px, 300px) minmax(360px, 1fr);
        gap: 26px;
        align-items: start;
    }
    .contour-card { padding: 14px; min-height: 330px; }
    .contour-score {
        font-size: 32px;
        font-weight: 1000;
        line-height: 1;
        margin-bottom: 10px;
    }
    .legend { display: grid; gap: 8px; margin-top: 14px; }
    .legend-item {
        display: flex;
        align-items: center;
        gap: 9px;
        color: #c8d6e6;
        font-size: 12px;
        font-weight: 900;
    }
    .swatch { width: 22px; height: 12px; border-radius: 4px; display: inline-block; }
    .yellow { background: #fff200; }
    .red { background: #f20f16; }
    .green { background: #15ef37; }
    .contour-copy {
        color: #c8d6e6;
        font-size: 12px;
        line-height: 1.55;
        margin-top: 14px;
    }
    .image-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 8px; justify-items: start; }
    .image-card.large { grid-column: 1 / -1; width: min(100%, 440px); justify-self: start; }
    .image-card img,
    .part-tile img {
        display: block;
        width: 100%;
        background: #11181d;
        object-fit: contain;
    }
    .image-card img { aspect-ratio: 16 / 9; }
    .image-caption,
    .part-caption {
        border-top: 1px solid rgba(255,255,255,.08);
        color: #c8d6e6;
        font-size: 13px;
        font-weight: 900;
        padding: 9px 11px;
    }
    .diff-map { background: #050505; display: block; width: fit-content; max-width: 100%; }
    .diff-map img { width: 100%; max-width: 440px; height: auto; display: block; object-fit: contain; aspect-ratio: auto; }
    .contour-footnote {
        width: min(100%, 440px);
        color: var(--muted);
        font-size: 13px;
        font-weight: 800;
        text-align: center;
    }
    .parts-summary {
        display: grid;
        grid-template-columns: 280px 1fr;
        gap: 18px;
        margin-bottom: 18px;
        align-items: start;
    }
    .parts-image-pair {
        display: grid;
        grid-template-columns: repeat(2, minmax(280px, 1fr));
        gap: 14px;
    }
    .part-grid { display: grid; grid-template-columns: minmax(0, 1fr); gap: 20px; }
    .part-card { padding: 16px; }
    .part-head {
        display: flex;
        justify-content: space-between;
        gap: 12px;
        align-items: center;
        margin-bottom: 12px;
    }
    .part-title {
        color: var(--text);
        font-size: 22px;
        font-weight: 1000;
        line-height: 1.25;
    }
    .part-badge {
        border: 1px solid rgba(52,211,153,.5);
        color: #bbf7d0;
        border-radius: 999px;
        padding: 6px 10px;
        font-size: 13px;
        font-weight: 1000;
        white-space: nowrap;
    }
    .part-badge.mid { border-color: rgba(46,168,229,.55); color: #bae6fd; }
    .part-badge.warn { border-color: rgba(251,191,36,.58); color: #fde68a; }
    .tiles {
        display: grid;
        grid-template-columns: repeat(5, minmax(120px, 1fr));
        column-gap: 16px;
        row-gap: 10px;
        background: #05080c;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .part-tile {
        background: var(--card-2);
        border: 1px solid rgba(255,255,255,.08);
        border-radius: 8px;
        overflow: hidden;
    }
    .part-tile img, .part-tile .tile-diff { aspect-ratio: 1 / 1; }
    .part-caption { padding: 7px 8px; color: #8ab7d8; font-size: 12px; }
    .tile-diff {
        background:
            radial-gradient(circle at 66% 30%, rgba(242,15,22,.95) 0 12%, transparent 13%),
            linear-gradient(120deg, transparent 0 21%, rgba(21,239,55,.96) 22% 35%, transparent 36%),
            linear-gradient(168deg, transparent 0 64%, rgba(242,15,22,.92) 65% 74%, transparent 75%),
            #10e739;
        border-bottom: 1px solid rgba(255,255,255,.08);
    }

    .page { padding: 24px 30px 34px; }
    .gallery-shell {
        max-width: 1560px;
        margin: 0 auto;
        background: var(--panel);
        border: 1px solid rgba(255,255,255,.07);
        border-radius: var(--radius);
        box-shadow: var(--shadow);
        overflow: hidden;
    }
    .gallery-head {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        gap: 18px;
        align-items: center;
        padding: 26px 32px 22px;
        border-bottom: 1px solid rgba(255,255,255,.08);
    }
    .gallery-head h1 {
        color: var(--text);
        font-size: 28px;
        line-height: 1.25;
        font-weight: 1000;
    }
    .sub {
        color: var(--muted);
        font-size: 14px;
        font-weight: 800;
        margin-top: 8px;
    }
    .gallery-grid {
        display: grid;
        grid-template-columns: repeat(auto-fill, minmax(220px, 1fr));
        gap: 16px;
        padding: 26px 32px 32px;
    }
    .gallery-card {
        margin: 0;
        background: rgba(12,17,23,.74);
        transition: transform .16s ease, border-color .16s ease, box-shadow .16s ease;
    }
    .gallery-card:hover {
        transform: translateY(-1px);
        border-color: rgba(46,168,229,.55);
        box-shadow: 0 18px 42px rgba(46,168,229,.10);
    }
    .gallery-card figcaption {
        border-top: 1px solid rgba(255,255,255,.08);
        padding: 10px 12px;
        color: #dce8f7;
        font-size: 14px;
        font-weight: 900;
    }

    .stSelectbox label, .stFileUploader label, .stNumberInput label {
        color: #edf4ff !important;
        font-size: 22px !important;
        font-weight: 900 !important;
    }
    div[data-baseweb="select"] > div,
    div[data-testid="stNumberInputContainer"] {
        min-height: 44px !important;
        background: var(--card-2) !important;
        border: 1px solid rgba(255,255,255,.42) !important;
        border-radius: 8px !important;
        color: #dce8f7 !important;
        box-shadow: none !important;
    }
    div[data-baseweb="select"] *,
    div[data-testid="stNumberInputContainer"] input {
        color: #dce8f7 !important;
        -webkit-text-fill-color: #dce8f7 !important;
        font-weight: 900 !important;
    }
    section[data-testid="stFileUploaderDropzone"] {
        min-height: 188px !important;
        border: 1px dashed rgba(255,255,255,.36) !important;
        background: var(--card-2) !important;
        border-radius: 8px !important;
        display: grid !important;
        place-items: center !important;
        color: var(--accent-dark) !important;
    }
    section[data-testid="stFileUploaderDropzone"]:hover {
        border-color: rgba(46,168,229,.78) !important;
        background: #101823 !important;
    }
    section[data-testid="stFileUploaderDropzone"] * {
        color: #1687c6 !important;
        font-size: 16px !important;
        font-weight: 800 !important;
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] {
        width: 72px !important;
        height: 64px !important;
        padding: 0 !important;
        border: 5px solid rgba(20, 121, 184, .85) !important;
        border-radius: 7px !important;
        background: transparent !important;
        box-shadow: none !important;
        color: transparent !important;
        position: relative !important;
        margin-bottom: 14px !important;
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::before {
        content: "";
        position: absolute;
        left: 15px;
        top: 15px;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: rgba(20, 121, 184, .95);
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"]::after {
        content: "";
        position: absolute;
        left: 13px;
        right: 11px;
        bottom: 10px;
        height: 30px;
        background:
            linear-gradient(135deg, transparent 0 28%, rgba(20,121,184,.95) 29% 52%, transparent 53%),
            linear-gradient(45deg, transparent 0 35%, rgba(20,121,184,.95) 36% 62%, transparent 63%);
    }
    section[data-testid="stFileUploaderDropzone"] button[data-testid="stBaseButton-secondary"] * {
        display: none !important;
    }
    div[data-testid="stFileUploaderFile"],
    [data-testid^="stFileUploaderFile"] {
        display: none !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] {
        color: transparent !important;
        display: block !important;
        font-size: 0 !important;
        font-weight: 900 !important;
        text-align: center !important;
        width: 100% !important;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"]::before {
        content: "上传";
        display: block;
        color: #1687c6;
        font-size: 18px;
        font-weight: 900;
        text-align: center;
        margin-bottom: 4px;
        white-space: nowrap;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"]::after {
        content: "JPG / PNG / WEBP / BMP";
        display: block;
        color: rgba(22,135,198,.82);
        font-size: 11px;
        font-weight: 800;
        text-align: center;
        white-space: nowrap;
    }
    div[data-testid="stFileUploaderDropzoneInstructions"] *,
    div[data-testid="stFileUploaderDropzoneInstructions"] > span,
    div[data-testid="stFileUploaderDropzoneInstructions"] > small {
        display: none !important;
    }
    button[data-testid="stBaseButton-primary"],
    button[kind="primary"] {
        width: 100% !important;
        height: 52px !important;
        border: 0 !important;
        border-radius: 8px !important;
        background: linear-gradient(180deg, #22b9eb, #12aada) !important;
        color: #f6fbff !important;
        font-size: 22px !important;
        font-weight: 900 !important;
        box-shadow: 0 10px 28px rgba(46,168,229,.20) !important;
    }
    button[data-testid="stBaseButton-primary"]:hover,
    button[kind="primary"]:hover { filter: brightness(1.08); color: #fff !important; }
    button[disabled] { opacity: .42 !important; }
    .upload-preview-card {
        margin-top: -6px;
        background: rgba(12,17,23,.72);
        border: 1px solid rgba(255,255,255,.10);
        border-radius: 8px;
        overflow: hidden;
    }
    .upload-preview-card img {
        width: 100%;
        aspect-ratio: 4 / 3;
        display: block;
        object-fit: contain;
        background: #11181d;
    }
    .upload-preview-name {
        border-top: 1px solid rgba(255,255,255,.08);
        color: #dce8f7;
        font-family: "Microsoft YaHei UI", "PingFang SC", "Source Han Sans SC", system-ui, sans-serif;
        font-size: 13px;
        font-weight: 900;
        line-height: 1.45;
        padding: 9px 11px;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    @media (max-width: 1280px) {
        .app-grid { grid-template-columns: 330px 1fr; }
        .app-grid > .sidebar { padding: 28px; }
        .app-grid > .sidebar .sidebar-title { font-size: 24px !important; }
        .summary, .parts-summary { grid-template-columns: 1fr; }
        .detail-canvas { min-width: 0; }
    }
    @media (max-width: 900px) {
        .app-header {
            grid-template-columns: 1fr;
            height: auto;
            gap: 10px;
            padding: 12px 16px;
        }
        .main-nav { gap: 10px; padding-bottom: 2px; }
        .main-nav a { font-size: 14px; }
        .shell { min-height: 100dvh; overflow: visible; padding: 12px; }
        div[data-testid="stHorizontalBlock"] {
            padding: 12px;
            min-height: auto;
            flex-direction: column !important;
            flex-wrap: nowrap !important;
        }
        .app-grid { grid-template-columns: 1fr; }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:first-child {
            flex: 1 1 auto !important;
            width: 100% !important;
            max-width: none !important;
        }
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"]:last-child {
            flex: 1 1 auto !important;
            width: 100% !important;
            max-width: none !important;
        }
        .sidebar { padding: 28px; }
        .result-head { margin: 28px 18px 18px; }
        .result-stage {
            margin: 0 18px 18px;
            grid-template-columns: 1fr;
            min-height: auto;
        }
        .candidate-grid { grid-template-columns: 1fr; }
        .detail-stage { margin: 0 18px 18px; overflow: visible; }
        .detail-canvas { min-width: 0; padding: 0; }
        .summary, .section { padding: 14px; }
        .contour-layout, .parts-image-pair, .tiles { grid-template-columns: 1fr; }
        .topbar { grid-template-columns: 1fr; position: static; }
        .page { padding: 12px; }
        .gallery-head { grid-template-columns: 1fr; padding: 22px 18px 18px; }
        .gallery-grid { grid-template-columns: 1fr; padding: 18px; }
    }
</style>
"""


def apply_prototype_chrome() -> None:
    st.markdown(PROTOTYPE_CSS, unsafe_allow_html=True)


def _header_html() -> str:
    return _html_block(
        """
        <div class="app-header" role="banner" aria-label="产品导航">
          <div class="brand">
            <span class="bot-mark" aria-hidden="true"></span>
            <span>Ragentic Designer</span>
          </div>
          <div class="main-nav" role="navigation" aria-label="主导航">
            <a href="#">首页</a>
            <a href="#">创意设计</a>
            <a href="#">图像渲染</a>
            <a href="#">图像编辑</a>
            <a href="#">图生视频</a>
            <a href="#">3D生成</a>
            <a href="#">资产管理</a>
            <a href="#">历史记录</a>
            <a class="active" href="?view=workbench" target="_self">对比查重</a>
          </div>
        </div>
        """
    )


def _path_uri(path: Path | str | None, size: tuple[int, int] = (640, 480)) -> str | None:
    if not path:
        return None
    p = Path(str(path))
    if not p.is_file():
        return None
    try:
        return _thumb_uri_for_path(str(p), float(p.stat().st_mtime), size=size)
    except Exception:
        return None


def _score_badge(score: object) -> tuple[str, str]:
    try:
        value = float(score)
    except (TypeError, ValueError):
        return "score warn", "局部相似"
    if value >= 85:
        return "score", "高相似"
    if value >= 70:
        return "score mid", "中高相似"
    return "score warn", "局部相似"


def _sample_rows(gallery_dir: Path, topk: int = 10) -> list[dict]:
    paths = _gallery_image_paths(gallery_dir, limit=topk)
    scores = [90, 78.8, 76.6, 75.2, 73.4, 70.8, 69.4, 68.7, 67.2, 65.9]
    return [
        {
            "candidate_id": path.stem,
            "candidate_path": str(path),
            "final_score": scores[idx] if idx < len(scores) else max(60, 90 - idx * 2),
        }
        for idx, path in enumerate(paths)
    ]


def _candidate_cards(rows: list[dict]) -> str:
    cards: list[str] = []
    for idx, row in enumerate(rows[:10], start=1):
        candidate_id = str(row.get("candidate_id", ""))
        score = row.get("final_score")
        klass, label = _score_badge(score)
        uri = _path_uri(row.get("candidate_path"), size=(360, 260))
        img = f'<img src="{uri}" alt="候选车辆：{escape(candidate_id)}" />' if uri else '<div class="empty-result">暂无图片</div>'
        href = f"?view=detail&cid={quote(candidate_id)}"
        cards.append(
            _html_block(
                f"""
                <a class="candidate-card" href="{href}" target="_self">
                  <span class="badge rank">Top {idx}</span>
                  <span class="badge {klass}">{escape(_fmt_score(score))} · {label}</span>
                  {img}
                  <div class="card-foot"><span class="car-name">{escape(candidate_id)}</span><span class="detail-link">查看详情</span></div>
                </a>
                """
            )
        )
    return "\n".join(cards)


def render_workbench_shell(
    *,
    gallery_dir: Path,
    gallery_count: int,
    result: dict | None,
    uploaded_preview_uri: str | None,
) -> tuple[object, int, bool]:
    rows = (result or {}).get("results") or []
    query_path = (result or {}).get("query")
    query_uri = uploaded_preview_uri or _path_uri(query_path, size=(640, 480))

    st.markdown(_header_html(), unsafe_allow_html=True)
    left_col, right_col = st.columns([0.28, 0.72], gap="large")
    with left_col:
        st.markdown(
            _html_block(
                """
                <div class="sidebar-content">
                  <h1 class="sidebar-title">相似度查重</h1>
                  <div class="select-row">
                    <label class="select-label" for="viewSelect">视角选择</label>
                    <select class="view-select" id="viewSelect" aria-label="视角选择">
                      <option selected>正脸视图</option>
                      <option>侧面车身视图</option>
                      <option>尾部视图</option>
                    </select>
                  </div>
                  <div class="select-row">
                    <label class="select-label" for="vehicleSelect">车型选择</label>
                    <select class="vehicle-select" id="vehicleSelect" aria-label="车型选择">
                      <option selected>SUV</option>
                      <option>轿车</option>
                      <option>轿跑</option>
                      <option>越野</option>
                      <option>MPV</option>
                      <option>皮卡</option>
                    </select>
                  </div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        st.markdown('<div class="field-title">上传待对比图片</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "上传待对比图片",
            type=["jpg", "jpeg", "png", "webp", "bmp"],
            key="prototype_upload",
            label_visibility="collapsed",
        )
        if uploaded is not None and uploaded_preview_uri:
            st.markdown(
                _html_block(
                    f"""
                    <figure class="upload-preview-card">
                      <img src="{uploaded_preview_uri}" alt="已上传图片缩略图" />
                      <figcaption class="upload-preview-name" title="{escape(uploaded.name)}">{escape(uploaded.name)}</figcaption>
                    </figure>
                    """
                ),
                unsafe_allow_html=True,
            )
        st.markdown(
            _html_block(
                """
                <div class="topk-row">
                  <div>
                    <div class="label">相似度Top-K</div>
                    <div class="hint">返回最相似的前 N 张候选图。</div>
                  </div>
                  <div class="number-like">10</div>
                </div>
                """
            ),
            unsafe_allow_html=True,
        )
        start = st.button("开始比对", type="primary", use_container_width=True, disabled=uploaded is None or gallery_count <= 0)
        st.markdown('<a class="gallery-entry" href="?view=gallery" target="_self">图库管理</a>', unsafe_allow_html=True)

    if start and uploaded is not None:
        result_body = _html_block(
            """
            <div class="result-loading" role="status" aria-live="polite">
              <div class="result-loading-card">
                <div class="loading-ring" aria-hidden="true"></div>
                <div>正在执行相似度计算，请稍候...</div>
                <div class="loading-note">系统正在提取轮廓与部件特征</div>
              </div>
            </div>
            """
        )
    elif result:
        query_img = f'<img src="{query_uri}" alt="上传待比对车辆" />' if query_uri else '<div class="empty-result">待比对图片信息</div>'
        result_body = _html_block(
            f"""
            <article class="query-card">
              {query_img}
              <div class="query-meta">
                <div class="query-name">上传比对图片</div>
                <div class="query-note">当前任务结果</div>
              </div>
            </article>
            <div class="candidate-grid" aria-label="Top-K 相似候选">
              {_candidate_cards(rows)}
            </div>
            """
        )
    else:
        result_body = '<div class="result-blank" aria-label="等待分析结果"></div>'

    with right_col:
        st.markdown(
            _html_block(
                f"""
                <section class="panel result-panel" style="min-height: calc(100dvh - 106px);">
                  <div class="result-head"><h2>分析结果</h2></div>
                  <div class="result-stage">
                    {result_body}
                  </div>
                </section>
                """
            ),
            unsafe_allow_html=True,
        )
    return uploaded, 10, start


def render_gallery_page(gallery_dir: Path, gallery_count: int) -> None:
    paths = _gallery_image_paths(gallery_dir, limit=500)
    cards = []
    for path in paths:
        uri = _path_uri(path, size=(520, 390))
        if not uri:
            continue
        name = path.stem
        cards.append(
            f'<figure class="gallery-card"><img src="{uri}" alt="{escape(name)}" /><figcaption>{escape(name)}</figcaption></figure>'
        )
    st.markdown(
        _html_block(
            f"""
            {_header_html()}
            <main class="page">
              <section class="gallery-shell" aria-label="图库管理">
                <div class="gallery-head">
                  <div>
                    <h1>图库管理</h1>
                    <p class="sub">{gallery_count} 张可检索图片 · 正面车辆图库</p>
                  </div>
                  <a class="back" href="?view=workbench" target="_self">返回分析工作台</a>
                </div>
                <div class="gallery-grid">{''.join(cards)}</div>
              </section>
            </main>
            """
        ),
        unsafe_allow_html=True,
    )


def render_detail_page(row: dict | None, result: dict | None, gallery_dir: Path) -> None:
    row = row or (_sample_rows(gallery_dir, 1)[0] if _sample_rows(gallery_dir, 1) else {})
    outputs = (result or {}).get("outputs") or {}
    label_dir = Path(outputs.get("front_label", ""))
    query_label = _find_label(label_dir, (result or {}).get("query_id", "")) if result else None
    candidate_id = str(row.get("candidate_id", "候选车辆"))
    final_score = row.get("final_score", 78.8)
    contour_score = row.get("contour_score", 90.6)
    part_score = row.get("part_score", 78.8)
    contour_uri = _path_uri(row.get("contour_diff_image"), size=(760, 520)) or _path_uri(
        Path("ui_prototypes/assets/edge_diff.jpg"), size=(760, 520)
    )
    query_label_uri = _path_uri(query_label, size=(520, 390)) or _path_uri(Path("ui_prototypes/assets/A_annotation.jpg"), size=(520, 390))
    cand_label = _find_label(label_dir, candidate_id) if result else None
    cand_label_uri = _path_uri(cand_label, size=(520, 390)) or _path_uri(Path("ui_prototypes/assets/B_annotation.jpg"), size=(520, 390))

    points = row.get("analysis") or [
        f"最终评分 {_fmt_score(final_score)} 分，判定为中高相似，适合进入人工复核。",
        f"整车轮廓相似度 {_fmt_score(contour_score)}，车身基础比例和正面姿态高度接近。",
        "后视镜与前灯区域相似度较高，前挡风玻璃和局部下沿存在更明显差异。",
    ]
    point_items = "".join(f"<li>{escape(str(point))}</li>" for point in points[:4])
    part_cards = _part_cards_html(row)

    st.markdown(_header_html(), unsafe_allow_html=True)
    sidebar_html = _html_block(
        """
            <aside class="panel sidebar">
              <h1 class="sidebar-title">汽车图片相似度分析</h1>
              <div>
                <div class="field-title">上传待对比图片</div>
                <div class="upload-box" role="button" tabindex="0" aria-label="上传待对比图片">
                  <div>
                    <div class="upload-icon"></div>
                    <div class="upload-text">上传</div>
                  </div>
                </div>
              </div>
              <div class="number-like">车型选择</div>
              <div class="topk-row">
                <div><div class="label">相似度Top-K</div><div class="hint">返回最相似的前 N 张候选图。</div></div>
                <div class="number-like">10</div>
              </div>
              <a class="gallery-entry" href="?view=workbench" target="_self">返回工作台</a>
            </aside>
        """
    )
    st.markdown(
        _html_block(
            f"""
            <main class="shell">
            <section class="app-grid" aria-label="汽车图片相似度分析工作台">
            {sidebar_html}
            <section class="panel result-panel">
              <div class="result-head"><h2>分析结果</h2></div>
              <div class="detail-stage">
                <div class="detail-canvas">
                  <header class="topbar">
                    <h1>A车 vs {escape(candidate_id)} 相似度分析详情</h1>
                    <a class="back" href="?view=workbench" target="_self" aria-label="返回概览">返回概览</a>
                  </header>
                  <nav class="anchor-nav" aria-label="详情页快捷导航">
                    <a href="#overview">概览</a>
                    <a href="#contour">轮廓</a>
                    <a href="#evidence">部件详情</a>
                  </nav>
                  <section id="overview" class="summary" aria-label="相似度总结">
                    <article class="score-card">
                      <div class="detail-label">总相似度</div>
                      <div class="big-score">{escape(_fmt_score(final_score))}</div>
                      <div class="tag">中高相似</div>
                    </article>
                    <article class="summary-points">
                      <div class="summary-title">相似度分析</div>
                      <ul class="point-list">{point_items}</ul>
                    </article>
                    <article class="weight-card">
                      <div class="weight-row">
                        <div class="weight-head"><span class="weight-name">轮廓分 · 权重 40%</span><span class="weight-score">{escape(_fmt_score(contour_score))}</span></div>
                        <div class="bar" style="--value:{escape(_fmt_score(contour_score))}%"><span></span></div>
                      </div>
                      <div class="weight-row">
                        <div class="weight-head"><span class="weight-name">部件分 · 权重 60%</span><span class="weight-score">{escape(_fmt_score(part_score))}</span></div>
                        <div class="bar" style="--value:{escape(_fmt_score(part_score))}%"><span></span></div>
                      </div>
                    </article>
                  </section>
                  <section id="contour" class="section" aria-label="整体轮廓相似对比">
                    <div class="section-head"><h2>整体轮廓相似对比</h2></div>
                    <div class="contour-layout">
                      <aside class="contour-card">
                        <div class="detail-label">轮廓分（40%）</div>
                        <div class="contour-score">{escape(_fmt_score(contour_score))}</div>
                        <div class="tag good">高相似</div>
                        <div class="legend" aria-label="轮廓差异图图例">
                          <div class="legend-item"><span class="swatch yellow"></span>黄色：两车高度重合区域</div>
                          <div class="legend-item"><span class="swatch red"></span>红色：A 图独有形状区域</div>
                          <div class="legend-item"><span class="swatch green"></span>绿色：B 图独有形状区域</div>
                        </div>
                        <p class="contour-copy">结论：两车正面主体轮廓接近，差异主要集中在车头下沿、车顶线和局部外扩区域。</p>
                      </aside>
                      <div class="image-grid">
                        <figure class="image-card large">
                          <div class="diff-map" role="img" aria-label="整车轮廓差异图"><img src="{contour_uri or ''}" alt="整车轮廓差异图" /></div>
                          <figcaption class="image-caption">整车轮廓差异图</figcaption>
                        </figure>
                        <div class="contour-footnote">红色和绿色只用于表达差异，不作为普通高亮色使用</div>
                      </div>
                    </div>
                  </section>
                  <section id="parts" class="section" aria-label="部件识别与对齐标注">
                    <div class="section-head">
                      <h2>部件识别与对齐标注</h2>
                      <div class="section-note">先给部件总分，再给每个部件的可追溯证据</div>
                    </div>
                    <div class="parts-summary">
                      <article class="part-score-card">
                        <div class="detail-label">部件分（60%）</div>
                        <div class="big-score">{escape(_fmt_score(part_score))}</div>
                        <div class="tag">中高相似</div>
                      </article>
                      <div class="parts-image-pair">
                        <figure class="image-card"><img src="{query_label_uri or ''}" alt="A 图部件识别标注" /><figcaption class="image-caption">A 图部件识别</figcaption></figure>
                        <figure class="image-card"><img src="{cand_label_uri or ''}" alt="B 图部件识别标注" /><figcaption class="image-caption">B 图部件识别</figcaption></figure>
                      </div>
                    </div>
                    <div id="evidence" class="part-grid">{part_cards}</div>
                  </section>
                </div>
              </div>
            </section>
            </section></main>
            """
        ),
        unsafe_allow_html=True,
    )


def _part_cards_html(row: dict) -> str:
    part_scores = row.get("part_scores") or {}
    if not part_scores:
        return _sample_part_cards_html(row)
    cards: list[str] = []
    for part, detail in part_scores.items():
        label = PART_LABELS.get(part, part)
        fused = detail.get("fused")
        badge_class = "part-badge" if _safe_float(fused) >= 85 else "part-badge mid" if _safe_float(fused) >= 70 else "part-badge warn"
        badge_text = "高相似" if _safe_float(fused) >= 85 else "中高相似" if _safe_float(fused) >= 70 else "差异明显"
        try:
            color_a, color_b, gray_a, gray_b, diff = _part_visuals(detail["query_path"], detail["candidate_path"], size=180)
            tiles = [
                ("A color", _img_to_data_uri(color_a, max_size=(180, 180), fmt="JPEG")),
                ("B color", _img_to_data_uri(color_b, max_size=(180, 180), fmt="JPEG")),
                ("A gray", _img_to_data_uri(gray_a, max_size=(180, 180), fmt="JPEG")),
                ("B gray", _img_to_data_uri(gray_b, max_size=(180, 180), fmt="JPEG")),
                ("diff", _img_to_data_uri(diff, max_size=(180, 180), fmt="JPEG")),
            ]
            tile_html = "".join(
                f'<div class="part-tile"><img src="{uri}" alt="{escape(cap)}" /><div class="part-caption">{escape(cap)}</div></div>'
                for cap, uri in tiles
            )
        except Exception:
            tile_html = _fallback_tiles_html(row)
        cards.append(
            _html_block(
                f"""
                <article class="part-card">
                  <div class="part-head">
                    <div class="part-title">{escape(label)}相似度评分：{escape(_fmt_score(fused))}</div>
                    <div class="{badge_class}">{badge_text}</div>
                  </div>
                  <div class="tiles">{tile_html}</div>
                </article>
                """
            )
        )
    return "".join(cards)


def _sample_part_cards_html(row: dict) -> str:
    samples = [
        ("后视镜", 96.8, "part-badge"),
        ("车灯", 86.4, "part-badge"),
        ("前保险杠", 81.6, "part-badge mid"),
        ("前挡风玻璃", 64.6, "part-badge warn"),
    ]
    return "".join(
        _html_block(
            f"""
            <article class="part-card">
              <div class="part-head">
                <div class="part-title">{name}相似度评分：{score}</div>
                <div class="{klass}">{'差异明显' if 'warn' in klass else '中高相似' if 'mid' in klass else '高相似'}</div>
              </div>
              <div class="tiles">{_fallback_tiles_html(row)}</div>
            </article>
            """
        )
        for name, score, klass in samples
    )


def _fallback_tiles_html(row: dict) -> str:
    candidate_uri = _path_uri(row.get("candidate_path"), size=(260, 260))
    sample_uri = _path_uri(DEFAULT_GALLERY_DIR / "星耀7.png", size=(260, 260)) or candidate_uri
    tiles = [
        ("A color", sample_uri),
        ("B color", candidate_uri or sample_uri),
        ("A gray", sample_uri),
        ("B gray", candidate_uri or sample_uri),
    ]
    html = "".join(
        f'<div class="part-tile"><img src="{uri or ""}" alt="{escape(cap)}" /><div class="part-caption">{escape(cap)}</div></div>'
        for cap, uri in tiles
    )
    return html + '<div class="part-tile"><div class="tile-diff"></div><div class="part-caption">diff</div></div>'


def _safe_float(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
