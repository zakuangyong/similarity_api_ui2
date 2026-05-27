from __future__ import annotations

from pathlib import Path

import streamlit as st
from PIL import Image

from similarity_pipeline import run_pipeline

from .config import (
    DEFAULT_FEATURES,
    DEFAULT_GALLERY_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_PARTS,
    DEFAULT_UPLOAD_ROOT,
)
from .prototype import (
    apply_prototype_chrome,
    render_detail_page,
    render_gallery_page,
    render_workbench_shell,
)
from .utils import (
    _clear_query_params,
    _count_gallery,
    _get_query_params,
    _img_to_data_uri,
    _save_upload,
)


def run_app() -> None:
    st.set_page_config(page_title="汽车图片相似度分析", layout="wide", initial_sidebar_state="collapsed")
    apply_prototype_chrome()

    gallery_dir = DEFAULT_GALLERY_DIR
    upload_root = DEFAULT_UPLOAD_ROOT
    output_dir = DEFAULT_OUTPUT_DIR
    features = DEFAULT_FEATURES
    selected_parts = DEFAULT_PARTS
    gallery_count = _count_gallery(gallery_dir)

    qp = _get_query_params()
    view = qp.get("view", "workbench")
    result = st.session_state.get("last_result")

    if view == "gallery":
        render_gallery_page(gallery_dir, gallery_count)
        return

    rows = (result or {}).get("results") or []
    rows_by_id = {str(row.get("candidate_id")): row for row in rows}
    selected_cid = qp.get("cid") or st.session_state.get("selected_candidate_id")
    if view == "detail" or selected_cid:
        render_detail_page(rows_by_id.get(str(selected_cid)), result, gallery_dir)
        return

    uploaded_preview_uri = _uploaded_preview_uri(st.session_state.get("prototype_upload"))
    uploaded, topk, start = render_workbench_shell(
        gallery_dir=gallery_dir,
        gallery_count=gallery_count,
        result=result,
        uploaded_preview_uri=uploaded_preview_uri,
    )

    if start and uploaded is not None:
        st.session_state.pop("selected_candidate_id", None)
        query_path = _save_upload(uploaded, upload_root)
        result = run_pipeline(
            input_dir=gallery_dir,
            query_image=query_path,
            output_dir=output_dir,
            parts=selected_parts,
            ignore_parts=[],
            features=",".join(features),
            topk=int(topk),
            device=None,
            skip_cutout=False,
        )
        st.session_state["last_result"] = result
        _clear_query_params()
        st.rerun()


def _uploaded_preview_uri(uploaded) -> str | None:
    if uploaded is None:
        return None
    try:
        image = Image.open(uploaded)
        return _img_to_data_uri(image.convert("RGB"), max_size=(640, 480), fmt="JPEG", quality=88)
    except Exception:
        return None
