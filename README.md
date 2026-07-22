# 汽车图片相似度比对整合项目

本项目把三个来源算法统一到一个流程中：

1. 上传单张待比对图片 A，保存到 `./img/_uploads/`。
2. 逐张遍历 `./img/front/` 中的 front 角度图库图片，与 A 做比对。
3. 用 YOLO 部件分割模型识别 front 角度部件，标注图输出到 `./result/front_label/<run_id>/`，部件截图输出到 `./result/front_parts/<run_id>/`。
4. 用 BiRefNet 对部件截图做主体识别，输出到 `./result/img-cutout/<run_id>/`。
5. 用 CDSE 对 `right_mirror`、`front_right_light`、`front_bumper`、`grille`、`hood`、`front_glass` 做横向相似度比对，支持忽略指定部件。
6. 用 YOLO COCO 分割整车轮廓并融合部件分，生成 JSON 和 Markdown 评判报告。

## 目录约定

- `./img/front/`: front 角度图库图片。
- `./img/_uploads/`: 页面上传的待比对图片 A。
- `./models/`: 已集中模型文件。
- `./result/`: 全部输出文件。bat
- `./tools/`: 来源算法脚本，已补充 `--input-dir`、`--weight`、`--output-dir` 参数别名。

## 命令行运行

```bash
python similarity_pipeline.py ^
  --input-dir ./img/front ^
  --query-image ./some_car.jpg ^
  --weight ./configs/default_weights.json ^
  --output-dir ./result ^
  --ignore-parts grille,front_glass
```

启用 CLIP：

```bash
python similarity_pipeline.py --input-dir ./img/front --query-image ./some_car.jpg --enable-clip
```

## 页面运行

```bash
pip install -r backend/requirements.cpu.txt
python -m streamlit run app.py
```

页面流程参考 `D:\work\中安\图像\PROJ\testpics\yolov11-seg-20260426\app.py`，但改为“上传单张 A，对比 ./img/front 图库”的单图检索式流程。

CPU 环境下 BiRefNet 会明显偏慢；调试页面流程时可以勾选“跳过 BiRefNet 主体识别”，或命令行增加 `--skip-cutout`。正式跑分建议使用 CUDA。

## 关键模型

- `models/yolo-seg/front-mirror-light-200epoch2.pt`: front 角度 `right_mirror`、`front_right_light`、`grille`。
- `models/yolo-seg/yolo11m-seg-carpart-epoch70.pt`: front 通用部件，如 `front_bumper`、`front_glass`、`hood`。
- `models/BiRefNet/`: 部件主体识别。
- `models/dino/`: DINOv2 本地仓库和权重。
- `models/clip-vit-large-patch14/`: CLIP 最小可运行本地文件。
- `models/BiRefNet/`: 普通背景图片的整车背景去除与轮廓 Mask 提取。

## Docker 构建（避免 segment-anything 下载超时）

Docker 构建时如果无法稳定访问 GitHub，可以把 `segment-anything` 的源码 zip 预先放到项目内，Dockerfile 会优先走本地安装。

- 下载：`https://github.com/facebookresearch/segment-anything/archive/refs/heads/main.zip`
- 保存到：`vendor/segment-anything-main.zip`

## V4 向量检索部署

V4 保留原教师 Pipeline 作为回退，通过 `SIM_RETRIEVAL_MODE` 选择运行模式：

- `legacy`：只运行原逐图库评分流程。
- `v4`：运行 V4 查询编码和 FAISS 检索，失败或非 front 图片自动回退。
- `shadow`：立即返回 V4，后台在 `result/v4-shadow/` 运行旧流程用于对照。

部署前确保 `gallery_store/indexes/v4/` 包含 `best.pt`、`manifest.json`、
`vector_score.f16.npy`、`valid_mask.u8.npy`、`gallery_manifest.parquet` 和
`faiss_score.index`。GPU Docker 启动命令：

```bash
SIM_V4_GALLERY_HOST_ROOT=/data/kuangyong/proj/similarity_api_ui2/gallery_store \
SIM_V4_PROJECT_HOST_ROOT=/data/kuangyong/proj/similarity_teach_data_pre \
docker compose \
  -f docker-compose.yml \
  -f docker-compose.gpu.yml \
  -f docker-compose.v4.yml \
  up -d --build
```

接口级临时灰度可以在 `/api/compare` 表单中传入 `retrieval_mode=legacy|v4|shadow`，
未传时使用环境变量。`/health?deep=true` 会返回 V4 是否就绪、图库数量和模型指纹。
