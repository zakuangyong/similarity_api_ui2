# similarity_teach_data_pre 第二阶段实施计划

## 0. 实施状态（2026-07-21）

当前代码实施已完成计划中的最小闭环：

- [x] Artifact Schema 2.0、六部件固定键、YOLO/SAM/BiRefNet/FeatureRef分层结构。
- [x] YOLO+SAM单图处理、BiRefNet整车处理、透明图alpha快速路径。
- [x] CLIP/DINO/Gray/Edge单图缓存，Pair Builder不加载重模型。
- [x] 轮廓、六部件、旧版教师分和质量修正版真实评分。
- [x] 无序pair、Parquet分块、损坏block重建和`--resume`校验。
- [x] 模型manifest校验、模型内容SHA256、配置与pipeline缓存身份。
- [x] DatasetManifest记录模型/编码器指纹，以及每个block的行数、大小、SHA256和Parquet Schema SHA256。
- [x] `verify-parity`逐项校验轮廓、CLIP、DINO、SSIM、Edge、部件融合和最终教师分。
- [x] 非真实模型测试123个全部通过；3张真实模型冒烟测试已通过。

尚未完成的验收项依赖真实数据或目标运行环境：

- [ ] 将模型权重按`models/manifest.json`安装到目标项目`models/`；当前真实冒烟配置仍指向来源项目模型目录。
- [ ] 固定20至50张真实回归图片并导出来源pipeline的`baseline_scores.json`。
- [ ] 运行`verify-parity --tolerance 0.1`并归档正式回归报告。
- [ ] 在目标GPU完成100张、1000张及全量数据性能压测，记录各阶段P50/P95和显存峰值。
- [ ] 根据真实压测结果决定是否增加GPU批处理、特征常驻缓存或并行解码。

## 1. 文档信息

- 计划日期：2026-07-20
- 来源项目：`D:\work\中安\图像\PROJ\similarity_api_ui2`
- 目标项目：`D:\work\中安\图像\PROJ\similarity_teach_data_pre`
- 当前基线：64个骨架测试通过，但真实模型、单图预处理和教师评分尚未接通
- 本阶段目标：完成从原图到真实教师图片对Parquet的最小闭环
- 实施路径：Artifact Schema升级 -> YOLO+SAM单图处理 -> BiRefNet整车处理 -> CLIP/DINO/Edge/Gray缓存 -> 真实组件评分

## 2. 阶段结论

目标项目当前已具备CLI、配置、fingerprint、Pydantic Schema、Artifact原子写入、无序pair枚举和Parquet块写入等基础能力。本阶段不重复搭建骨架，重点完成以下缺口：

1. 将旧版扁平Artifact升级为可追溯的YOLO、SAM和特征分层Schema。
2. 将三个`NotImplementedError`适配器替换为真实模型实现。
3. 将`preprocess`从固定返回0改为可恢复的单图生产流程。
4. 将CLIP、DINO、Edge和Gray从逐图片对重复提取改为单图永久缓存。
5. 将Pair Builder中的固定0分替换为与现有pipeline一致的真实轮廓、部件和综合评分。

本阶段完成后，以下命令必须能够生成真实教师数据：

```powershell
python -m similarity_teach_data_pre preprocess `
  --input D:\dataset\front `
  --store D:\dataset\similarity_store `
  --config configs\default.yaml `
  --resume

python -m similarity_teach_data_pre validate-cache `
  --store D:\dataset\similarity_store

python -m similarity_teach_data_pre build-pairs `
  --store D:\dataset\similarity_store `
  --output D:\dataset\teacher_v1 `
  --block-size 10000 `
  --resume
```

## 3. 实施原则

### 3.1 算法一致性优先

- 第一版迁移尽量保持来源项目的输入尺寸、阈值、权重和图像预处理不变。
- 每个组件同时保存原始数值和取整后的兼容数值。
- 先生成`legacy_teacher_score`，确保与现有pipeline一致；质量修正版单独输出，不覆盖旧评分。
- 真实模型接入后必须通过固定样本的新旧pipeline逐项回归。

### 3.2 单图重模型只运行一次

- YOLO、SAM、BiRefNet、CLIP和DINO只能在`preprocess`阶段运行。
- `build-pairs`只能读取Artifact、Mask、Gray和Embedding缓存。
- Pair Builder中不得加载或初始化任何重模型。
- 修改评分权重不能使分割或Embedding缓存失效。

### 3.3 产物可追溯

- 每个特征必须记录来源文件hash、编码器fingerprint和预处理fingerprint。
- YOLO保存结构化中间结果；SAM最终mask和cutout作为部件特征标准输入。
- 原图、模型权重、配置、Schema和评分策略都必须有版本或fingerprint。

### 3.4 测试分层

- 单元测试使用轻量fixture和fake adapter，不依赖GPU。
- 集成测试使用5至10张固定图片验证完整缓存链路。
- 真实模型冒烟测试使用1至3张图片，使用pytest marker隔离。
- 回归测试使用20至50张固定图片，对比来源pipeline输出。

## 4. 前置工作

### 4.1 目标项目版本管理

目标项目当前不是Git仓库。正式实施前完成：

- [ ] 初始化Git仓库并创建基线提交。
- [ ] 增加`.gitignore`，排除`.pytest_cache/`、`.pytest_tmp*/`、`tmp*/`、`store/`、`teacher_dataset/`、模型权重和生成数据。
- [ ] 将当前64个通过的测试作为阶段基线。
- [ ] 更新README，移除“只有Task 1骨架”的过期说明。
- [ ] 每完成一个阶段形成独立、可回滚提交。

### 4.2 固定回归样本

从现有图库和上传样本中选择20至50张图片，至少覆盖：

- 六部件全部识别成功。
- 缺失一个或多个部件。
- 透明背景cutout。
- 普通复杂背景。
- 中文文件名和长文件名。
- 同车型高相似、同品牌近似车型和明显不同车型。

保存来源pipeline的以下基线：

```text
YOLO类别、置信度、bbox
SAM最终mask面积和cutout路径
BiRefNet整车mask
CLIP、DINO、SSIM、Edge单项分
轮廓分、部件综合分、最终分
缺失部件和实际使用权重
```

### 4.3 模型资产策略

目标项目最终必须独立运行，不得通过Python路径直接import来源仓库。实施期间可从来源项目迁移算法代码，但必须保留来源文件和版本记录。

需要准备的模型：

```text
YOLO：models/yolo-seg/yolo11m-seg-front-6label-2000.pt
SAM：models/sam/sam_vit_h.pth
BiRefNet：models/BiRefNet/
CLIP：models/clip-vit-large-patch14/
DINO源码：models/dino/facebookresearch_dinov2_main/
DINO权重：models/dino/dinov2_vitb14_reg4_pretrain.pth
```

模型文件不进入Git；通过`models/manifest.json`记录相对路径、文件大小、SHA256和用途。启动真实任务前由`validate-models`或配置加载阶段检查完整性。

## 5. 阶段一：升级Artifact Schema

### 5.1 目标

将当前`schema_version=1.0`升级为`2.0`，明确区分原图、整车、YOLO中间结果、SAM最终结果和特征引用。由于当前项目尚未产生正式Artifact，本阶段允许破坏性升级，不实现1.0数据迁移器。

### 5.2 目标数据结构

新增或调整以下Schema：

```text
YoloArtifact
SamArtifact
FeatureRef
FeatureBundle
QualityArtifact
VehicleArtifact
PartArtifact
ImageArtifact
PairScore
DatasetManifest
```

推荐的单部件结构：

```json
{
  "part_name": "grille",
  "status": "valid",
  "yolo": {
    "class_name": "grille",
    "confidence": 0.94,
    "bbox_xyxy": [100, 200, 500, 400],
    "bbox_normalized": [0.05, 0.18, 0.26, 0.37],
    "mask_path": "...",
    "model_fingerprint": "..."
  },
  "sam": {
    "status": "valid",
    "mask_path": "...",
    "cutout_path": "...",
    "crop_bbox_xyxy": [92, 192, 508, 408],
    "predicted_iou": 0.93,
    "stability_score": 0.96,
    "model_fingerprint": "..."
  },
  "quality": {
    "score": 0.94,
    "mask_area_ratio": 0.08,
    "reasons": []
  },
  "feature_source_hash": "...",
  "features": {
    "clip": {"path": "features.npz", "key": "grille_clip", "dim": 768},
    "dino": {"path": "features.npz", "key": "grille_dino", "dim": 768},
    "edge": {"path": "features.npz", "key": "grille_edge"},
    "gray": {"path": "features.npz", "key": "grille_gray"}
  }
}
```

`parts`必须始终包含以下六个键：

```text
front_glass
front_right_light
front_bumper
grille
hood
right_mirror
```

缺失部件保留键，并使用`not_detected`、`invalid_mask`、`low_quality`或`processing_error`状态，不得省略，不得用0分表示缺失。

### 5.3 PairScore升级

`PairScore`至少新增：

```text
legacy_teacher_score
quality_adjusted_teacher_score
teacher_target_name
sample_weight
part_details[6]
weights_used
common_valid_parts
missing_parts_a
missing_parts_b
one_sided_missing_parts
```

保留旧的`final_teacher_score`会造成训练目标不明确，建议删除并通过`teacher_target_name`选择正式目标。若需要兼容旧消费者，可将其改为只读别名并在Schema 3.0删除。

### 5.4 需要修改的目标项目文件

- `src/similarity_teach_data_pre/schemas.py`
- `src/similarity_teach_data_pre/cache/validation.py`
- `src/similarity_teach_data_pre/cache/manifest.py`
- `src/similarity_teach_data_pre/scoring/teacher_score.py`
- `tests/test_schemas.py`
- `tests/test_artifact_store.py`
- `tests/test_missing_policy.py`

### 5.5 测试要求

- [ ] 六个部件键始终存在且顺序稳定。
- [ ] YOLO有效而SAM失败时能够合法序列化。
- [ ] `status=valid`时强制要求SAM mask和cutout存在。
- [ ] 特征存在时强制要求`feature_source_hash`存在。
- [ ] 二值mask、RGBA cutout和NPZ引用缺失时缓存校验失败。
- [ ] `legacy_teacher_score`和质量修正版均限制在`[0, 100]`。
- [ ] A/B交换后PairScore元数据仍满足对称约束。

### 5.6 完成标准

- Schema版本升级为2.0。
- 所有旧测试完成适配并通过。
- 新增Schema测试不少于15个。
- `validate-cache`能够验证YOLO、SAM和FeatureRef引用。
- README包含2.0 Artifact示例。

## 6. 阶段二：接入YOLO+SAM单图处理

### 6.1 目标

实现真实`PartSegmenter`，输入单张图片，输出六部件的YOLO结构化结果和SAM最终mask/cutout；同时将`run_preprocess`改造成可遍历、可缓存、可恢复的单图流水线。

### 6.2 来源能力

优先迁移：

- `tools/car_front_seg.py::detect_processed_instances`
- `tools/car_front_seg.py::load_rgb_with_white_bg`
- `tools/car_front_seg.py::export_rgba_crops`
- `tools/cutout_by_sam.py::load_sam_predictor`
- `tools/cutout_by_sam.py::run_sam_cutout_from_instances`

迁移算法逻辑，不迁移run目录、查询图角色、全量标注预览和CLI耦合。

### 6.3 实现设计

新增真实适配器，例如：

```text
extractors/yolo_sam_part_segmenter.py
models/model_registry.py
io/image_io.py
```

要求：

- YOLO和SAM在进程内各加载一次，不能每张图片重新加载。
- Adapter接受模型对象或ModelRegistry注入，便于测试fake模型。
- SAM输入使用YOLO处理后的部件bbox和mask提示，保持来源pipeline参数一致。
- SAM最终mask保存为单通道无损PNG，像素值只能是0或255。
- SAM cutout保存为RGBA PNG，透明背景alpha为0。
- YOLO全量预览图默认不保存；失败样本和配置抽样可保存。
- 所有路径写入Artifact前必须完成文件存在性校验。

### 6.4 preprocess编排

`run_preprocess`需要实现：

```text
遍历支持的图片扩展名
  -> 计算content_sha256
  -> 计算segmentation_fingerprint
  -> 查询Artifact缓存
  -> 缓存未命中时执行YOLO+SAM
  -> 写入临时Artifact目录
  -> 校验完整性
  -> 原子切换正式目录
  -> 更新images/parts manifest
```

CLI行为：

- `--dry-run`输出待处理、缓存命中和无效图片数量，不加载模型。
- `--resume`跳过完整有效Artifact，重试失败或不完整Artifact。
- 单图失败记录错误并继续，除非模型初始化失败。
- 进程退出码区分输入错误、模型错误和部分失败。

### 6.5 配置扩展

`configs/default.yaml`新增：

```yaml
models:
  yolo_front_parts: models/yolo-seg/yolo11m-seg-front-6label-2000.pt
  sam_checkpoint: models/sam/sam_vit_h.pth
  sam_type: vit_h

part_segmentation:
  conf: 0.25
  iou: 0.70
  imgsz: 640
  box_margin_ratio: 0.03
  keep_largest_component: true
  save_yolo_preview_sample_rate: 0.01
```

### 6.6 测试要求

- [ ] Fake YOLO和Fake SAM集成测试能够生成完整Artifact。
- [ ] 未检测部件形成负缓存，重复运行不再次调用SAM。
- [ ] YOLO识别但SAM空mask时状态为`invalid_mask`。
- [ ] 中文路径和PNG/JPEG/WebP输入可处理。
- [ ] 中断后完整图片保持有效，半成品不进入manifest。
- [ ] 真实模型对1至3张图片完成冒烟测试。
- [ ] 真实输出与来源pipeline的bbox、mask面积和部件集合一致。

### 6.7 完成标准

- `preprocess`能够对5至10张图片生成真实YOLO+SAM产物。
- 第二次执行缓存命中率为100%，YOLO和SAM调用次数为0。
- 每个有效部件都有YOLO元数据、SAM mask和SAM cutout。
- 单图和批量处理均不会每图重复加载模型。

## 7. 阶段三：接入BiRefNet整车处理

### 7.1 目标

实现真实`VehicleSegmenter`，为每张图片生成整车mask、整车RGBA cutout和轮廓评分所需的标准化mask。

### 7.2 来源能力

迁移：

- `tools/cutout_by_birefnet.py::BiRefNetSegmenter`
- `tools/cutout_by_birefnet.py::load_birefnet_segmenter`
- `similarity_pipeline.py::_alpha_mask_from_image`
- `similarity_pipeline.py::_birefnet_cutout`

### 7.3 实现设计

- 如果原图包含真实透明背景，优先直接使用alpha mask，避免BiRefNet推理。
- 普通背景图片才调用BiRefNet。
- BiRefNet模型在进程内常驻，处理完成后由任务级生命周期统一释放。
- 原始mask用于审计和cutout；标准化mask用于轮廓评分。
- 标准化步骤必须保持当前`keep_aspect`和`align=centroid`规则。
- 保存mask面积比例、是否触边、连通域数量和质量分。

### 7.4 Artifact字段

`vehicle`至少包含：

```text
status
source_mode：alpha或birefnet
mask_path
cutout_path
normalized_mask_path
mask_area_ratio
quality
model_fingerprint
```

### 7.5 测试要求

- [ ] 透明PNG走alpha快速路径，不调用BiRefNet。
- [ ] 普通JPEG调用BiRefNet并生成有效mask。
- [ ] 空mask、全mask和异常小mask被拒绝或降级。
- [ ] 同一图片重复执行命中缓存。
- [ ] 新旧pipeline轮廓mask面积误差在约定阈值内。
- [ ] 固定20个图片对的轮廓分误差不超过0.1分。

### 7.6 完成标准

- 5至10张试验图片全部生成整车Artifact或明确错误状态。
- 透明图无需BiRefNet即可完成。
- `validate-cache`能够检查整车mask、cutout和标准化mask。
- 轮廓评分输入与来源pipeline保持一致。

## 8. 阶段四：缓存CLIP/DINO/Edge/Gray

### 8.1 目标

实现真实`FeatureEncoder`，在单图阶段对整车和有效部件提取可复用特征，并保存到每图独立的二进制特征文件。图片对阶段不得再次调用模型或图像预处理函数。

### 8.2 来源能力

迁移：

- `tools/cdse_similarity.py::CdseSimilarityEngine`
- `tools/cdse_similarity.py::preprocess_color`
- `tools/cdse_similarity.py::preprocess_gray`
- `tools/cdse_similarity.py::extract_edge`
- `tools/cdse_similarity.py::_extract_clip`
- `tools/cdse_similarity.py::_extract_dino`

不能直接复用`compare_paths()`作为pair评分入口，因为该函数会重复提取A/B特征。

### 8.3 特征内容

每个有效组件至少缓存：

```text
clip：L2归一化float32或float16向量
dino：L2归一化float32或float16向量
edge：当前算法使用的标准化edge表示
gray：当前SSIM使用的224x224灰度图或等价数组
```

整车至少缓存DINO或预留全局向量接口，但本阶段教师评分仍以轮廓分和六部件分为准。

### 8.4 存储格式

每张图片推荐写入一个`features.npz`：

```text
vehicle_dino
front_glass_clip
front_glass_dino
front_glass_edge
front_glass_gray
front_right_light_clip
...
```

Artifact只保存`path + key + dtype + shape + dim + encoder_fingerprint + source_hash`。禁止将高维向量直接写入JSON。

如果NPZ写入和随机读取性能不能满足批量评分，再评估SafeTensors或分块Arrow；第一版不提前引入复杂存储。

### 8.5 批处理策略

- 同一图片的整车和六部件组成一个batch或小批次进入CLIP/DINO。
- 查询图片内相同部件特征只提取一次。
- CLIP和DINO模型由ModelRegistry统一常驻。
- CPU线程负责解码、Gray、Edge和写盘；GPU单进程执行CLIP/DINO。
- 显存不足时按组件batch降级，不启动多个共享GPU模型的线程。

### 8.6 Fingerprint规则

特征缓存键必须包含：

```text
SAM cutout内容hash
编码器权重hash
输入分辨率
预处理版本
dtype
特征类型
```

修改教师评分权重不得使特征缓存失效；修改DINO权重只能重建DINO特征，不应重跑YOLO、SAM或BiRefNet。

### 8.7 测试要求

- [ ] Fake Encoder能够写入并重新读取NPZ。
- [ ] 数组shape、dtype、L2范数满足契约。
- [ ] 修改评分配置不改变feature fingerprint。
- [ ] 修改编码器权重hash只使对应特征失效。
- [ ] FeatureRef的source hash与SAM cutout一致。
- [ ] 第二次预处理不会调用CLIP或DINO。
- [ ] 真实模型对1至3张图片完成特征冒烟测试。
- [ ] 固定部件的新旧CLIP、DINO、Edge和Gray结果数值一致。

### 8.8 完成标准

- 5至10张图片可生成完整特征缓存。
- 有效部件的CLIP/DINO余弦自相似度接近1。
- Pair Builder读取特征时不初始化模型。
- 单图特征文件损坏时`validate-cache`能够定位具体图片和键名。

## 9. 阶段五：实现真实组件评分

### 9.1 目标

将当前固定0分的`build_component_score_row()`替换为真实评分，实现轮廓、六部件、旧版综合分和质量修正版，并批量写入Parquet。

### 9.2 评分组成

轮廓评分：

```text
读取A/B标准化整车mask
  -> 调用contour_similarity纯评分逻辑
  -> 默认不生成差异图
  -> 输出contour_score
```

部件评分：

```text
CLIP：缓存向量余弦
DINO：缓存向量余弦
SSIM：缓存Gray数组计算
Edge：缓存Edge表示余弦
  -> 按部件feature weights融合
  -> 按part overall weights融合
```

当前部件权重从来源配置迁移，不在代码中硬编码。

### 9.3 双教师分数

必须同时输出：

```text
legacy_teacher_score
  = 完全复刻来源pipeline的动态可用权重归一化

quality_adjusted_teacher_score
  = coverage * evidence_score
    + (1 - coverage) * prior_score
    - lambda * one_sided_missing_weight
```

第一版默认：

```text
teacher_target_name = legacy_teacher_score
```

只有质量修正版通过人工审核和检索指标验证后，才能更改正式训练目标。

### 9.4 Parquet字段

每个图片对至少保存：

```text
pair_id
image_a_id、image_b_id
pipeline_fingerprint
score_policy_version
contour_score
part_score
六部件的clip、dino、ssim、edge、fused
六部件A/B状态和质量
common_valid_parts
missing_parts_a、missing_parts_b
weights_used
coverage、confidence、sample_weight
legacy_teacher_score
quality_adjusted_teacher_score
teacher_target_name
```

推荐使用固定PyArrow Schema，六部件使用固定struct列或固定前缀列，避免不同block自动推断出不兼容类型。

### 9.5 Pair Builder改造

- 保持无序pair和稳定`pair_id`。
- 按block读取Artifact和特征，不一次加载全部图片对。
- block成功写入Parquet并校验行数后再写完成checkpoint。
- `--resume`必须验证已有block，而不是仅看到文件存在就跳过。
- 混用不同pipeline fingerprint的Artifact时明确拒绝。
- 默认不生成轮廓差异图和部件热力图；审计命令可按样本生成。
- 输出`dataset_manifest.json`，记录图片数、pair数、block数、模型hash、配置hash、Schema版本和评分策略。

### 9.6 测试要求

- [ ] A/B交换后所有组件分和最终分相同。
- [ ] 完整六部件样本的新旧单项分误差不超过0.1分。
- [ ] `legacy_teacher_score`与来源pipeline最终分误差不超过0.1分。
- [ ] 缺失部件不会被当作0分参与旧版融合。
- [ ] 质量修正版coverage和单侧缺失惩罚符合配置。
- [ ] 100张图片严格生成4,950个唯一无序pair。
- [ ] Pair Builder运行期间YOLO、SAM、BiRefNet、CLIP和DINO调用次数为0。
- [ ] 中断后能够从最后一个完整block继续。
- [ ] 不同block具有完全相同的Parquet Schema。

### 9.7 完成标准

- 20至50张回归图片可生成真实pair数据。
- 新旧pipeline逐项评分满足误差阈值。
- Parquet无固定0分占位、无重复pair、无非预期空字段。
- 数据集manifest能够完整复现模型、配置和评分版本。

## 10. 端到端验收

### 10.1 功能验收

- [ ] 5张图片能够完成`preprocess -> validate-cache -> build-pairs -> audit`闭环。
- [ ] 100张图片能够生成4,950个真实教师样本对。
- [ ] 同一批图片第二次预处理时重模型调用次数为0。
- [ ] 删除单个特征文件后`validate-cache`能准确报错。
- [ ] 修复或重建单个图片不要求重跑全库。
- [ ] Pair Builder不依赖来源项目运行时路径。

### 10.2 一致性验收

- 轮廓分误差：不超过0.1分。
- 单部件CLIP/DINO/SSIM/Edge误差：不超过0.1分。
- 部件综合分误差：不超过0.1分。
- `legacy_teacher_score`误差：不超过0.1分。
- A/B顺序对结果无影响。
- 固定模型和配置重复运行结果一致。

### 10.3 性能验收

在目标GPU上记录：

```text
YOLO平均单图耗时
SAM平均单图和单部件耗时
BiRefNet平均单图耗时
CLIP/DINO每组件及batch耗时
单个Artifact平均磁盘占用
每10,000个pair的评分和Parquet写入耗时
GPU峰值显存、CPU峰值内存
```

阶段目标不是立即满足在线30秒，而是确保图库重模型计算能够永久缓存，图片对阶段不再产生GPU推理。

### 10.4 测试命令

```powershell
# 快速单元测试
python -m pytest -m "not real_models" -q

# 真实模型冒烟测试
python -m pytest -m real_models tests\integration -v

# 固定样本新旧回归
python -m pytest tests\regression -v

# 全量测试
python -m pytest -q
```

## 11. 文件级实施清单

### 11.1 目标项目修改

```text
pyproject.toml
README.md
configs/default.yaml
configs/missing_score.yaml
src/similarity_teach_data_pre/config.py
src/similarity_teach_data_pre/schemas.py
src/similarity_teach_data_pre/fingerprint.py
src/similarity_teach_data_pre/cli.py
src/similarity_teach_data_pre/cache/artifact_store.py
src/similarity_teach_data_pre/cache/manifest.py
src/similarity_teach_data_pre/cache/validation.py
src/similarity_teach_data_pre/extractors/part_segmenter.py
src/similarity_teach_data_pre/extractors/vehicle_segmenter.py
src/similarity_teach_data_pre/extractors/feature_encoder.py
src/similarity_teach_data_pre/extractors/quality.py
src/similarity_teach_data_pre/pipeline/preprocess.py
src/similarity_teach_data_pre/pipeline/build_pairs.py
src/similarity_teach_data_pre/pipeline/audit.py
src/similarity_teach_data_pre/scoring/component_scores.py
src/similarity_teach_data_pre/scoring/missing_policy.py
src/similarity_teach_data_pre/scoring/teacher_score.py
src/similarity_teach_data_pre/exporters/parquet_writer.py
```

### 11.2 建议新增文件

```text
models/manifest.json
src/similarity_teach_data_pre/models/model_registry.py
src/similarity_teach_data_pre/io/image_io.py
src/similarity_teach_data_pre/extractors/yolo_sam_part_segmenter.py
src/similarity_teach_data_pre/extractors/birefnet_vehicle_segmenter.py
src/similarity_teach_data_pre/extractors/cdse_feature_encoder.py
src/similarity_teach_data_pre/scoring/contour_score.py
src/similarity_teach_data_pre/scoring/part_score.py
src/similarity_teach_data_pre/cache/locking.py
tests/integration/test_preprocess_fake_models.py
tests/integration/test_real_model_smoke.py
tests/regression/test_pipeline_parity.py
tests/fixtures/baseline_scores.json
```

## 12. 依赖与配置管理

当前`pyproject.toml`只包含数据工程依赖。本阶段需要增加模型相关依赖，但建议拆成可选依赖，保证不带GPU环境也能运行Schema和Pair Builder测试：

```toml
[project.optional-dependencies]
models = [
  "torch",
  "torchvision",
  "ultralytics",
  "opencv-python",
  "Pillow",
  "transformers",
  "scikit-image"
]
dev = ["pytest", "pytest-cov"]
```

`segment-anything`根据现有部署方式使用本地vendor包或固定commit安装，禁止运行时从网络临时下载。

配置至少拆分为：

```text
模型路径与设备
YOLO/SAM参数
BiRefNet参数
特征预处理参数
部件feature weights
部件overall weights
轮廓参数
旧版综合权重
质量修正版策略
缓存和输出参数
```

## 13. 任务顺序与里程碑

| 里程碑 | 内容 | 前置依赖 | 预计工作量 |
| --- | --- | --- | ---: |
| M0 | Git、README、回归样本和模型manifest | 无 | 0.5至1人日 |
| M1 | Artifact Schema 2.0和校验 | M0 | 1至2人日 |
| M2 | YOLO+SAM单图闭环和缓存 | M1 | 2至4人日 |
| M3 | BiRefNet整车mask和轮廓输入 | M2 | 1至2人日 |
| M4 | CLIP/DINO/Edge/Gray单图缓存 | M2、M3 | 2至4人日 |
| M5 | 真实组件评分和Parquet闭环 | M4 | 2至4人日 |
| M6 | 100张压测、新旧回归和文档 | M5 | 1至2人日 |

单人实施粗估9至19人日，实际取决于目标GPU环境、模型依赖是否已可直接运行以及旧算法迁移中的路径兼容问题。

## 14. 风险与控制

| 风险 | 影响 | 控制措施 |
| --- | --- | --- |
| Schema先接模型后频繁变化 | 缓存反复失效 | 先完成Schema 2.0和fake adapter集成测试 |
| SAM ViT-H模型过大 | 初始化慢、显存不足 | 模型常驻、单GPU进程、记录峰值显存 |
| 直接复用`compare_paths()` | 图片对阶段重复提取特征 | 拆分单图extract和缓存compare接口 |
| 中文路径读取失败 | 批处理漏图 | 复用`np.fromfile + cv2.imdecode`路径方案并测试 |
| 新旧图像预处理不一致 | 教师分漂移 | 固定20至50张回归集逐项对比 |
| YOLO成功但SAM失败 | Artifact状态混乱 | YOLO和SAM分层状态，失败不覆盖中间证据 |
| 模型或配置变化混用缓存 | 标签不可复现 | 分层fingerprint、模型manifest和dataset manifest |
| Pair Builder只检查文件存在 | 损坏block被错误跳过 | checkpoint保存行数、hash和Parquet Schema摘要 |
| 质量修正版过早替代旧评分 | 学生目标发生无依据变化 | 双教师标签，默认拟合legacy分数 |
| 大量临时目录污染项目 | 审计和版本管理困难 | 统一临时根目录、Git忽略、任务结束清理 |

## 15. 本阶段退出条件

只有同时满足以下条件，才进入下一阶段的图片对采样和学生向量训练：

- [ ] Artifact Schema 2.0稳定并有版本化示例。
- [ ] YOLO、SAM、BiRefNet、CLIP和DINO均为真实实现，不再抛`NotImplementedError`。
- [ ] `preprocess`能够生成真实、完整、可校验的Artifact。
- [ ] 第二次预处理不调用重模型。
- [ ] Pair Builder中所有组件分来自缓存，不存在固定0分占位。
- [ ] `legacy_teacher_score`与来源pipeline误差不超过0.1分。
- [ ] 100张图片生成4,950个真实、唯一、可恢复的pair。
- [ ] 所有单元、集成、真实模型冒烟和回归测试通过。
- [ ] README、配置说明、模型manifest和数据集manifest完整。

完成以上条件后，下一阶段才能安全开展分层pair采样、困难样本挖掘、教师数据扩展、学生向量蒸馏和FAISS索引训练。
