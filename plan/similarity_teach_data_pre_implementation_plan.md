# similarity_teach_data_pre 项目实施计划

## 1. 文档信息

- 计划日期：2026-07-20
- 来源项目：`similarity_api_ui2`
- 目标项目：`D:\work\中安\图像\PROJ\similarity_teach_data_pre`
- 项目类型：离线教师数据预处理与训练样本生成
- 核心目标：将当前相似度pipeline拆分为“单图处理一次、图片对重复计算”的数据生产系统，为后续512至768维学生向量训练和FAISS检索提供可复现、可审计的教师数据。

## 2. 可行性结论

项目可行。现有项目已经具备YOLO部件分割、SAM抠图、BiRefNet整车轮廓、CLIP/DINO/SSIM/edge部件评分和综合评分能力，主要改造工作是解除这些能力与单次run目录、查询图角色、报告输出之间的耦合。

新项目不直接调用`similarity_pipeline.run_pipeline()`生成教师数据。应抽取并重组底层能力，形成以下两级稳定接口：

```python
extract_image_artifact(image_path, config) -> ImageArtifact
compare_artifacts(artifact_a, artifact_b, score_policy) -> PairScore
```

新项目第一阶段不负责FAISS在线检索、API和前端展示，但输出的数据契约必须能够直接支持后续512至768维向量蒸馏和FAISS索引构建。

## 3. 范围定义

### 3.1 本期范围

- 图片注册、去重和内容哈希。
- 整车轮廓与六个部件的单图预处理。
- 单图裁剪图、mask、预处理数据和embedding持久化缓存。
- 缓存版本管理、失效判断、中断续跑和完整性检查。
- 部件状态、检测置信度和mask质量记录。
- 基于缓存生成无序图片对。
- 轮廓、六部件和综合教师评分。
- 缺失部件的coverage、confidence及保守评分。
- 分块生成Parquet教师训练集。
- 与当前pipeline的基线一致性验证。

### 3.2 非本期范围

- 在线查询API和Web UI。
- FAISS索引构建与在线检索。
- 512至768维学生向量模型训练。
- 人工标注平台。
- 完整替换当前`similarity_api_ui2`线上pipeline。

## 4. 当前主流程与抽取边界

当前主流程：

```text
输入图片与图库
  -> run级图片复制和item_id生成
  -> YOLO部件分割
  -> SAM部件mask精修
  -> BiRefNet整车mask
  -> 查询图与候选图逐对提取CDSE特征
  -> 轮廓评分与六部件评分
  -> 动态权重融合
  -> TopK、JSON、Markdown和API输出
```

目标主流程：

```text
图片注册
  -> 按内容hash查询缓存
  -> 缓存未命中时执行一次单图分割和特征提取
  -> 写入ImageArtifact和manifest
  -> 从manifest按block枚举图片对
  -> 仅读取缓存计算pair score
  -> 应用缺失策略和综合评分
  -> 分区写入Parquet
```

现有代码的抽取映射：

| 现有模块 | 复用内容 | 改造要求 |
| --- | --- | --- |
| `tools/car_front_seg.py` | YOLO结果解析、部件筛选、mask后处理 | 返回结构化部件对象，不只导出PNG |
| `tools/cutout_by_sam.py` | SAM mask精修规则 | 返回精修mask及质量指标 |
| `tools/cutout_by_birefnet.py`、`similarity_pipeline.py` | BiRefNet模型加载与整车抠图 | 移除run级依赖，封装单图segmenter |
| `tools/contour_similarity.py` | 轮廓数值评分 | 拆分纯评分与可视化，批处理默认不渲染 |
| `tools/cdse_similarity.py` | CLIP/DINO/edge提取和SSIM | 拆分单图特征提取与缓存特征比较 |
| `configs/default_weights.json` | 模型路径、部件和初始权重 | 转为独立配置并记录配置fingerprint |

不直接复用以下职责：

- run级`input_flat`复制。
- query/gallery角色绑定。
- 每个图片对生成轮廓差异图。
- TopK截断。
- Markdown报告和前端数据适配。

## 5. 项目结构

```text
similarity_teach_data_pre/
├── pyproject.toml
├── README.md
├── configs/
│   ├── default.yaml
│   ├── missing_score.yaml
│   └── logging.yaml
├── src/similarity_teach_data_pre/
│   ├── __init__.py
│   ├── cli.py
│   ├── config.py
│   ├── schemas.py
│   ├── fingerprint.py
│   ├── extractors/
│   │   ├── part_segmenter.py
│   │   ├── vehicle_segmenter.py
│   │   ├── feature_encoder.py
│   │   └── quality.py
│   ├── cache/
│   │   ├── artifact_store.py
│   │   ├── manifest.py
│   │   ├── locking.py
│   │   └── validation.py
│   ├── scoring/
│   │   ├── component_scores.py
│   │   ├── missing_policy.py
│   │   └── teacher_score.py
│   ├── pipeline/
│   │   ├── register_images.py
│   │   ├── preprocess.py
│   │   ├── build_pairs.py
│   │   └── audit.py
│   └── exporters/
│       └── parquet_writer.py
└── tests/
    ├── fixtures/
    ├── test_fingerprint.py
    ├── test_artifact_store.py
    ├── test_missing_policy.py
    ├── test_teacher_score.py
    └── test_pipeline_parity.py
```

## 6. 数据契约

### 6.1 ImageArtifact

每张原图生成一个独立Artifact。整车信息放入`vehicle`，六个标准部件放入`parts`；YOLO结构化中间结果和SAM最终结果分层保存：

```json
{
  "schema_version": "1.0",
  "image_id": "sha256前缀或业务ID",
  "source_path": "...",
  "content_sha256": "...",
  "width": 1920,
  "height": 1080,
  "pipeline_fingerprint": "...",
  "vehicle": {
    "status": "valid",
    "mask_path": "...",
    "cutout_path": "...",
    "normalized_mask_path": "...",
    "embedding_path": "...",
    "quality": 0.96
  },
  "parts": {
    "grille": {
      "status": "valid",
      "yolo": {
        "class_name": "grille",
        "confidence": 0.91,
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
        "mask_area_ratio": 0.08,
        "mask_quality": 0.94
      },
      "feature_source": "sam.cutout_path",
      "features": {
        "clip": {"path": "...", "key": "grille_clip", "dim": 768},
        "dino": {"path": "...", "key": "grille_dino", "dim": 768},
        "edge": {"path": "...", "key": "grille_edge"},
        "gray": {"path": "...", "key": "grille_gray"}
      }
    }
  },
  "errors": [],
  "timings": {}
}
```

固定部件集合：

```text
front_glass
front_right_light
front_bumper
grille
hood
right_mirror
```

`parts`中六个键必须始终存在。未检测到的部件写入明确状态和空引用，不得省略键，也不得将缺失解释为相似度0。特征向量不直接展开写入JSON，推荐每张图片写入一个NPZ、SafeTensors或同类二进制容器，Artifact只保存路径、键名、维度和编码器fingerprint。

### 6.2 部件状态

允许状态必须枚举化：

| 状态 | 含义 | 是否参与相似度 |
| --- | --- | --- |
| `valid` | 检测、mask和裁剪均合格 | 是 |
| `low_quality` | 有结果但质量低于阈值 | 按质量折减或不参与 |
| `not_detected` | YOLO没有检测到 | 否 |
| `invalid_mask` | mask为空或几何异常 | 否 |
| `out_of_frame` | 部件明显超出画面 | 否 |
| `ignored` | 被配置明确忽略 | 不计入coverage分母 |
| `processing_error` | 模型或文件处理失败 | 否，并记录错误 |

### 6.3 PairScore

每个无序图片对输出：

```text
pair_id
image_a_id、image_b_id
contour_score
part_scores[6]
component_status_a[7]、component_status_b[7]
component_quality[7]
missing_parts_a、missing_parts_b
common_valid_parts、one_sided_missing_parts
evidence_score、coverage、confidence
legacy_teacher_score、quality_adjusted_teacher_score
teacher_target_name、sample_weight
pipeline_fingerprint、score_policy_version
```

`pair_id`根据排序后的两个`content_sha256`确定，保证换序后仍得到同一ID。

### 6.4 分层产物保存策略

| 层级 | 必须保存 | 可选保存 | 不建议长期保存 |
| --- | --- | --- | --- |
| 原始层 | 原图、内容hash、尺寸和来源 | 业务元数据 | 重复原图副本 |
| YOLO层 | 类别、置信度、bbox、结构化mask或polygon、模型与参数fingerprint | 失败样本和抽样预览图 | 全量标注预览图、推理Tensor、NMS前候选 |
| SAM层 | 最终二值PNG mask、RGBA cutout、质量指标、输入YOLO框 | 多候选mask的审计信息 | 有损JPEG mask |
| 特征层 | DINO、CLIP、edge、gray及预处理版本 | 便于诊断的缩略图 | JSON内展开高维向量 |

正式教师评分和学生训练默认使用SAM最终cutout。保留轻量YOLO结构化结果，使未来更换SAM、调整box margin或排查错误时无需重新运行YOLO。所有特征必须记录`feature_source_hash`，保证能够追溯到具体SAM产物。

## 7. 单图特征缓存方案

### 7.1 缓存层级

```text
分割缓存键 = 原图hash + 分割模型hash + 分割参数 + 后处理版本
特征缓存键 = crop或mask hash + 编码器hash + 预处理版本
评分缓存键 = 两个artifact版本 + score policy版本
```

模型hash至少覆盖YOLO、SAM、BiRefNet、DINO和CLIP权重。参数fingerprint至少覆盖`conf`、`iou`、`imgsz`、alpha threshold、输入分辨率及部件清理规则版本。

### 7.2 缓存目录

```text
store/
├── manifests/
│   ├── images.parquet
│   └── parts.parquet
├── artifacts/
│   └── {sha256前2位}/{sha256}/{pipeline_fingerprint}/artifact.json
├── masks/
├── crops/
├── preprocessed/
└── embeddings/
    └── {encoder_fingerprint}/...
```

### 7.3 必须具备的行为

- 同一图片内容即使路径或文件名改变，也能命中缓存。
- 修改评分权重不使分割和embedding缓存失效。
- 修改DINO权重只重建DINO相关embedding。
- 缺失部件写入负缓存，重复运行不再次尝试。
- 单个Artifact采用临时目录写入，完成后原子切换。
- 进程中断后已完成Artifact保持有效。
- manifest只引用完整且校验通过的Artifact。
- 缓存验证命令能够发现缺文件、hash不一致和schema不兼容。

### 7.4 性能策略

- GPU推理先采用单进程批处理，避免多个线程共享模型导致显存抖动或线程安全问题。
- CPU线程负责解码、hash、质量计算和写盘。
- CLIP/DINO按部件批量编码，不在图片对循环中调用模型。
- edge向量和标准化灰度图在单图阶段缓存。
- 轮廓mask在单图阶段裁边和标准化，但保留当前算法需要的原始几何信息。

## 8. 缺失部件评分规则

### 8.1 固定权重

使用绝对权重，禁止因为某个组件缺失而把剩余组件自动补满：

```text
整车轮廓：0.40
六部件合计：0.60
```

部件绝对权重为：

```text
absolute_part_weight_i = 0.60 * configured_part_weight_i
```

被配置为`ignored`的组件在本次任务中从权重集合移除，其余缺失组件保留权重但coverage为0。

### 8.2 质量系数

双方部件均有效时：

```text
q_i = sqrt(confidence_a * confidence_b) * mask_quality_i
```

`q_i`限定在`[0, 1]`。整车轮廓使用独立的`vehicle_quality`。任意一侧不可用时`q_i = 0`。

### 8.3 教师标签双轨制

教师数据必须同时保留两类分数：

```text
legacy_teacher_score = 当前pipeline完全一致的动态归一化最终分
quality_adjusted_teacher_score = 引入coverage、quality和缺失惩罚后的实验分
```

第一版学生向量默认拟合`legacy_teacher_score`，用于验证FAISS向量是否能复现现有排序；质量修正版在人工审核和离线指标确认优于旧规则后，才能切换为正式训练目标。禁止在没有基线对照的情况下直接用新缺失策略覆盖当前教师标签。

### 8.4 质量修正版评分输出

```text
effective_weight_i = w_i * q_i

evidence_score =
    sum(effective_weight_i * component_score_i)
    / sum(effective_weight_i)

coverage = sum(effective_weight_i)

quality_adjusted_teacher_score =
    coverage * evidence_score
    + (1 - coverage) * prior_score
    - lambda * one_sided_missing_weight
```

规则说明：

- `prior_score`使用训练集统计值或验证集校准值，不能直接硬编码为0。
- `one_sided_missing_weight`是仅A或仅B缺失组件的绝对权重之和。
- 双方均缺失时不增加单侧缺失惩罚，但降低coverage和confidence。
- 评分限定在`[0, 100]`。
- `evidence_score`、`coverage`、`legacy_teacher_score`和`quality_adjusted_teacher_score`必须同时保存，不能只保存单一最终分。
- `prior_score`、`lambda`、质量阈值和低质量处理策略写入独立版本化配置。

### 8.5 初始置信度规则

第一版可采用以下标记，阈值最终由验证集校准：

```text
coverage < 0.40：insufficient_evidence
有效部件数 = 0：contour_only
有效部件数 < 3：low_confidence
存在processing_error：processing_warning
```

低coverage结果可以保留在教师数据中，但训练时必须支持按coverage过滤或加权。

## 9. 教师样本生成

### 9.1 图片对规模

无序且不包含自身的图片对数量：

```text
pair_count = N * (N - 1) / 2
```

典型规模：

| 图片数 | 图片对数 |
| ---: | ---: |
| 1,000 | 499,500 |
| 1,415 | 1,000,405 |
| 3,000 | 4,498,500 |

`A-B`与`B-A`为同一无序样本，不能重复计数。1000张图片最多产生499,500个有效无序图片对，而不是约100万个独立训练对。

### 9.2 数据切分

必须先按图片、车辆实体或车型分组切分`train/validation/test`，再在各集合内部生成图片对。禁止先生成图片对再随机切分，否则同一图片会同时出现在训练和测试中，造成指标虚高。

推荐：

```text
train：70%
validation：15%
test：15%
```

若目标包含未见车型泛化，测试集应按车系或车型年份隔离；若存在同一车辆的多张图片，同一车辆必须完整落在同一集合。

### 9.3 图片对采样

小规模基线可枚举全部无序图片对；5000张以上默认采用分层采样，不建议生成全组合。初始采样比例：

| 样本类型 | 建议比例 |
| --- | ---: |
| 同车型、同年份或近重复 | 20% |
| 同车型、不同年份 | 15% |
| 同品牌、相近年份、不同车型 | 25% |
| 跨品牌但DINO近邻 | 25% |
| 随机低相似样本 | 15% |

目标是让每张图片参与30至100个有价值配对，并保证教师分数各区间有足够覆盖。第一版学生模型训练后，再基于假阳性和假阴性执行hard-negative/hard-positive mining。

### 9.4 生成策略

- 按固定block枚举pair，禁止一次构建完整pair列表。
- 每个block拥有确定性编号和状态文件。
- 已完成block跳过，失败block可单独重跑。
- CLIP、DINO和edge余弦分尽量采用矩阵运算。
- SSIM和轮廓评分读取缓存数据后按CPU block执行。
- 默认只写数值，不为全部pair生成差异图。
- 可配置抽样生成诊断图，用于人工审查。
- 每个Parquet文件控制在合理行数，避免单个超大文件。
- 图片对生成阶段不得调用YOLO、SAM、BiRefNet、CLIP或DINO模型。
- score分布按`0-20、20-40、40-60、60-75、75-85、85-100`分桶审计，避免低分样本淹没中高分样本。

### 9.5 输出布局

```text
teacher_dataset/
├── dataset_manifest.json
├── images.parquet
├── parts.parquet
├── pairs/
│   ├── block-000000.parquet
│   ├── block-000001.parquet
│   └── ...
├── checkpoints/
└── audit/
```

`dataset_manifest.json`记录数据范围、pair数量、模型版本、评分规则版本、生成时间和校验摘要。

## 10. CLI设计

```powershell
# 注册图片并生成单图缓存
python -m similarity_teach_data_pre preprocess `
  --input D:\dataset\front `
  --store D:\dataset\similarity_store `
  --config configs\default.yaml

# 检查缓存完整性
python -m similarity_teach_data_pre validate-cache `
  --store D:\dataset\similarity_store

# 分块生成教师图片对
python -m similarity_teach_data_pre build-pairs `
  --store D:\dataset\similarity_store `
  --output D:\dataset\teacher_v1 `
  --split-manifest D:\dataset\splits.json `
  --sampling-config configs\pair_sampling.yaml `
  --block-size 10000

# 抽样审计
python -m similarity_teach_data_pre audit `
  --dataset D:\dataset\teacher_v1 `
  --sample-size 500
```

所有命令必须支持`--resume`、`--dry-run`、日志文件和明确的非零错误码。

## 11. 分阶段实施

### 阶段0：基线冻结

任务：

- [ ] 选择20至50张覆盖典型情况的固定测试图片。
- [ ] 保存当前pipeline的单项分和最终分。
- [ ] 保存当前模型文件hash和完整配置。
- [ ] 记录当前耗时、显存和失败样本。
- [ ] 选取1000至3000个图片对建立人工审核集，审核集不参与训练。

完成标准：形成可重复运行的baseline数据包。

### 阶段1：项目骨架与Schema

任务：

- [ ] 创建独立Python package和CLI入口。
- [ ] 定义`ImageArtifact`、`PartArtifact`和`PairScore`。
- [ ] 实现配置加载、schema版本和fingerprint。
- [ ] 建立单元测试和最小CI命令。

完成标准：Schema可以序列化、反序列化并通过兼容性测试。

### 阶段2：单图分割与Artifact

任务：

- [ ] 抽取YOLO部件检测及后处理。
- [ ] 抽取SAM精修并返回mask，不只写PNG。
- [ ] 抽取BiRefNet整车mask。
- [ ] 计算部件置信度、mask面积和边界质量。
- [ ] 生成完整Artifact。

完成标准：单张图可以独立生成整车和六部件结构化结果。

### 阶段3：特征缓存

任务：

- [ ] 将CDSE拆成单图特征提取和特征比较。
- [ ] 持久化CLIP、DINO、edge及灰度预处理结果。
- [ ] 增加整车embedding接口，为后续512至768维向量预留。
- [ ] 实现缓存命中、分层失效、负缓存和原子写入。
- [ ] 实现manifest和缓存检查命令。

完成标准：第二次处理同一批图片时不调用重模型，缓存命中率接近100%。

### 阶段4：评分与缺失策略

任务：

- [ ] 实现七个固定绝对权重。
- [ ] 完整复刻当前动态权重融合，输出`legacy_teacher_score`。
- [ ] 实现组件质量系数。
- [ ] 实现`evidence_score`、`coverage`和`confidence`。
- [ ] 实现prior回归和单侧缺失惩罚。
- [ ] 将规则和参数完全配置化并版本化。
- [ ] 覆盖单侧缺失、双方缺失、全部缺失和ignored测试。
- [ ] 通过配置明确指定正式`teacher_target_name`，默认使用旧评分。

完成标准：`legacy_teacher_score`与当前行为一致；质量修正版在缺失六部件时不再让轮廓自动占100%，两套评分均可解释且可重复。

### 阶段5：Pair Builder与Parquet

任务：

- [ ] 实现确定性无序pair枚举。
- [ ] 实现block checkpoint和断点续跑。
- [ ] 实现缓存特征批量评分。
- [ ] 实现图片级数据切分和分层pair采样。
- [ ] 实现教师分数分桶统计和困难样本标记。
- [ ] 实现Parquet分区写入和数据集manifest。
- [ ] 实现pair数量、重复、空值和分数分布审计。

完成标准：100张图片可完整生成4,950个pair，重复运行不会重复计算完成的block。

### 阶段6：一致性、性能与全量运行

任务：

- [ ] 对完整组件样本执行新旧pipeline评分一致性测试。
- [ ] 对缺失组件样本进行人工抽样审查。
- [ ] 用100张图片进行端到端压测。
- [ ] 根据压测估算1000至3000张图片的时间、空间和显存。
- [ ] 完成全量预处理和教师pair生成。
- [ ] 固化`teacher-v1`模型、配置和数据版本。
- [ ] 建立200至500张查询图对完整图库的检索基准，用于后续Recall@200和NDCG@10评估。

完成标准：全量任务可恢复、可审计，输出可直接被向量蒸馏训练读取。

## 12. 测试计划

### 单元测试

- 相同内容不同路径得到相同内容hash。
- 模型或预处理参数变化导致正确的缓存失效。
- 修改评分权重不导致embedding缓存失效。
- 原子写入失败不会留下可见的完整Artifact。
- 文件名包含多个部件名称时不发生错误映射。
- 六个`parts`键始终存在，缺失部件保留状态而不是被省略。
- 特征输入hash与SAM最终cutout一致。
- 七组件完整时权重和为1。
- 任意缺失组合下coverage计算正确。
- A/B交换后pair_id和分数保持不变。
- 最终分始终位于`[0, 100]`。

### 集成测试

- 单图首次处理和二次缓存命中。
- 批处理中断后恢复。
- 低置信度、空mask和模型异常状态持久化。
- 100张图pair数量严格等于4,950。
- Parquet block合并后无重复pair。

### 回归测试

- 七项均有效时，新旧pipeline各单项评分误差不超过`0.1`分。
- `legacy_teacher_score`与当前pipeline最终分误差不超过`0.1`分。
- 缺失部件新评分与旧评分差异必须符合新策略预期。
- 固定模型、配置和输入时重复运行结果一致。

## 13. 验收指标

- 第二次预处理同一数据集时，重模型调用次数为0。
- 缓存命中率不低于99%，排除损坏或配置变化的Artifact。
- 所有部件缺失都有侧别、状态和原因。
- 缺少全部部件时，不允许轮廓分直接成为最终分。
- 1000张图生成499,500个唯一无序pair。
- Pair Builder中不调用YOLO、SAM、BiRefNet、CLIP或DINO。
- YOLO结构化结果、SAM最终mask/cutout和特征来源可双向追溯。
- train、validation和test之间不存在相同内容hash或车辆实体泄漏。
- 中断后能够从已完成图片或block继续。
- 数据集包含模型、配置、Schema和评分规则版本。
- 输出Parquet可被后续训练程序顺序读取和分区读取。

## 14. 风险与控制

| 风险 | 影响 | 控制措施 |
| --- | --- | --- |
| 初次SAM处理耗时高 | 1000张图可能需要数小时 | 单图永久缓存、批处理、先以100张压测 |
| GPU并发导致显存不稳定 | 任务中断或结果异常 | 单GPU推理进程，CPU侧并行I/O |
| 新旧算法抽取后产生漂移 | 教师标签不可比较 | 基线数据包和逐项评分回归测试 |
| 百万pair生成大量差异图 | 存储爆炸 | 默认只保存数值，差异图仅抽样 |
| 文件名推断部件错误 | 错误训练标签 | 使用结构化manifest，不以子串作为主映射 |
| 缺失策略参数主观 | 排序偏差 | 保留原始分、coverage和最终分，使用验证集校准 |
| 教师模型本身存在偏差 | 学生模型继承偏差 | 增加人工审核集和困难样本集 |
| 多版本缓存混用 | 数据不可复现 | 分层fingerprint和数据集manifest |

## 15. 资源与时间估算方法

当前实测为查询图加12张图库图，共13张输入：部件分割与SAM抠图约330秒，部件特征比对约114秒，总耗时约450秒。仅对YOLO+SAM阶段线性粗估约25.4秒/张，1000张首次处理约7.1小时；该值包含模型初始化和当前串行实现影响，正式估算必须在模型常驻、批处理模式下用100张图片重新测量。图片对阶段若仍调用`compare_paths()`重复提取A/B特征将完全不可接受，必须使用单图缓存。

正式全量运行前必须输出以下估算：

- 单图YOLO、SAM、BiRefNet和embedding平均耗时。
- GPU峰值显存和CPU峰值内存。
- 单个Artifact平均磁盘占用。
- 每10,000个pair的SSIM、轮廓和Parquet写入耗时。
- 1000、1415和3000张图的总时间与磁盘预测。

## 16. 后续接口

本项目完成后，向量蒸馏项目直接消费：

- 单图整车及六部件embedding，以及每个embedding对应的SAM产物hash。
- 每个组件的有效性mask和质量分。
- 教师单项分、旧版综合分、质量修正版综合分、coverage和训练样本权重。
- 训练集、验证集、测试集的图片级划分。

FAISS阶段不得把原始余弦分直接作为最终相似度。向量召回后仍需使用本项目输出的组件有效性、coverage和缺失规则进行精排或置信度修正。

## 17. 开工条件

开始编码前确认以下事项：

- [ ] 目标项目作为独立仓库还是当前仓库子目录管理。
- [ ] 模型权重由新项目独立保存，还是通过配置引用现有模型目录。
- [ ] 教师数据的正式输入图库位置和车辆实体ID来源。
- [ ] 是否具备用于校准缺失规则的人工审核样本。
- [ ] `prior_score`初始值采用统计中位数还是固定业务值。
- [ ] 首期是否启用CLIP，还是只使用DINO、SSIM和edge建立基线。
