<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { ZoomIn, X } from 'lucide-vue-next'
import { RouterLink, useRoute } from 'vue-router'
import { getCandidateDetail } from '@/api/client'
import type { CandidateDetailResponse, ScoreTag } from '@/api/types'
import { loadLastQueryName } from '@/state/runCache'

const route = useRoute()
const runId = computed(() => String(route.params.runId || 'latest'))
const candidateId = computed(() => String(route.params.candidateId || ''))

const data = ref<CandidateDetailResponse | null>(null)
const loading = ref(false)
const errorMessage = ref<string | null>(null)
const queryName = computed(() => {
  const q = route.query.q
  if (typeof q === 'string' && q.trim()) return q.trim()
  const last = loadLastQueryName()
  if (last) return last
  return ''
})
const queryThumb = computed(() => data.value?.query.image_url || '')
const filteredEvidence = computed(() => {
  const list = data.value?.parts.evidence ?? []
  return list
})

const zoomOpen = ref(false)
const zoomUrl = ref('')
const zoomAlt = ref('')

function openZoom(url: string, alt: string) {
  if (!url) return
  zoomUrl.value = url
  zoomAlt.value = alt
  zoomOpen.value = true
  document.documentElement.style.overflow = 'hidden'
}

function closeZoom() {
  zoomOpen.value = false
  zoomUrl.value = ''
  zoomAlt.value = ''
  document.documentElement.style.overflow = ''
}

function onKeydown(e: KeyboardEvent) {
  if (e.key === 'Escape') closeZoom()
}

onMounted(async () => {
  window.addEventListener('keydown', onKeydown)
})

watch([runId, candidateId], async ([rid, cid]) => {
  if (!cid) {
    data.value = null
    return
  }
  loading.value = true
  errorMessage.value = null
  data.value = null
  try {
    data.value = await getCandidateDetail(rid, cid)
  } catch (e) {
    data.value = null
    errorMessage.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}, { immediate: true })

onUnmounted(() => {
  window.removeEventListener('keydown', onKeydown)
  document.documentElement.style.overflow = ''
})

function tagClass(tag: ScoreTag) {
  if (tag === '高相似') return 'tag good'
  if (tag === '中高相似') return 'tag'
  if (tag === '局部相似') return 'tag warn'
  return 'tag danger'
}

function partBadgeClass(tag: ScoreTag) {
  if (tag === '高相似') return 'part-badge'
  if (tag === '中高相似') return 'part-badge mid'
  return 'part-badge warn'
}

function cropClass(partName: string) {
  if (partName === '后视镜') return 'tile-crop mirror'
  if (partName === '车灯') return 'tile-crop light'
  if (partName === '前保险杠') return 'tile-crop grille'
  return 'tile-crop glass'
}
</script>

<template>
  <main class="shell">
    <section class="app-grid" aria-label="相似度查重工作台">
      <aside class="panel sidebar">
        <h1>相似度查重</h1>

        <div>
          <div class="field-title">上传待对比图片</div>
          <div class="upload-box" tabindex="0" aria-label="上传待对比图片">
            <div v-if="queryThumb" class="upload-preview">
              <img :src="queryThumb" alt="上传图片缩略图" />
              <div class="upload-filename">{{ queryName || '上传比对图片' }}</div>
            </div>
            <div v-else>
              <div class="upload-icon"></div>
              <div class="upload-text">上传</div>
            </div>
          </div>
        </div>

        <div class="control">
          <div class="select-like">车型选择 <span aria-hidden="true"></span></div>
        </div>

        <div class="topk-row">
          <div>
            <div class="label topk-label">返回结果Top-N</div>
          </div>
          <div class="number-like">10</div>
        </div>

        <button class="primary" type="button">开始比对</button>
      </aside>

      <section class="panel result-panel">
        <div class="result-head">
          <h2>分析结果</h2>
        </div>
        <div class="detail-stage">
          <div v-if="loading" class="loading-wrap" role="status" aria-live="polite">
            <div class="loading-card">
              <div class="loading-text">正在加载详情...</div>
            </div>
          </div>
          <div v-else-if="errorMessage" class="loading-wrap" role="status">
            <div class="loading-card">
              <div class="loading-text">{{ errorMessage }}</div>
            </div>
          </div>
          <div v-else-if="data" class="detail-canvas">
            <header class="topbar">
              <h1>A车 vs B车 相似度分析详情</h1>
              <RouterLink class="back" :to="{ name: 'workbench', query: { restore: '1' } }" aria-label="返回概览">返回概览</RouterLink>
            </header>

            <nav class="anchor-nav" aria-label="详情页快捷导航">
              <a href="#overview">概览</a>
              <a href="#contour">轮廓</a>
              <a href="#evidence">部件详情</a>
            </nav>

            <section id="overview" class="panel summary" aria-label="相似度总结">
              <article class="score-card">
                <div class="label">总相似度</div>
                <div class="big-score">{{ (data?.summary.final_score ?? 78.8).toFixed(1).replace(/\.0$/, '') }}</div>
                <div :class="tagClass(data?.summary.tag ?? '中高相似')">{{ data?.summary.tag ?? '中高相似' }}</div>
              </article>

              <article class="summary-points">
                <div class="summary-title">相似度分析</div>
                <ul class="point-list">
                  <li v-for="(p, idx) in (data?.summary.points ?? [])" :key="idx">{{ p }}</li>
                  <li v-if="(data?.summary.points?.length ?? 0) === 0">最终评分 78.8 分，判定为中高相似，适合进入人工复核。</li>
                  <li v-if="(data?.summary.points?.length ?? 0) === 0">整车轮廓相似度 90.6，车身基础比例和正面姿态高度接近。</li>
                  <li v-if="(data?.summary.points?.length ?? 0) === 0">共对比 6 个有效部件，后视镜相似度最高，前挡风玻璃差异较明显。</li>
                </ul>
              </article>

              <article class="weight-card">
                <div class="weight-row">
                  <div class="weight-head">
                    <span class="weight-name">轮廓分 · 权重 40%</span>
                    <span class="weight-score">{{ (data?.summary.weights.contour.score ?? 90.6).toFixed(1).replace(/\.0$/, '') }}</span>
                  </div>
                  <div class="bar" :style="{ '--value': `${data?.summary.weights.contour.score ?? 90.6}%` }"><span></span></div>
                </div>
                <div class="weight-row">
                  <div class="weight-head">
                    <span class="weight-name">部件分 · 权重 60%</span>
                    <span class="weight-score">{{ (data?.summary.weights.parts.score ?? 78.8).toFixed(1).replace(/\.0$/, '') }}</span>
                  </div>
                  <div class="bar" :style="{ '--value': `${data?.summary.weights.parts.score ?? 78.8}%` }"><span></span></div>
                </div>
              </article>
            </section>

            <section id="contour" class="panel section" aria-label="整体轮廓相似对比">
              <div class="section-head">
                <h2>整体轮廓相似对比</h2>
              </div>

              <div class="contour-layout">
                <aside class="contour-card">
                  <div class="label">轮廓分（40%）</div>
                  <div class="contour-score">{{ (data?.contour.score ?? 90.6).toFixed(1).replace(/\.0$/, '') }}</div>
                  <div :class="tagClass(data?.contour.tag ?? '高相似')">{{ data?.contour.tag ?? '高相似' }}</div>
                  <div class="legend" aria-label="轮廓差异图图例">
                    <div class="legend-item"><span class="swatch yellow"></span>黄色：两车高度重合区域</div>
                    <div class="legend-item"><span class="swatch red"></span>红色：A 图独有形状区域</div>
                    <div class="legend-item"><span class="swatch green"></span>绿色：B 图独有形状区域</div>
                  </div>
                  <p class="contour-copy">{{ data?.contour.conclusion ?? '结论：两车正面主体轮廓接近，差异主要集中在车头下沿、车顶线和局部外扩区域。' }}</p>
                </aside>

                <div class="image-grid">
                  <figure class="image-card large">
                    <div class="diff-map" role="img" aria-label="整车轮廓差异图">
                      <img :src="data?.contour.diff_image_url || '/prototype-assets/edge_diff.jpg'" alt="整车轮廓差异图" />
                    </div>
                    <figcaption class="image-caption">整车轮廓差异图</figcaption>
                  </figure>
                  <div class="contour-footnote">红色和绿色只用于表达差异，不作为普通高亮色使用</div>
                </div>
              </div>
            </section>

            <section id="parts" class="panel section" aria-label="部件识别与对齐标注">
              <div class="section-head">
                <h2>部件识别与对齐标注</h2>
                <div class="section-note">先给部件总分，再给每个部件的可追溯证据</div>
              </div>

              <div class="parts-summary">
                <article class="part-score-card">
                  <div class="label">部件分（60%）</div>
                  <div class="big-score">{{ (data?.parts.score ?? 78.8).toFixed(1).replace(/\.0$/, '') }}</div>
                  <div :class="tagClass(data?.parts.tag ?? '中高相似')">{{ data?.parts.tag ?? '中高相似' }}</div>
                </article>
                <div class="parts-image-pair">
                  <figure class="image-card">
                    <img :src="data?.parts.a_annotation_url ?? '/prototype-assets/A_annotation.jpg'" alt="A 图部件识别标注" />
                    <button
                      class="zoom-btn"
                      type="button"
                      aria-label="放大图片"
                      @click="openZoom(data?.parts.a_annotation_url ?? '/prototype-assets/A_annotation.jpg', 'A 图部件识别标注')"
                    >
                      <ZoomIn class="zoom-icon" />
                    </button>
                    <figcaption class="image-caption">A 图部件识别</figcaption>
                  </figure>
                  <figure class="image-card">
                    <img :src="data?.parts.b_annotation_url ?? '/prototype-assets/B_annotation.jpg'" alt="B 图部件识别标注" />
                    <button
                      class="zoom-btn"
                      type="button"
                      aria-label="放大图片"
                      @click="openZoom(data?.parts.b_annotation_url ?? '/prototype-assets/B_annotation.jpg', 'B 图部件识别标注')"
                    >
                      <ZoomIn class="zoom-icon" />
                    </button>
                    <figcaption class="image-caption">B 图部件识别</figcaption>
                  </figure>
                </div>
              </div>

              <div id="evidence" class="part-grid">
                <article v-for="p in filteredEvidence" :key="p.part_name" class="part-card">
                  <div class="part-head">
                    <div class="part-title">{{ p.part_name }}相似度评分：{{ p.fused.toFixed(1).replace(/\.0$/, '') }}</div>
                    <div :class="partBadgeClass(p.tag)">{{ p.tag }}</div>
                  </div>
                  <div class="tiles">
                    <div class="part-tile">
                      <img :class="cropClass(p.part_name)" :src="p.tiles.a_color" :alt="`A 原图 ${p.part_name}`" />
                      <div class="part-caption">A 原图</div>
                    </div>
                    <div class="part-tile">
                      <img :class="cropClass(p.part_name)" :src="p.tiles.b_color" :alt="`B 原图 ${p.part_name}`" />
                      <div class="part-caption">B 原图</div>
                    </div>
                    <div class="part-tile">
                      <img :class="cropClass(p.part_name)" :src="p.tiles.a_gray" :alt="`A 灰度图 ${p.part_name}`" />
                      <div class="part-caption">A 灰度图</div>
                    </div>
                    <div class="part-tile">
                      <img :class="cropClass(p.part_name)" :src="p.tiles.b_gray" :alt="`B 灰度图 ${p.part_name}`" />
                      <div class="part-caption">B 灰度图</div>
                    </div>
                    <div class="part-tile">
                      <img v-if="p.tiles.diff" :src="p.tiles.diff" :alt="`内部差异 ${p.part_name}`" />
                      <div v-else class="tile-diff"></div>
                      <div class="part-caption">内部差异</div>
                    </div>
                  </div>
                  <div class="metrics">
                    <div class="metric"><span>CLIP</span><b>{{ p.metrics?.clip?.toFixed(1) ?? '—' }}</b></div>
                    <div class="metric"><span>DINO</span><b>{{ p.metrics?.dino?.toFixed(1) ?? '—' }}</b></div>
                    <div class="metric"><span>SSIM</span><b>{{ p.metrics?.ssim?.toFixed(1) ?? '—' }}</b></div>
                    <div class="metric"><span>EDGE</span><b>{{ p.metrics?.edge?.toFixed(1) ?? '—' }}</b></div>
                  </div>
                </article>
              </div>
            </section>
          </div>
        </div>
      </section>
    </section>

    <Teleport to="body">
      <div v-if="zoomOpen" class="zoom-overlay" role="dialog" aria-modal="true" @click.self="closeZoom">
        <button class="zoom-close" type="button" aria-label="关闭" @click="closeZoom">
          <X class="zoom-icon" />
        </button>
        <img class="zoom-image" :src="zoomUrl" :alt="zoomAlt || '预览图片'" />
      </div>
    </Teleport>
  </main>
</template>

<style scoped src="../styles/detail.css"></style>
