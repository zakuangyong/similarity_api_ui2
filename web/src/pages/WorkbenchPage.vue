<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import { RouterLink, useRoute } from 'vue-router'
import { compare } from '@/api/client'
import type { CandidateCard, CompareResponse } from '@/api/types'
import { loadLastCompare, saveLastCompare, saveLastQueryName } from '@/state/runCache'

const viewOptions = [
  { label: '正脸视图', value: 'front' },
  { label: '侧面车身视图', value: 'side' },
  { label: '尾部视图', value: 'rear' },
]

const vehicleOptions = ['SUV', '轿车', '轿跑', '越野', 'MPV', '皮卡']

const view = ref(viewOptions[0]!.value)
const vehicleType = ref(vehicleOptions[0]!)
const topk = ref(10)

const fileInputEl = ref<HTMLInputElement | null>(null)
const queryFile = ref<File | null>(null)
const queryObjectUrl = ref<string | null>(null)

const route = useRoute()
const run = ref<CompareResponse | null>(null)
const results = computed<CandidateCard[]>(() => (run.value?.results ?? []).slice(0, 10))

const loading = ref(false)
const errorMessage = ref<string | null>(null)
const progress = ref(0)
let progressTimer: number | null = null

const uploadedVehicleName = computed(() => {
  const file = queryFile.value
  if (!file) return ''
  return file.name.replace(/\.[^/.]+$/, '')
})

const queryPreview = computed(() => {
  return run.value?.query_staged_path ?? ''
})

function openFilePicker() {
  fileInputEl.value?.click()
}

function onFileChange(e: Event) {
  const el = e.target as HTMLInputElement
  const file = el.files?.[0]
  if (!file) return

  queryFile.value = file
  if (queryObjectUrl.value) URL.revokeObjectURL(queryObjectUrl.value)
  queryObjectUrl.value = URL.createObjectURL(file)
  saveLastQueryName(file.name.replace(/\.[^/.]+$/, ''))
  run.value = null
  progress.value = 0
  el.value = ''
}

function stopProgress() {
  if (progressTimer !== null) {
    window.clearInterval(progressTimer)
    progressTimer = null
  }
}

function startProgress() {
  stopProgress()
  progress.value = 0
  let ticks = 0
  progressTimer = window.setInterval(() => {
    ticks += 1
    if (progress.value < 90) {
      progress.value = Math.min(90, progress.value + 2)
      return
    }
    if (progress.value < 95 && ticks % 5 === 0) {
      progress.value += 1
    }
  }, 160)
}

async function startCompare() {
  loading.value = true
  errorMessage.value = null
  startProgress()
  try {
    const k = Number(topk.value)
    const k1 = Number.isFinite(k) ? Math.floor(k) : 10
    const k2 = Math.min(10, Math.max(1, k1))
    topk.value = k2
    const resp = await compare({
      queryImage: queryFile.value ?? undefined,
      view: view.value,
      vehicleType: vehicleType.value,
      topk: k2,
    })
    progress.value = 100
    run.value = resp
    saveLastCompare(resp)
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
    stopProgress()
  }
}

onBeforeUnmount(() => {
  stopProgress()
  if (queryObjectUrl.value) URL.revokeObjectURL(queryObjectUrl.value)
})

onMounted(() => {
  if (route.query.restore === '1') {
    run.value = loadLastCompare()
  }
})

function scoreClass(score: number) {
  if (score >= 85) return 'score'
  if (score >= 72) return 'score mid'
  return 'score warn'
}

function scoreTagText(score: number) {
  if (score >= 85) return '高相似'
  if (score >= 72) return '中高相似'
  return '局部相似'
}
</script>

<template>
  <main class="shell">
    <section class="app-grid" id="workbenchGrid" aria-label="相似度查重工作台">
      <aside class="panel sidebar">
        <h1>相似度查重</h1>

        <div class="select-row">
          <label class="select-label" for="viewSelect">视角选择</label>
          <select class="view-select" id="viewSelect" v-model="view" aria-label="视角选择">
            <option v-for="opt in viewOptions" :key="opt.value" :value="opt.value">{{ opt.label }}</option>
          </select>
        </div>

        <div class="select-row">
          <label class="select-label" for="vehicleSelect">车型选择</label>
          <select class="vehicle-select" id="vehicleSelect" v-model="vehicleType" aria-label="车型选择">
            <option v-for="opt in vehicleOptions" :key="opt" :value="opt">{{ opt }}</option>
          </select>
        </div>

        <div>
          <div class="field-title">上传待对比图片</div>
          <div class="upload-box" role="button" tabindex="0" aria-label="上传待对比图片" @click="openFilePicker" @keydown.enter="openFilePicker">
            <div v-if="queryObjectUrl" class="upload-preview">
              <img :src="queryObjectUrl" alt="上传图片缩略图" />
              <div class="upload-filename" :title="uploadedVehicleName">{{ uploadedVehicleName }}</div>
            </div>
            <div v-else>
              <div class="upload-icon" aria-hidden="true">
                <svg viewBox="0 0 64 48" fill="none" xmlns="http://www.w3.org/2000/svg">
                  <rect x="3.5" y="3.5" width="57" height="41" rx="8" stroke="rgba(20,121,184,0.9)" stroke-width="5" />
                  <circle cx="20" cy="18" r="5" fill="rgba(20,121,184,0.9)" />
                  <path
                    d="M11 38L25.5 25.5C27.1 24.1 29.5 24.2 31 25.8L37.5 33.3C38.8 34.8 41 35.1 42.7 34.1L53 28V41H11V38Z"
                    fill="rgba(20,121,184,0.9)"
                  />
                </svg>
              </div>
              <div class="upload-text">上传</div>
            </div>
          </div>
          <input ref="fileInputEl" class="sr-only" type="file" accept="image/*" @change="onFileChange" />
        </div>

        <div class="topk-row">
          <div>
              <div class="label topk-label">返回结果Top-N</div>
            <div class="hint">返回最相似的前 N 张候选图。</div>
          </div>
            <input v-model.number="topk" class="topk-input" type="number" inputmode="numeric" min="1" max="10" step="1" :disabled="loading" />
        </div>

        <button class="primary" type="button" :disabled="loading" @click="startCompare">开始比对</button>
        <div v-if="errorMessage" class="error" role="status">{{ errorMessage }}</div>
        <RouterLink class="gallery-entry" to="/gallery">图库管理</RouterLink>
      </aside>

      <section class="panel result-panel">
        <div class="result-head">
          <h2>分析结果</h2>
        </div>
        <div class="result-stage">
          <template v-if="run">
            <article class="query-card">
              <img v-if="queryPreview" :src="queryPreview" alt="上传待比对车辆" />
              <div class="query-meta">
                <div class="query-name">上传比对图片</div>
                <div class="query-note" :title="uploadedVehicleName || run.query_name">当前车型：{{ uploadedVehicleName || run.query_name }}</div>
              </div>
            </article>

            <div class="candidate-grid" aria-label="Top-K 相似候选">
              <RouterLink
                v-for="(item, idx) in results"
                :key="item.candidate_id"
                class="candidate-card"
                :to="{ name: 'detail', params: { runId: run.run_id, candidateId: item.candidate_id }, query: { q: uploadedVehicleName || run.query_name } }"
              >
                <span class="badge rank">Top {{ idx + 1 }}</span>
                <span class="badge" :class="scoreClass(item.final_score)">{{ item.final_score.toFixed(1).replace(/\.0$/, '') }} · {{ scoreTagText(item.final_score) }}</span>
                <img :src="item.candidate_path" :alt="`候选车辆：${item.candidate_name}`" />
                <div class="card-foot">
                  <span class="car-name" :title="item.candidate_name">{{ item.candidate_name }}</span>
                  <span class="detail-link">查看详情</span>
                </div>
              </RouterLink>
            </div>
          </template>
          <template v-else-if="loading">
            <div class="loading-wrap" role="status" aria-live="polite">
              <div class="loading-card">
                <div class="progress-ring" :style="{ '--p': String(progress) }">
                  <div class="progress-text">{{ progress }}%</div>
                </div>
                <div class="loading-text">正在进行相似度查重，请稍后...</div>
              </div>
            </div>
          </template>
        </div>
      </section>
    </section>
  </main>
</template>

<style scoped src="../styles/workbench.css"></style>
