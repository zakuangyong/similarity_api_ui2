<script setup lang="ts">
import { computed, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { compare } from '@/api/client'
import type { CandidateCard, CompareResponse } from '@/api/types'
import { loadLastCompare, saveLastCompare } from '@/state/runCache'

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

const last = loadLastCompare()
const run = ref<CompareResponse | null>(last)
const results = computed<CandidateCard[]>(() => run.value?.results ?? [])

const loading = ref(false)
const errorMessage = ref<string | null>(null)

const queryPreview = computed(() => {
  if (queryObjectUrl.value) return queryObjectUrl.value
  if (run.value) return run.value.query_staged_path
  return ''
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
}

async function startCompare() {
  loading.value = true
  errorMessage.value = null
  try {
    const resp = await compare({
      queryImage: queryFile.value ?? undefined,
      view: view.value,
      vehicleType: vehicleType.value,
      topk: topk.value,
    })
    run.value = resp
    saveLastCompare(resp)
  } catch (e) {
    errorMessage.value = e instanceof Error ? e.message : String(e)
  } finally {
    loading.value = false
  }
}

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
            <div>
              <div class="upload-icon"></div>
              <div class="upload-text">上传</div>
            </div>
          </div>
          <input ref="fileInputEl" class="sr-only" type="file" accept="image/*" @change="onFileChange" />
        </div>

        <div class="topk-row">
          <div>
            <div class="label">相似度Top-K</div>
            <div class="hint">返回最相似的前 N 张候选图。</div>
          </div>
          <div class="number-like">{{ topk }}</div>
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
          <article class="query-card">
            <img v-if="queryPreview" :src="queryPreview" alt="上传待比对车辆" />
            <div v-else class="query-empty" aria-label="未上传图片"></div>
            <div class="query-meta">
              <div class="query-name">上传比对图片</div>
              <div class="query-note">当前样例：星耀7</div>
            </div>
          </article>

          <div class="candidate-grid" aria-label="Top-K 相似候选">
            <RouterLink
              v-for="(item, idx) in results"
              :key="item.candidate_id"
              class="candidate-card"
              :to="{ name: 'detail', params: { runId: run?.run_id ?? 'latest', candidateId: item.candidate_id } }"
            >
              <span class="badge rank">Top {{ idx + 1 }}</span>
              <span class="badge" :class="scoreClass(item.final_score)">{{ item.final_score.toFixed(1).replace(/\.0$/, '') }} · {{ scoreTagText(item.final_score) }}</span>
              <img :src="item.candidate_path" :alt="`候选车辆：${item.candidate_name}`" />
              <div class="card-foot">
                <span class="car-name">{{ item.candidate_name }}</span>
                <span class="detail-link">查看详情</span>
              </div>
            </RouterLink>
          </div>
        </div>
      </section>
    </section>
  </main>
</template>

<style scoped src="../styles/workbench.css"></style>
