<script setup lang="ts">
import { computed, onBeforeUnmount, onMounted, ref } from 'vue'
import type { UploadProps, UploadRawFile, UploadUserFile } from 'element-plus'
import { RouterLink, useRoute, useRouter } from 'vue-router'
import { compare } from '@/api/client'
import type { CandidateCard, CompareResponse } from '@/api/types'
import GenerateWorkspaceLayout from '@/components/Layout.vue'
import { loadLastCompare, saveLastCompare, saveLastQueryName } from '@/state/runCache'

const viewOptions = [
  { label: '正脸视图', value: 'front' },
  { label: '侧面车身视图', value: 'side' },
  { label: '尾部视图', value: 'rear' },
]

const vehicleOptions = ['轿车', 'SUV', '轿跑', '越野', 'MPV', '皮卡']

const view = ref(viewOptions[0]!.value)
const vehicleType = ref('轿车')
const topk = ref(10)

const queryFile = ref<File | null>(null)
const queryObjectUrl = ref<string | null>(null)
const uploadFileList = ref<UploadUserFile[]>([])

const route = useRoute()
const router = useRouter()
const run = ref<CompareResponse | null>(null)
const results = computed<CandidateCard[]>(() => (run.value?.results ?? []).slice(0, 20))

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

const resultSummary = computed(() => {
  if (loading.value) return '正在执行相似度查重'
  if (run.value) return `已返回 ${results.value.length} 个候选结果`
  return '上传图片并设置参数后开始比对'
})

function clearQueryObjectUrl() {
  if (queryObjectUrl.value) {
    URL.revokeObjectURL(queryObjectUrl.value)
    queryObjectUrl.value = null
  }
}

function restoreQueryUploadList() {
  uploadFileList.value =
    queryFile.value && queryObjectUrl.value ? [{ name: queryFile.value.name, url: queryObjectUrl.value }] : []
}

function syncQueryFile(file: File) {
  queryFile.value = file
  clearQueryObjectUrl()
  const nextObjectUrl = URL.createObjectURL(file)
  queryObjectUrl.value = nextObjectUrl
  uploadFileList.value = [{ name: file.name, url: nextObjectUrl }]
  saveLastQueryName(file.name.replace(/\.[^/.]+$/, ''))
  run.value = null
  progress.value = 0
}

function handleSelectedQueryFile(rawFile: File) {
  const isValidType = ['image/jpeg', 'image/png'].includes(rawFile.type)
  if (!isValidType) {
    restoreQueryUploadList()
    errorMessage.value = '仅支持 JPG、JPEG、PNG 格式图片'
    return
  }

  const isValidSize = rawFile.size / 1024 / 1024 <= 10
  if (!isValidSize) {
    restoreQueryUploadList()
    errorMessage.value = '图片大小不能超过 10MB'
    return
  }

  errorMessage.value = null
  syncQueryFile(rawFile)
}

const handleQueryUploadChange: UploadProps['onChange'] = (uploadFile) => {
  if (!uploadFile.raw) {
    restoreQueryUploadList()
    return
  }
  handleSelectedQueryFile(uploadFile.raw)
}

const handleUploadExceed: UploadProps['onExceed'] = (files) => {
  const nextFile = files[0] as UploadRawFile | undefined
  if (!nextFile) return
  handleSelectedQueryFile(nextFile)
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
    const k2 = Math.min(20, Math.max(1, k1))
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
  clearQueryObjectUrl()
})

onMounted(() => {
  const imageUrlRaw = route.query.imageUrl
  const imageUrl =
    typeof imageUrlRaw === 'string' ? imageUrlRaw : Array.isArray(imageUrlRaw) ? imageUrlRaw[0] ?? null : null

  if (imageUrl) {
    void (async () => {
      try {
        const res = await fetch(imageUrl)
        if (!res.ok) {
          throw new Error(`图片获取失败（HTTP ${res.status}）`)
        }
        const blob = await res.blob()

        let fileName = ''
        try {
          const url = new URL(imageUrl)
          fileName = url.pathname.split('/').filter(Boolean).pop() ?? ''
        } catch {
          fileName = imageUrl.split('?')[0]?.split('#')[0]?.split('/').filter(Boolean).pop() ?? ''
        }

        try {
          fileName = decodeURIComponent(fileName)
        } catch {}

        let mime = blob.type || ''
        if (mime === 'image/jpg') mime = 'image/jpeg'

        const ext = fileName.split('.').pop()?.toLowerCase()
        if (!mime) {
          if (ext === 'png') mime = 'image/png'
          if (ext === 'jpg' || ext === 'jpeg') mime = 'image/jpeg'
        }

        if (!fileName) {
          fileName = mime === 'image/png' ? 'query.png' : 'query.jpg'
        } else if (!/\.(png|jpe?g)$/i.test(fileName)) {
          fileName = `${fileName}${mime === 'image/png' ? '.png' : '.jpg'}`
        }

        const file = new File([blob], fileName, { type: mime || undefined })
        handleSelectedQueryFile(file)
      } catch (e) {
        queryFile.value = null
        clearQueryObjectUrl()
        uploadFileList.value = []
        run.value = null
        progress.value = 0
        errorMessage.value = e instanceof Error ? e.message : String(e)
      }
    })()
    return
  }

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

function goGallery() {
  void router.push('/gallery')
}
</script>

<template>
  <GenerateWorkspaceLayout panel-title="相似度查重">
    <template #panel>
      <el-form  label-position="left" label-width="210">
        <el-form-item  label="视角选择" label-width="210">
          <el-select v-model="view" placeholder="请选择视角" :disabled="loading">
            <el-option v-for="opt in viewOptions" :key="opt.value" :label="opt.label" :value="opt.value" />
          </el-select>
        </el-form-item>

        <el-form-item label="车型选择">
          <el-select v-model="vehicleType" placeholder="请选择车型" :disabled="loading">
            <el-option v-for="opt in vehicleOptions" :key="opt" :label="opt" :value="opt" />
          </el-select>
        </el-form-item>

        <el-form-item label-position="top" required label="上传待对比图片">
          <div class="upload-field">

            <el-upload
              v-model:file-list="uploadFileList"
              class="upload-box"
              action="#"
              :show-file-list="false"
              :auto-upload="false"
              :limit="1"
              accept=".jpg,.jpeg,.png,image/jpeg,image/png"
              :disabled="loading"
              :on-change="handleQueryUploadChange"
              :on-exceed="handleUploadExceed"
            >
              <div v-if="queryObjectUrl" class="upload-preview">
                <img :src="queryObjectUrl" alt="上传图片缩略图" />
              </div>
              <div v-else class="upload-empty">
                <div class="upload-icon" aria-hidden="true">
                  <svg viewBox="0 0 64 64" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <rect x="12" y="12" width="40" height="40" rx="8" fill="rgba(34, 137, 255, 0.16)" />
                    <path d="M24 40L32 32L37 37L44 30" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                    <path d="M40 24H29C26.7909 24 25 25.7909 25 28V39" stroke="currentColor" stroke-width="3" stroke-linecap="round" />
                    <path d="M35 20H44V29" stroke="currentColor" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" />
                  </svg>
                </div>
                <div class="upload-text">上传</div>
                <div class="upload-hint">支持 JPG、PNG 格式，最大文件大小为 10MB</div>
              </div>
            </el-upload>
          </div>
        </el-form-item>

        <el-form-item label="返回结果Top-N">
          <el-input-number v-model="topk" class="topk-input" :min="1" :max="20" :step="1" :controls="false" :disabled="loading" />
        </el-form-item>
        <el-form-item v-if="errorMessage" class="workbench-form__item workbench-form__item--error" label-width="0">
          <div class="error" role="status">{{ errorMessage }}</div>
        </el-form-item>

        <el-form-item class="workbench-form__item workbench-form__actions-item" label-width="0">
          <div class="panel-actions">
            <el-button class="primary" type="primary" :disabled="loading" @click="startCompare">开始比对</el-button>
          </div>
        </el-form-item>

        <el-form-item class="workbench-form__item workbench-form__bottom-item" label-width="0">
          <el-button class="gallery-entry" :disabled="loading" @click="goGallery">查看图库</el-button>
        </el-form-item>
      </el-form>
    </template>

    <template #preview>
      <section class="result-panel">
        <div class="result-head">
          <div>
            <h2>分析结果</h2>
          </div>
        </div>
        <div class="result-stage">
          <template v-if="run">
            <div v-if="queryPreview" class="query-card">
              <img :src="queryPreview" alt="上传待比对车辆" />
            </div>
            <div class="candidate-grid" aria-label="Top-K 相似候选">
              <RouterLink
                v-for="(item, idx) in results"
                :key="item.candidate_id"
                class="candidate-card"
                :to="{ name: 'detail', params: { runId: run.run_id, candidateId: item.candidate_id }, query: { q: uploadedVehicleName || run.query_name } }"
              >
                <div class="card-media">
                  <span
                    :class="['badge', 'rank', { 'rank-1': idx === 0, 'rank-2': idx === 1, 'rank-3': idx === 2 }]"
                  >
                    Top {{ idx + 1 }}
                  </span>
                  <img :src="item.candidate_path" :alt="`候选车辆：${item.candidate_name}`" />
                </div>
                <div class="card-foot">
                  <span class="car-name" :title="item.candidate_name">{{ item.candidate_name }}</span>
                  <span class="badge footer-score" :class="scoreClass(item.final_score)">{{ item.final_score.toFixed(1).replace(/\.0$/, '') }} · {{ scoreTagText(item.final_score) }}</span>
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
          <template v-else>
            <div class="empty-wrap" role="status" aria-live="polite">
              <div class="empty-card">
                <div class="empty-title">等待开始比对</div>
                <div class="empty-text">请先在左侧选择视角、车型并上传待比对图片，再点击“开始比对”。</div>
              </div>
            </div>
          </template>
        </div>
      </section>
    </template>
  </GenerateWorkspaceLayout>
</template>

<style scoped src="../styles/workbench.css"></style>
