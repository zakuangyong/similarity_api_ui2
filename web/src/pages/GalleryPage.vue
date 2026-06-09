<script setup lang="ts">
import { computed, nextTick, onMounted, ref } from 'vue'
import { ChevronLeft, ChevronRight } from 'lucide-vue-next'
import { RouterLink } from 'vue-router'
import { getGallery } from '@/api/client'
import type { GalleryItem } from '@/api/types'

const loading = ref(true)
const items = ref<GalleryItem[]>([])
const currentPage = ref(1)
const pageSize = ref(18)
const galleryShell = ref<HTMLElement | null>(null)

const viewLabel = '正面'

onMounted(async () => {
  loading.value = true
  try {
    const resp = await getGallery({ view: 'front', vehicleType: 'SUV' })
    items.value = resp.items
    currentPage.value = 1
  } finally {
    loading.value = false
  }
})

const subText = computed(() => `${items.value.length} 张可检索图片 · ${viewLabel}车辆图库`)
const totalPages = computed(() => Math.max(1, Math.ceil(items.value.length / pageSize.value)))
const pagedItems = computed(() => {
  const start = (currentPage.value - 1) * pageSize.value
  return items.value.slice(start, start + pageSize.value)
})
const visiblePages = computed(() => {
  const total = totalPages.value
  const current = currentPage.value
  const start = Math.max(1, Math.min(current - 2, total - 4))
  const end = Math.min(total, start + 4)
  return Array.from({ length: end - start + 1 }, (_, index) => start + index)
})
const pageRangeText = computed(() => {
  if (!items.value.length) return '0 / 0'
  const start = (currentPage.value - 1) * pageSize.value + 1
  const end = Math.min(currentPage.value * pageSize.value, items.value.length)
  return `${start}-${end} / ${items.value.length}`
})

async function goToPage(page: number) {
  const target = Math.min(totalPages.value, Math.max(1, page))
  if (target === currentPage.value) return
  currentPage.value = target
  await nextTick()
  galleryShell.value?.scrollIntoView({ behavior: 'smooth', block: 'start' })
}

function changePageSize() {
  currentPage.value = 1
}
</script>

<template>
  <main class="page">
    <section ref="galleryShell" class="gallery-shell" aria-label="图库管理">
      <div class="gallery-head">
        <div>
          <h1>图库管理</h1>
          <p class="sub">{{ subText }}</p>
        </div>
        <RouterLink class="back" to="/workbench">返回分析工作台</RouterLink>
      </div>

      <div class="gallery-grid" :aria-busy="loading ? 'true' : 'false'">
        <figure v-for="it in pagedItems" :key="it.id" class="gallery-card">
          <img :src="it.url" :alt="it.name" />
          <figcaption>{{ it.name }}</figcaption>
        </figure>
      </div>

      <nav v-if="items.length > 0" class="pagination" aria-label="图库分页">
        <div class="page-summary">{{ pageRangeText }}</div>

        <div class="page-controls">
          <button
            class="page-button icon-button"
            type="button"
            title="上一页"
            aria-label="上一页"
            :disabled="currentPage === 1"
            @click="goToPage(currentPage - 1)"
          >
            <ChevronLeft :size="18" aria-hidden="true" />
          </button>

          <button
            v-for="pageNumber in visiblePages"
            :key="pageNumber"
            class="page-button"
            :class="{ active: pageNumber === currentPage }"
            type="button"
            :aria-current="pageNumber === currentPage ? 'page' : undefined"
            @click="goToPage(pageNumber)"
          >
            {{ pageNumber }}
          </button>

          <button
            class="page-button icon-button"
            type="button"
            title="下一页"
            aria-label="下一页"
            :disabled="currentPage === totalPages"
            @click="goToPage(currentPage + 1)"
          >
            <ChevronRight :size="18" aria-hidden="true" />
          </button>
        </div>

        <label class="page-size">
          <span>每页</span>
          <select v-model.number="pageSize" @change="changePageSize">
            <option :value="12">12</option>
            <option :value="18">18</option>
            <option :value="24">24</option>
            <option :value="36">36</option>
          </select>
          <span>张</span>
        </label>
      </nav>
    </section>
  </main>
</template>

<style scoped src="../styles/gallery.css"></style>
