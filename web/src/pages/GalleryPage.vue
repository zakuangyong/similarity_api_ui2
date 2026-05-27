<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'
import { RouterLink } from 'vue-router'
import { getGallery } from '@/api/client'
import type { GalleryItem } from '@/api/types'

const loading = ref(true)
const items = ref<GalleryItem[]>([])

const viewLabel = '正面'

onMounted(async () => {
  loading.value = true
  try {
    const resp = await getGallery({ view: 'front', vehicleType: 'SUV' })
    items.value = resp.items
  } finally {
    loading.value = false
  }
})

const subText = computed(() => `${items.value.length} 张可检索图片 · ${viewLabel}车辆图库`)
</script>

<template>
  <main class="page">
    <section class="gallery-shell" aria-label="图库管理">
      <div class="gallery-head">
        <div>
          <h1>图库管理</h1>
          <p class="sub">{{ subText }}</p>
        </div>
        <RouterLink class="back" to="/workbench">返回分析工作台</RouterLink>
      </div>

      <div class="gallery-grid" :aria-busy="loading ? 'true' : 'false'">
        <figure v-for="it in items" :key="it.id" class="gallery-card">
          <img :src="it.url" :alt="it.name" />
          <figcaption>{{ it.name }}</figcaption>
        </figure>
      </div>
    </section>
  </main>
</template>

<style scoped src="../styles/gallery.css"></style>
