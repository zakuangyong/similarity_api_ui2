import { createRouter, createWebHistory } from 'vue-router'
import type { RouteRecordRaw } from 'vue-router'
import WorkbenchPage from '@/pages/WorkbenchPage.vue'
import DetailPage from '@/pages/DetailPage.vue'
import GalleryPage from '@/pages/GalleryPage.vue'

// 定义路由配置
const routes: RouteRecordRaw[] = [
  {
    path: '/',
    redirect: { name: 'workbench' },
  },
  {
    path: '/workbench',
    name: 'workbench',
    component: WorkbenchPage,
  },
  {
    path: '/detail/:runId/:candidateId',
    name: 'detail',
    component: DetailPage,
  },
  {
    path: '/gallery',
    name: 'gallery',
    component: GalleryPage,
  },
]

// 创建路由实例
const router = createRouter({
  history: createWebHistory(),
  routes,
})

export default router
