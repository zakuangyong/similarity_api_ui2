import type { CandidateDetailResponse, CompareResponse, GalleryResponse } from './types'
import { mockCompare, mockGetCandidateDetail, mockGetGallery } from './mock'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''
const USE_MOCK = (import.meta.env.VITE_USE_MOCK as string | undefined) === 'true'

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init)
  if (!res.ok) {
    const raw = await res.text().catch(() => '')
    const html = raw.trim().toLowerCase()
    if (html.startsWith('<!doctype html') || html.startsWith('<html')) {
      if (html.includes('504 gateway time-out') || html.includes('504 gateway timeout')) {
        throw new Error('后端网关超时（504）：比对耗时过长或服务不可达，请稍后重试。')
      }
      if (html.includes('502 bad gateway')) {
        throw new Error('后端网关错误（502）：后端服务不可用或代理配置异常。')
      }
      if (html.includes('503 service temporarily unavailable') || html.includes('503 service unavailable')) {
        throw new Error('后端服务不可用（503）：后端正在重启或过载，请稍后重试。')
      }
      throw new Error(`请求失败（HTTP ${res.status}）：后端返回了 HTML 错误页。`)
    }
    try {
      const obj = JSON.parse(raw) as { detail?: unknown }
      if (obj && typeof obj === 'object' && 'detail' in obj && obj.detail) {
        throw new Error(String(obj.detail))
      }
    } catch {
      // ignore
    }
    throw new Error(raw || `HTTP ${res.status}`)
  }
  const payload = (await res.json()) as T & { detail?: unknown }
  if (payload && typeof payload === 'object' && payload.detail) {
    throw new Error(String(payload.detail))
  }
  return payload as T
}

export async function compare(input: { queryImage?: File; view: string; vehicleType: string; topk: number }): Promise<CompareResponse> {
  if (USE_MOCK) return await mockCompare(input.topk)

  const form = new FormData()
  if (input.queryImage) form.append('query_image', input.queryImage)
  form.append('view', input.view)
  form.append('vehicle_type', input.vehicleType)
  form.append('topk', String(input.topk))
  return await http<CompareResponse>('/api/compare', { method: 'POST', body: form })
}

export async function getCandidateDetail(runId: string, candidateId: string): Promise<CandidateDetailResponse> {
  if (USE_MOCK) return await mockGetCandidateDetail(runId, candidateId)
  return await http<CandidateDetailResponse>(`/api/runs/${encodeURIComponent(runId)}/candidates/${encodeURIComponent(candidateId)}`)
}

export async function getGallery(input: { view: string; vehicleType: string }): Promise<GalleryResponse> {
  if (USE_MOCK) return await mockGetGallery()

  const sp = new URLSearchParams()
  sp.set('view', input.view)
  sp.set('vehicle_type', input.vehicleType)
  return await http<GalleryResponse>(`/api/gallery?${sp.toString()}`)
}
