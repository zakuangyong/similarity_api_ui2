import type { CandidateDetailResponse, CompareResponse, GalleryResponse } from './types'
import { mockCompare, mockGetCandidateDetail, mockGetGallery } from './mock'

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL as string | undefined) ?? ''
const USE_MOCK = (import.meta.env.VITE_USE_MOCK as string | undefined) !== 'false'

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, init)
  if (!res.ok) {
    const raw = await res.text().catch(() => '')
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
  return (await res.json()) as T
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
