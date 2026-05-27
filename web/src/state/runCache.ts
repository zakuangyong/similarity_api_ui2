import type { CompareResponse } from '@/api/types'

const KEY = 'similarity_ui:last_compare'

export function saveLastCompare(payload: CompareResponse) {
  sessionStorage.setItem(KEY, JSON.stringify(payload))
}

export function loadLastCompare(): CompareResponse | null {
  const raw = sessionStorage.getItem(KEY)
  if (!raw) return null
  try {
    return JSON.parse(raw) as CompareResponse
  } catch {
    return null
  }
}

