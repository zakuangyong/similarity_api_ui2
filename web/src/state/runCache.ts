import type { CompareResponse } from '@/api/types'

const KEY = 'similarity_ui:last_compare'
const QUERY_NAME_KEY = 'similarity_ui:last_query_name'

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

export function saveLastQueryName(name: string) {
  sessionStorage.setItem(QUERY_NAME_KEY, String(name || ''))
}

export function loadLastQueryName(): string {
  return sessionStorage.getItem(QUERY_NAME_KEY) || ''
}

