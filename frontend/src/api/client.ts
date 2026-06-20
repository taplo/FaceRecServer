import type { ApiResponse, GalleryListData, RecognizeItem, StatsData, ReindexResult } from '@/types'

const BASE = '/api/v1'

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${url}`, {
    headers: { 'Content-Type': 'application/json', ...options?.headers },
    ...options,
  })
  const json: ApiResponse<T> = await res.json()
  if (json.code !== 0) throw new Error(json.message)
  return json.data as T
}

export const api = {
  getStats: () => request<StatsData>('/stats'),

  listGallery: (page = 1, pageSize = 20, search = '') =>
    request<GalleryListData>(`/gallery?page=${page}&page_size=${pageSize}&search=${encodeURIComponent(search)}`),

  registerFace: (formData: FormData) =>
    fetch(`${BASE}/gallery`, { method: 'POST', body: formData }).then(r => r.json()),

  registerFaceZip: (formData: FormData) =>
    fetch(`${BASE}/gallery/batch`, { method: 'POST', body: formData }).then(r => r.json()),

  deleteFace: (faceId: string) =>
    request<null>(`/gallery/${faceId}`, { method: 'DELETE' }),

  clearGallery: () =>
    request<null>('/gallery', { method: 'DELETE' }),

  recognize: async (formData: FormData, topK = 5): Promise<{ results: RecognizeItem[] }> => {
    const res = await fetch(`${BASE}/gallery/recognize?top_k=${topK}`, { method: 'POST', body: formData })
    const json: ApiResponse<{ results: RecognizeItem[] }> = await res.json()
    if (json.code !== 0) throw new Error(json.message)
    return json.data!
  },

  compare: async (formData: FormData): Promise<{ score: number }> => {
    const res = await fetch(`${BASE}/compare`, { method: 'POST', body: formData })
    const json = await res.json()
    if (json.code !== 0) throw new Error(json.message)
    const data = json.data
    return { score: data.similarity ?? data.score ?? 0 }
  },

  rebuildIndex: () => request<ReindexResult>('/gallery/reindex', { method: 'POST' }),
}
