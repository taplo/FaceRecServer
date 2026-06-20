export interface ApiResponse<T = any> {
  code: number
  message: string
  data: T | null
}

export interface FaceRecord {
  face_id: string
  name: string
  employee_id?: string
  created_at: string
  image_url?: string | null
}

export interface GalleryListData {
  items: FaceRecord[]
  total: number
  page: number
  page_size: number
}

export interface RecognizeItem {
  face_id: string
  name: string
  employee_id?: string
  score: number
  image_url?: string | null
}

export interface StatsData {
  gallery: {
    total_faces: number
    index_size: number
    dimension: number
  }
  server: {
    uptime_seconds: number
    device: string
  }
}

export interface ReindexResult {
  total_faces: number
}
