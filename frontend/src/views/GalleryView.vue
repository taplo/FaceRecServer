<template>
  <div class="gallery">
    <div class="toolbar">
      <input v-model="searchQuery" placeholder="搜索姓名..." class="search-input" @keyup.enter="loadFaces(1)" />
      <button class="btn btn-primary" @click="showRegister = true">+ 注册人脸</button>
      <button class="btn btn-danger-outline" @click="confirmClear">清空底库</button>
    </div>

    <div class="content-split">
      <div class="detail-panel" v-if="selectedFace">
        <div class="detail-header">
          <img v-if="selectedFace.image_url" :src="selectedFace.image_url" class="detail-avatar" />
          <div class="detail-info">
            <strong>{{ selectedFace.name }}</strong>
            <div class="detail-meta">ID: {{ selectedFace.face_id.slice(0, 8) }}...</div>
            <div class="detail-meta">注册: {{ selectedFace.created_at }}</div>
          </div>
        </div>
        <button class="btn btn-danger btn-sm" @click="deleteFace(selectedFace.face_id)">删除此人</button>
      </div>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr><th>头像</th><th>姓名</th><th>ID</th><th>注册时间</th><th>操作</th></tr>
          </thead>
          <tbody>
            <tr v-for="face in faces" :key="face.face_id" :class="{ selected: selectedFace?.face_id === face.face_id }" @click="selectedFace = face">
              <td><img v-if="face.image_url" :src="face.image_url" class="thumb" /></td>
              <td>{{ face.name }}</td>
              <td class="mono">{{ face.face_id.slice(0, 8) }}...</td>
              <td>{{ formatTime(face.created_at) }}</td>
              <td><button class="btn btn-sm btn-danger-text" @click.stop="deleteFace(face.face_id)">删除</button></td>
            </tr>
            <tr v-if="faces.length === 0">
              <td colspan="5" class="empty">暂无数据</td>
            </tr>
          </tbody>
        </table>
        <Pagination :page="page" :total="total" :pageSize="pageSize" @change="loadFaces" />
      </div>
    </div>

    <!-- Register dialog -->
    <div class="modal-overlay" v-if="showRegister" @click.self="showRegister = false">
      <div class="modal">
        <h3>注册人脸</h3>
        <p class="subtitle">支持 JPG / PNG 格式</p>
        <div class="upload-area" @click="$refs.fileInput.click()">
          <img v-if="previewUrl" :src="previewUrl" class="preview-img" />
          <span v-else class="upload-hint">点击选择图片</span>
        </div>
        <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileSelect" />
        <div class="modal-actions" v-if="registerFile">
          <button class="btn btn-primary" @click="registerFace">确认注册</button>
          <button class="btn" @click="showRegister = false">取消</button>
        </div>
        <p v-if="registerError" class="error">{{ registerError }}</p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { api } from '@/api/client'
import type { FaceRecord } from '@/types'
import Pagination from '@/components/Pagination.vue'

const faces = ref<FaceRecord[]>([])
const selectedFace = ref<FaceRecord | null>(null)
const page = ref(1)
const total = ref(0)
const pageSize = 20
const searchQuery = ref('')
const showRegister = ref(false)
const registerFile = ref<File | null>(null)
const previewUrl = ref('')
const registerError = ref('')

onMounted(() => loadFaces(1))

async function loadFaces(p: number) {
  page.value = p
  try {
    const data = await api.listGallery(p, pageSize, searchQuery.value)
    faces.value = data.items.map(f => ({ ...f, image_url: `/api/v1/gallery/${f.face_id}/image` }))
    total.value = data.total
  } catch (e) {
    console.error('加载失败', e)
  }
}

function onFileSelect(e: Event) {
  const input = e.target as HTMLInputElement
  const file = input.files?.[0]
  if (!file) return
  registerFile.value = file
  const reader = new FileReader()
  reader.onload = () => { previewUrl.value = reader.result as string }
  reader.readAsDataURL(file)
}

async function registerFace() {
  if (!registerFile.value) return
  registerError.value = ''
  const fd = new FormData()
  fd.append('file', registerFile.value)
  try {
    const res = await api.registerFace(fd)
    if (res.code !== 0) { registerError.value = res.message; return }
    showRegister.value = false
    registerFile.value = null
    previewUrl.value = ''
    loadFaces(1)
  } catch (e: any) {
    registerError.value = e.message
  }
}

async function deleteFace(faceId: string) {
  if (!confirm('确认删除此条记录？')) return
  try {
    await api.deleteFace(faceId)
    if (selectedFace.value?.face_id === faceId) selectedFace.value = null
    loadFaces(page.value)
  } catch (e) { console.error('删除失败', e) }
}

function confirmClear() {
  if (!confirm('确认清空整个底库？此操作不可恢复！')) return
  api.clearGallery().then(() => { selectedFace.value = null; loadFaces(1) })
}

function formatTime(iso: string): string {
  if (!iso) return '--'
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}
</script>

<style scoped>
.toolbar { display: flex; gap: 8px; margin-bottom: 16px; }
.search-input { flex: 1; padding: 8px 12px; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 14px; }
.content-split { display: flex; gap: 16px; }
.detail-panel { width: 240px; background: #fff; border-radius: 8px; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); flex-shrink: 0; }
.detail-header { display: flex; gap: 12px; margin-bottom: 12px; }
.detail-avatar { width: 64px; height: 64px; border-radius: 8px; object-fit: cover; }
.detail-info { flex: 1; }
.detail-meta { font-size: 11px; color: #888; margin-top: 2px; }
.table-wrapper { flex: 1; background: #fff; border-radius: 8px; padding: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); overflow-x: auto; }
.thumb { width: 32px; height: 32px; border-radius: 4px; object-fit: cover; }
.mono { font-family: monospace; font-size: 12px; color: #888; }
.selected { background: #e6f7ff; cursor: pointer; }
tr { cursor: pointer; }
td, th { padding: 8px; text-align: left; border-bottom: 1px solid #f0f0f0; font-size: 13px; }
th { font-weight: 600; color: #555; }
.empty { text-align: center; color: #bbb; padding: 40px !important; }
.btn { padding: 8px 16px; border: 1px solid #d9d9d9; border-radius: 4px; background: #fff; font-size: 13px; }
.btn-primary { background: #4A90D9; color: #fff; border-color: #4A90D9; }
.btn-danger { background: #ff4d4f; color: #fff; border-color: #ff4d4f; }
.btn-danger-outline { color: #ff4d4f; border-color: #ff4d4f; background: #fff; }
.btn-danger-text { color: #ff4d4f; background: none; border: none; padding: 4px 8px; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 100; }
.modal { background: #fff; border-radius: 8px; padding: 24px; min-width: 400px; }
.modal h3 { margin-bottom: 4px; }
.modal .subtitle { font-size: 12px; color: #888; margin-bottom: 12px; }
.upload-area { border: 2px dashed #d9d9d9; border-radius: 8px; padding: 20px; text-align: center; cursor: pointer; margin-bottom: 12px; }
.upload-area:hover { border-color: #4A90D9; }
.preview-img { max-height: 200px; max-width: 100%; }
.upload-hint { color: #bbb; font-size: 14px; }
.modal-actions { display: flex; gap: 8px; }
.error { color: #ff4d4f; font-size: 12px; margin-top: 8px; }
</style>
