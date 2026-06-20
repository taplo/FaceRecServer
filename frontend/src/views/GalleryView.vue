<template>
  <div class="gallery">
    <div class="toolbar">
      <input v-model="searchQuery" placeholder="搜索姓名/工号..." class="search-input" @keyup.enter="loadFaces(1)" />
      <button class="btn btn-primary" @click="showRegister = true">+ 注册人脸</button>
      <button class="btn btn-outline" @click="showBatch = true">批量导入</button>
      <button class="btn btn-danger-outline" @click="confirmClear">清空底库</button>
    </div>

    <div class="content-split">
      <div class="detail-panel" v-if="selectedFace">
        <div class="detail-header">
          <img v-if="selectedFace.image_url" :src="selectedFace.image_url" class="detail-avatar" />
          <div class="detail-info">
            <strong>{{ selectedFace.name }}</strong>
            <div class="detail-meta" v-if="selectedFace.employee_id">工号: {{ selectedFace.employee_id }}</div>
            <div class="detail-meta">ID: {{ selectedFace.face_id.slice(0, 8) }}...</div>
            <div class="detail-meta">注册: {{ formatTime(selectedFace.created_at) }}</div>
          </div>
        </div>
        <button class="btn btn-danger btn-sm" @click="deleteFace(selectedFace.face_id)">删除此人</button>
      </div>

      <div class="table-wrapper">
        <table>
          <thead>
            <tr><th>头像</th><th>姓名</th><th>工号</th><th>ID</th><th>注册时间</th><th>操作</th></tr>
          </thead>
          <tbody>
            <tr v-for="face in faces" :key="face.face_id" :class="{ selected: selectedFace?.face_id === face.face_id }" @click="selectedFace = face">
              <td><img v-if="face.image_url" :src="face.image_url" class="thumb" /></td>
              <td>{{ face.name }}</td>
              <td>{{ face.employee_id || '--' }}</td>
              <td class="mono">{{ face.face_id.slice(0, 8) }}...</td>
              <td>{{ formatTime(face.created_at) }}</td>
              <td><button class="btn btn-sm btn-danger-text" @click.stop="deleteFace(face.face_id)">删除</button></td>
            </tr>
            <tr v-if="faces.length === 0">
              <td colspan="6" class="empty">暂无数据</td>
            </tr>
          </tbody>
        </table>
        <Pagination :page="page" :total="total" :pageSize="pageSize" @change="loadFaces" />
      </div>
    </div>

    <Modal :show="showRegister" title="注册人脸" @close="showRegister = false">
      <p class="subtitle">支持 JPG / PNG 格式</p>
      <div class="form-row">
        <div class="form-field">
          <label>姓名</label>
          <input v-model="regName" placeholder="输入姓名" class="form-input" />
        </div>
        <div class="form-field">
          <label>工号</label>
          <input v-model="regEmpId" placeholder="选填" class="form-input" />
        </div>
      </div>
      <div class="upload-area" @click="fileInput?.click()">
        <img v-if="previewUrl" :src="previewUrl" class="preview-img" />
        <span v-else class="upload-hint">点击选择图片</span>
      </div>
      <input ref="fileInput" type="file" accept="image/*" hidden @change="onFileSelect" />
      <template #footer>
        <button class="btn" @click="showRegister = false">取消</button>
        <button class="btn btn-primary" @click="registerFace" :disabled="!registerFile">确认注册</button>
      </template>
    </Modal>

    <Modal :show="showBatch" title="批量导入" @close="showBatch = false">
      <p class="subtitle">上传 ZIP 压缩包，文件名格式: <code>姓名-工号.jpg</code></p>
      <div class="batch-upload-area" @click="zipInput?.click()">
        <span v-if="!batchFile" class="upload-hint">点击选择 ZIP 文件</span>
        <div v-else class="batch-file-info">
          <span class="batch-icon">📦</span>
          <span>{{ batchFile.name }} ({{ (batchFile.size / 1024 / 1024).toFixed(1) }} MB)</span>
        </div>
      </div>
      <input ref="zipInput" type="file" accept=".zip" hidden @change="onZipSelect" />
      <div v-if="batchProgress" class="batch-progress">
        <p>{{ batchProgress }}</p>
      </div>
      <template #footer>
        <button class="btn" @click="showBatch = false">取消</button>
        <button class="btn btn-primary" @click="doBatchImport" :disabled="!batchFile || batchImporting">
          {{ batchImporting ? '导入中...' : '开始导入' }}
        </button>
      </template>
    </Modal>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { api } from '@/api/client'
import type { FaceRecord } from '@/types'
import Pagination from '@/components/Pagination.vue'
import Modal from '@/components/Modal.vue'

const faces = ref<FaceRecord[]>([])
const selectedFace = ref<FaceRecord | null>(null)
const page = ref(1)
const total = ref(0)
const pageSize = 20
const searchQuery = ref('')

const showRegister = ref(false)
const registerFile = ref<File | null>(null)
const previewUrl = ref('')
const fileInput = ref<HTMLInputElement | null>(null)
const regName = ref('')
const regEmpId = ref('')

const showBatch = ref(false)
const batchFile = ref<File | null>(null)
const zipInput = ref<HTMLInputElement | null>(null)
const batchImporting = ref(false)
const batchProgress = ref('')

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
  const fd = new FormData()
  fd.append('file', registerFile.value)
  if (regName.value) fd.append('name', regName.value)
  else regName.value = registerFile.value.name.replace(/\.[^.]+$/, '')
  if (regEmpId.value) fd.append('employee_id', regEmpId.value)
  try {
    const res = await api.registerFace(fd)
    if (res.code !== 0) { alert(res.message); return }
    showRegister.value = false
    registerFile.value = null
    previewUrl.value = ''
    regName.value = ''
    regEmpId.value = ''
    loadFaces(1)
  } catch (e: any) {
    alert(e.message)
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

async function confirmClear() {
  if (!confirm('确认清空整个底库？此操作不可恢复！')) return
  await api.clearGallery()
  selectedFace.value = null
  loadFaces(1)
}

function onZipSelect(e: Event) {
  const input = e.target as HTMLInputElement
  batchFile.value = input.files?.[0] || null
  batchProgress.value = ''
}

async function doBatchImport() {
  if (!batchFile.value) return
  batchImporting.value = true
  batchProgress.value = '上传中...'
  const fd = new FormData()
  fd.append('file', batchFile.value)
  try {
    const res = await api.registerFaceZip(fd)
    if (res.code !== 0) { batchProgress.value = `导入失败: ${res.message}`; return }
    const data = res.data
    batchProgress.value = `导入完成: 成功 ${data.succeeded} 条, 失败 ${data.failed} 条`
    if (data.failures?.length) {
      batchProgress.value += ` (详情见服务器日志)`
    }
    showBatch.value = false
    batchFile.value = null
    batchProgress.value = ''
    loadFaces(1)
  } catch (e: any) {
    batchProgress.value = `导入失败: ${e.message}`
  } finally {
    batchImporting.value = false
  }
}

function formatTime(iso: string): string {
  if (!iso) return '--'
  const d = new Date(iso)
  return d.toLocaleString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit' })
}
</script>

<style scoped>
.toolbar { display: flex; gap: 8px; margin-bottom: 16px; flex-wrap: wrap; }
.search-input { flex: 1; min-width: 200px; padding: 8px 12px; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 14px; }
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
.btn { padding: 8px 16px; border: 1px solid #d9d9d9; border-radius: 4px; background: #fff; font-size: 13px; cursor: pointer; }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn-primary { background: #4A90D9; color: #fff; border-color: #4A90D9; }
.btn-danger { background: #ff4d4f; color: #fff; border-color: #ff4d4f; }
.btn-danger-outline { color: #ff4d4f; border-color: #ff4d4f; background: #fff; }
.btn-danger-text { color: #ff4d4f; background: none; border: none; padding: 4px 8px; }
.btn-outline { color: #555; border-color: #d9d9d9; background: #fff; }
.btn-sm { padding: 4px 10px; font-size: 12px; }
.subtitle { font-size: 12px; color: #888; margin-bottom: 12px; }
.subtitle code { background: #f5f5f5; padding: 1px 5px; border-radius: 3px; }
.form-row { display: flex; gap: 12px; margin-bottom: 12px; }
.form-field { flex: 1; }
.form-field label { display: block; font-size: 12px; color: #666; margin-bottom: 4px; }
.form-input { width: 100%; padding: 6px 10px; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 13px; box-sizing: border-box; }
.upload-area { border: 2px dashed #d9d9d9; border-radius: 8px; padding: 20px; text-align: center; cursor: pointer; margin-bottom: 0; }
.upload-area:hover { border-color: #4A90D9; }
.preview-img { max-height: 200px; max-width: 100%; }
.upload-hint { color: #bbb; font-size: 14px; }
.batch-upload-area { border: 2px dashed #d9d9d9; border-radius: 8px; padding: 24px; text-align: center; cursor: pointer; }
.batch-upload-area:hover { border-color: #4A90D9; }
.batch-file-info { display: flex; align-items: center; justify-content: center; gap: 8px; font-size: 14px; }
.batch-icon { font-size: 24px; }
.batch-progress { margin-top: 8px; font-size: 13px; color: #555; }
</style>
