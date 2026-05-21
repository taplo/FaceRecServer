<template>
  <div class="recognize">
    <div class="split-panels">
      <!-- 1:1 Comparison -->
      <div class="panel">
        <div class="panel-header" style="background:#4A90D9">1:1 人脸比对</div>
        <div class="panel-body">
          <div class="image-pair">
            <div class="upload-box" @click="triggerUpload('imgA')">
              <img v-if="imgAPreview" :src="imgAPreview" class="preview" />
              <span v-else class="upload-hint">点击上传图片A</span>
            </div>
            <div class="upload-box" @click="triggerUpload('imgB')">
              <img v-if="imgBPreview" :src="imgBPreview" class="preview" />
              <span v-else class="upload-hint">点击上传图片B</span>
            </div>
          </div>
          <input ref="fileInputA" type="file" accept="image/*" hidden @change="onFileA" />
          <input ref="fileInputB" type="file" accept="image/*" hidden @change="onFileB" />
          <button class="btn btn-compare" @click="doCompare" :disabled="!imgAFile || !imgBFile">开始比对</button>
          <div v-if="compareScore !== null" class="result-box" :class="scoreColor(compareScore)">
            相似度: <strong>{{ compareScore.toFixed(4) }}</strong>
          </div>
          <p v-if="compareError" class="error">{{ compareError }}</p>
        </div>
      </div>

      <!-- 1:N Search -->
      <div class="panel">
        <div class="panel-header" style="background:#722ed1">1:N 人脸搜索</div>
        <div class="panel-body">
          <div class="search-upload-box" @click="triggerUpload('search')">
            <img v-if="searchPreview" :src="searchPreview" class="preview" />
            <span v-else class="upload-hint">点击上传查询图片</span>
          </div>
          <input ref="fileInputSearch" type="file" accept="image/*" hidden @change="onSearchFile" />
          <div class="search-controls">
            <select v-model="topK">
              <option :value="5">Top-5</option>
              <option :value="10">Top-10</option>
              <option :value="20">Top-20</option>
            </select>
            <button class="btn btn-search" @click="doSearch" :disabled="!searchFile">开始搜索</button>
          </div>
          <div v-if="searchResults.length" class="results">
            <div v-for="(item, i) in searchResults" :key="item.face_id" class="result-item">
              <img v-if="item.image_url" :src="item.image_url" class="result-thumb" />
              <div class="result-info">
                <span>{{ rankLabel(i) }} {{ item.name }}<span v-if="item.employee_id" class="emp-tag">#{{ item.employee_id }}</span></span>
                <span class="score" :class="scoreColor(item.score)">{{ item.score.toFixed(4) }}</span>
              </div>
            </div>
          </div>
          <p v-if="searchError" class="error">{{ searchError }}</p>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref } from 'vue'
import { api } from '@/api/client'
import type { RecognizeItem } from '@/types'

const fileInputA = ref<HTMLInputElement>()
const fileInputB = ref<HTMLInputElement>()
const fileInputSearch = ref<HTMLInputElement>()
const imgAPreview = ref('')
const imgBPreview = ref('')
const searchPreview = ref('')
const imgAFile = ref<File | null>(null)
const imgBFile = ref<File | null>(null)
const searchFile = ref<File | null>(null)
const compareScore = ref<number | null>(null)
const compareError = ref('')
const searchResults = ref<RecognizeItem[]>([])
const searchError = ref('')
const topK = ref(5)

function triggerUpload(target: string) {
  if (target === 'imgA') fileInputA.value?.click()
  else if (target === 'imgB') fileInputB.value?.click()
  else fileInputSearch.value?.click()
}

function readPreview(file: File): Promise<string> {
  return new Promise(resolve => {
    const r = new FileReader()
    r.onload = () => resolve(r.result as string)
    r.readAsDataURL(file)
  })
}

async function onFileA(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return; imgAFile.value = f; imgAPreview.value = await readPreview(f)
  compareScore.value = null; compareError.value = ''
}

async function onFileB(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return; imgBFile.value = f; imgBPreview.value = await readPreview(f)
  compareScore.value = null; compareError.value = ''
}

async function onSearchFile(e: Event) {
  const f = (e.target as HTMLInputElement).files?.[0]
  if (!f) return; searchFile.value = f; searchPreview.value = await readPreview(f)
  searchResults.value = []; searchError.value = ''
}

async function doCompare() {
  if (!imgAFile.value || !imgBFile.value) return
  compareError.value = ''
  const fdA = new FormData()
  fdA.append('file', imgAFile.value)
  const fdB = new FormData()
  fdB.append('file', imgBFile.value)
  try {
    const registerRes = await api.registerFace(fdB)
    if (registerRes.code !== 0) {
      compareError.value = registerRes.message
      return
    }
    const tempFaceId = registerRes.data?.face_id
    const searchFd = new FormData()
    searchFd.append('file', imgAFile.value)
    const searchData = await api.recognize(searchFd, 1)
    if (tempFaceId) {
      await api.deleteFace(tempFaceId).catch(() => {})
    }
    compareScore.value = searchData.results[0]?.score ?? 0
  } catch (e: any) {
    compareError.value = e.message || '比对失败'
  }
}

async function doSearch() {
  if (!searchFile.value) return
  searchError.value = ''
  const fd = new FormData()
  fd.append('file', searchFile.value)
  try {
    const data = await api.recognize(fd, topK.value)
    searchResults.value = data.results.map(r => ({
      ...r,
      image_url: `/api/v1/gallery/${r.face_id}/image`,
    }))
  } catch (e: any) {
    searchError.value = e.message || '搜索失败'
  }
}

function rankLabel(i: number): string {
  return ['🥇', '🥈', '🥉', '', ''][i] || `${i + 1}.`
}

function scoreColor(score: number): string {
  return score >= 0.6 ? 'score-high' : score >= 0.4 ? 'score-mid' : 'score-low'
}


</script>

<style scoped>
.split-panels { display: flex; gap: 16px; }
.panel { flex: 1; background: #fff; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.panel-header { color: #fff; padding: 10px 14px; font-weight: bold; font-size: 14px; }
.panel-body { padding: 16px; }
.image-pair { display: flex; gap: 12px; margin-bottom: 12px; }
.upload-box { flex: 1; height: 120px; border: 2px dashed #d9d9d9; border-radius: 8px; display: flex; align-items: center; justify-content: center; cursor: pointer; overflow: hidden; }
.upload-box:hover { border-color: #4A90D9; }
.search-upload-box { width: 120px; height: 120px; border: 2px dashed #d9d9d9; border-radius: 8px; display: flex; align-items: center; justify-content: center; cursor: pointer; overflow: hidden; margin-bottom: 12px; }
.search-upload-box:hover { border-color: #722ed1; }
.upload-hint { font-size: 12px; color: #bbb; text-align: center; padding: 8px; }
.preview { width: 100%; height: 100%; object-fit: cover; }
.btn { padding: 8px 16px; border: none; border-radius: 4px; font-size: 13px; cursor: pointer; }
.btn:disabled { opacity: 0.5; cursor: default; }
.btn-compare { background: #4A90D9; color: #fff; width: 100%; margin-bottom: 12px; padding: 10px; }
.btn-search { background: #722ed1; color: #fff; flex: 1; }
.result-box { padding: 12px; border-radius: 6px; text-align: center; font-size: 14px; }
.result-box.score-high { background: #f6ffed; color: #52c41a; }
.result-box.score-mid { background: #fffbe6; color: #faad14; }
.result-box.score-low { background: #fff2f0; color: #ff4d4f; }
.search-controls { display: flex; gap: 8px; margin-bottom: 12px; }
.search-controls select { flex: 1; padding: 6px; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 13px; }
.results { display: flex; flex-direction: column; gap: 2px; }
.result-item { display: flex; align-items: center; gap: 8px; padding: 6px 0; border-bottom: 1px solid #f0f0f0; font-size: 13px; }
.result-thumb { width: 32px; height: 32px; border-radius: 4px; object-fit: cover; }
.result-info { flex: 1; display: flex; justify-content: space-between; align-items: center; }
.score { font-weight: bold; }
.score-high { color: #52c41a; }
.score-mid { color: #faad14; }
.score-low { color: #ff4d4f; }
.emp-tag { font-size: 11px; color: #888; margin-left: 4px; background: #f5f5f5; padding: 1px 5px; border-radius: 3px; }
.error { color: #ff4d4f; font-size: 12px; margin-top: 8px; }
</style>
