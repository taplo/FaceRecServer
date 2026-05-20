<template>
  <div class="settings">
    <section class="section">
      <h3>🧠 模型管理</h3>
      <div class="field">
        <label>当前模型</label>
        <select v-model="currentModel" disabled>
          <option>{{ currentModel }}</option>
        </select>
      </div>
      <div class="meta-info">
        <span>路径: models/swin_arcface_webface4m_tinyface/model.pt</span>
        <span>维度: 128</span>
        <span>推理: ~500ms/张</span>
      </div>
    </section>

    <section class="section">
      <h3>🔗 底库信息</h3>
      <div class="stats-row" v-if="stats">
        <div class="stat-item">
          <span class="stat-key">底库总数</span>
          <span>{{ stats.gallery.total_faces }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-key">索引大小</span>
          <span>{{ stats.gallery.index_size }}</span>
        </div>
        <div class="stat-item">
          <span class="stat-key">特征维度</span>
          <span>{{ stats.gallery.dimension }}</span>
        </div>
      </div>
      <div v-else-if="loading" class="loading">加载中...</div>
      <div class="actions">
        <div class="btn-group">
          <button class="btn btn-warning" @click="rebuildIndex">重建索引</button>
          <button class="btn btn-danger" @click="doClear">清空底库</button>
        </div>
      </div>
    </section>

    <section class="section">
      <h3>ℹ️ 关于</h3>
      <div class="about-info">
        <div class="stat-item"><span class="stat-key">后端版本</span><span>v0.1.0</span></div>
        <div class="stat-item"><span class="stat-key">运行设备</span><span>{{ stats?.server.device || '--' }}</span></div>
        <div class="stat-item"><span class="stat-key">运行时间</span><span>{{ formatUptime(stats?.server.uptime_seconds || 0) }}</span></div>
        <div class="stat-item" style="margin-top:8px;font-size:11px;color:#bbb">基于 PETALface (WACV 2025) 算法构建</div>
      </div>
    </section>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { api } from '@/api/client'
import type { StatsData } from '@/types'

const currentModel = ref('swin_arcface_webface4m_tinyface')
const stats = ref<StatsData | null>(null)
const loading = ref(true)

onMounted(async () => {
  try { stats.value = await api.getStats() } catch {}
  finally { loading.value = false }
})

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}

function rebuildIndex() {
  alert('重建索引功能需重启服务器生效。请手动重启: python -m facerecserver')
}

async function doClear() {
  if (!confirm('确认清空整个底库？此操作不可恢复！')) return
  try {
    await api.clearGallery()
    alert('底库已清空')
  } catch (e) {
    alert('清空失败')
  }
}
</script>

<style scoped>
.section { background: #fff; border-radius: 8px; padding: 20px; margin-bottom: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.section h3 { margin-bottom: 16px; font-size: 15px; }
.field { display: flex; align-items: center; gap: 12px; margin-bottom: 8px; }
.field label { width: 80px; font-size: 13px; color: #666; flex-shrink: 0; }
.field select { flex: 1; padding: 6px; border: 1px solid #d9d9d9; border-radius: 4px; font-size: 13px; max-width: 400px; }
.meta-info { display: flex; gap: 16px; font-size: 12px; color: #888; margin-top: 8px; }
.stats-row { display: flex; gap: 24px; margin-bottom: 16px; }
.stat-item { display: flex; flex-direction: column; font-size: 13px; }
.stat-key { color: #888; font-size: 12px; }
.actions { display: flex; gap: 8px; }
.btn-group { display: flex; gap: 8px; }
.btn { padding: 8px 16px; border: none; border-radius: 4px; cursor: pointer; font-size: 13px; }
.btn-warning { background: #faad14; color: #fff; }
.btn-danger { background: #ff4d4f; color: #fff; }
.about-info { display: flex; flex-direction: column; gap: 4px; }
.loading { font-size: 13px; color: #888; padding: 8px 0; }
</style>
