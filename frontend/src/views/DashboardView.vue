<template>
  <div class="dashboard">
    <div class="stats-grid" v-if="stats">
      <div class="stat-card">
        <div class="stat-value">{{ stats.gallery.total_faces }}</div>
        <div class="stat-label">底库人脸总数</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.gallery.dimension }}</div>
        <div class="stat-label">特征维度</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ formatUptime(stats.server.uptime_seconds) }}</div>
        <div class="stat-label">服务器运行</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">{{ stats.server.device }}</div>
        <div class="stat-label">运行设备</div>
      </div>
    </div>
    <div v-else class="loading">加载中...</div>
    <p v-if="error" class="error">{{ error }}</p>
  </div>
</template>

<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { api } from '@/api/client'
import type { StatsData } from '@/types'

const stats = ref<StatsData | null>(null)
const error = ref('')

onMounted(async () => {
  try {
    stats.value = await api.getStats()
  } catch (e: any) {
    error.value = '获取统计失败: ' + (e.message || '未知错误')
  }
})

function formatUptime(seconds: number): string {
  const h = Math.floor(seconds / 3600)
  const m = Math.floor((seconds % 3600) / 60)
  return `${h}h ${m}m`
}
</script>

<style scoped>
.stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.stat-card { background: #fff; border-radius: 8px; padding: 24px; text-align: center; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }
.stat-value { font-size: 32px; font-weight: bold; color: #4A90D9; margin-bottom: 4px; }
.stat-label { font-size: 13px; color: #888; }
.loading { text-align: center; padding: 40px; color: #888; }
.error { color: #ff4d4f; padding: 12px; background: #fff2f0; border-radius: 6px; }
</style>
