<template>
  <div class="pagination" v-if="totalPages > 0">
    <span class="page-info">共 {{ total }} 条 · 第 {{ page }}/{{ totalPages }} 页</span>
    <div class="page-actions">
      <button :disabled="page <= 1" @click="$emit('change', page - 1)">‹ 上一页</button>
      <button :disabled="page >= totalPages" @click="$emit('change', page + 1)">下一页 ›</button>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'

const props = defineProps<{ page: number; total: number; pageSize: number }>()
const emit = defineEmits<{ change: [page: number] }>()

const totalPages = computed(() => Math.ceil(props.total / props.pageSize))
</script>

<style scoped>
.pagination { display: flex; justify-content: space-between; align-items: center; padding: 8px 0; font-size: 12px; color: #888; }
.page-actions { display: flex; gap: 4px; }
.page-actions button { padding: 4px 10px; border: 1px solid #d9d9d9; border-radius: 4px; background: #fff; cursor: pointer; }
.page-actions button:disabled { opacity: 0.4; cursor: default; }
</style>
