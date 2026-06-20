<template>
  <Teleport to="body">
    <div class="toast-container">
      <TransitionGroup name="toast">
        <div v-for="t in toasts" :key="t.id" class="toast" :class="'toast-' + t.type">
          <span>{{ t.message }}</span>
          <button class="toast-close" @click="remove(t.id)">×</button>
        </div>
      </TransitionGroup>
    </div>
  </Teleport>
</template>

<script setup lang="ts">
interface ToastItem { id: number; message: string; type: 'success' | 'error' | 'info' }
const toasts = ref<ToastItem[]>([])
let nextId = 0
function add(message: string, type: 'success' | 'error' | 'info' = 'info', duration = 3000) {
  const id = nextId++
  toasts.value.push({ id, message, type })
  if (duration > 0) setTimeout(() => remove(id), duration)
}
function remove(id: number) { toasts.value = toasts.value.filter(t => t.id !== id) }
defineExpose({ add })
import { ref } from 'vue'
</script>

<style scoped>
.toast-container { position: fixed; top: 16px; right: 16px; z-index: 1000; display: flex; flex-direction: column; gap: 8px; }
.toast { display: flex; align-items: center; gap: 8px; padding: 10px 16px; border-radius: 6px; font-size: 13px; box-shadow: 0 4px 12px rgba(0,0,0,0.15); min-width: 200px; max-width: 400px; }
.toast-success { background: #f6ffed; color: #52c41a; border: 1px solid #b7eb8f; }
.toast-error { background: #fff2f0; color: #ff4d4f; border: 1px solid #ffccc7; }
.toast-info { background: #e6f7ff; color: #1890ff; border: 1px solid #91d5ff; }
.toast-close { background: none; border: none; font-size: 16px; cursor: pointer; margin-left: auto; opacity: 0.5; }
.toast-close:hover { opacity: 1; }
.toast-enter-active { transition: all 0.3s ease; }
.toast-leave-active { transition: all 0.2s ease; }
.toast-enter-from { opacity: 0; transform: translateX(30px); }
.toast-leave-to { opacity: 0; transform: translateX(30px); }
</style>
