<template>
  <Teleport to="body">
    <Transition name="modal">
      <div v-if="show" class="modal-overlay" @click.self="$emit('close')">
        <div class="modal">
          <div class="modal-header">
            <h3>{{ title }}</h3>
            <button class="modal-close" @click="$emit('close')">×</button>
          </div>
          <div class="modal-body">
            <slot />
          </div>
          <div v-if="$slots.footer" class="modal-footer">
            <slot name="footer" />
          </div>
        </div>
      </div>
    </Transition>
  </Teleport>
</template>

<script setup lang="ts">
defineProps<{ show: boolean; title: string }>()
defineEmits<{ close: [] }>()
</script>

<style scoped>
.modal-overlay { position: fixed; inset: 0; background: rgba(0,0,0,0.4); display: flex; align-items: center; justify-content: center; z-index: 200; }
.modal { background: #fff; border-radius: 8px; min-width: 420px; max-width: 560px; max-height: 80vh; overflow-y: auto; }
.modal-header { display: flex; align-items: center; justify-content: space-between; padding: 16px 20px 0; }
.modal-header h3 { font-size: 15px; margin: 0; }
.modal-close { background: none; border: none; font-size: 20px; cursor: pointer; color: #999; padding: 0 4px; }
.modal-close:hover { color: #333; }
.modal-body { padding: 16px 20px; }
.modal-footer { padding: 0 20px 16px; display: flex; gap: 8px; justify-content: flex-end; }
.modal-enter-active { transition: all 0.2s ease; }
.modal-leave-active { transition: all 0.15s ease; }
.modal-enter-from { opacity: 0; }
.modal-enter-from .modal { transform: scale(0.95); }
.modal-leave-to { opacity: 0; }
</style>
