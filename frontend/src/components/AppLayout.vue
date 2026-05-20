<template>
  <div class="layout">
    <aside class="sidebar">
      <div class="logo">FaceRecServer</div>
      <nav>
        <router-link v-for="item in navItems" :key="item.path" :to="item.path" class="nav-item" active-class="active">
          <span class="nav-icon">{{ item.icon }}</span>
          <span>{{ item.label }}</span>
        </router-link>
      </nav>
    </aside>
    <div class="main">
      <header class="topbar">
        <span class="title">{{ currentTitle }}</span>
      </header>
      <main class="content">
        <router-view />
      </main>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useRoute } from 'vue-router'

const route = useRoute()

const navItems = [
  { path: '/dashboard', icon: '📊', label: '仪表盘' },
  { path: '/gallery', icon: '👤', label: '人脸底库' },
  { path: '/recognize', icon: '🔍', label: '人脸识别' },
  { path: '/settings', icon: '⚙', label: '系统设置' },
]

const currentTitle = computed(() => {
  const item = navItems.find(n => route.path.startsWith(n.path))
  return item?.label || 'FaceRecServer'
})
</script>

<style scoped>
.layout { display: flex; min-height: 100vh; }
.sidebar { width: 200px; background: #1a1a2e; color: #fff; display: flex; flex-direction: column; flex-shrink: 0; }
.logo { padding: 20px 16px; font-size: 14px; font-weight: bold; letter-spacing: 1px; }
.nav-item { display: flex; align-items: center; gap: 8px; padding: 12px 16px; color: #a0a0b8; text-decoration: none; transition: 0.2s; font-size: 14px; }
.nav-item:hover { color: #fff; background: rgba(255,255,255,0.05); }
.nav-item.active { color: #fff; background: #4A90D9; }
.nav-icon { font-size: 16px; }
.main { flex: 1; display: flex; flex-direction: column; min-width: 0; }
.topbar { height: 48px; background: #fff; border-bottom: 1px solid #e8e8e8; display: flex; align-items: center; padding: 0 20px; font-weight: bold; flex-shrink: 0; }
.content { flex: 1; padding: 20px; overflow-y: auto; }
</style>
