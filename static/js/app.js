/**
 * Main application controller.
 * Central state management, API clients, and event handling.
 */
const App = (() => {
    // ─── State ───
    const state = {
        currentConversationId: null,
        conversations: [],
        isStreaming: false,
        abortController: null,
        attachments: [],
        systemStatus: null,
    };

    // ─── API Client ───
    const api = {
        async get(url) {
            const resp = await fetch(url);
            if (!resp.ok) throw new Error(`GET ${url} failed: ${resp.status}`);
            return resp.json();
        },

        async post(url, data) {
            const resp = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (!resp.ok) throw new Error(`POST ${url} failed: ${resp.status}`);
            return resp.json();
        },

        async put(url, data) {
            const resp = await fetch(url, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(data),
            });
            if (!resp.ok) throw new Error(`PUT ${url} failed: ${resp.status}`);
            return resp.json();
        },

        async delete(url) {
            const resp = await fetch(url, { method: 'DELETE' });
            if (!resp.ok) throw new Error(`DELETE ${url} failed: ${resp.status}`);
            return resp.json();
        },

        streamChat(message, conversationId, attachment, webSearch = false) {
            const formData = new FormData();
            formData.append('message', message);
            if (attachment) formData.append('attachment', attachment);
            if (webSearch) formData.append('web_search', 'true');

            const url = conversationId
                ? `/api/chat/${conversationId}`
                : '/api/chat';

            state.abortController = new AbortController();

            return fetch(url, {
                method: 'POST',
                body: formData,
                signal: state.abortController.signal,
            });
        },
    };

    // ─── Toast Notifications ───
    function showToast(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.textContent = message;
        container.appendChild(toast);
        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(20px)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }

    // ─── System Status ───
    async function checkStatus() {
        try {
            state.systemStatus = await api.get('/api/status');
            updateStatusDisplay();
        } catch (e) {
            console.error('Status check failed:', e);
        }
    }

    function updateStatusDisplay() {
        const el = document.getElementById('system-status');
        if (!el || !state.systemStatus) return;
        const s = state.systemStatus;
        const dotClass = s.ollama_available && s.model_loaded ? 'online' : 'offline';
        const label = s.model_loaded
            ? `${s.model_name} · ${s.total_chunks} chunks`
            : 'LLM offline';
        el.innerHTML = `<span class="status-dot ${dotClass}"></span>${label}`;

        // Dynamically update model name in UI
        const modelDisplay = s.model_name || 'Ollama';
        const subtitle = document.getElementById('welcome-subtitle');
        if (subtitle) {
            subtitle.innerHTML = `Powered by ${modelDisplay} &bull; FAISS &bull; NeMo Retriever`;
        }
        const hint = document.getElementById('input-hint-text');
        if (hint) {
            hint.textContent = `Local LLM uses ${modelDisplay} + FAISS RAG. Responses may not always be accurate.`;
        }
    }

    // ─── Keyboard Shortcuts ───
    function initShortcuts() {
        document.addEventListener('keydown', (e) => {
            // Ctrl+N → new chat
            if (e.ctrlKey && e.key === 'n') {
                e.preventDefault();
                newChat();
            }

            // Escape → stop streaming
            if (e.key === 'Escape' && state.isStreaming) {
                stopStreaming();
            }
        });
    }

    function newChat() {
        state.currentConversationId = null;
        Chat.clear();
        Sidebar.setActive(null);
        document.getElementById('message-input').focus();
    }

    function stopStreaming() {
        if (state.abortController) {
            state.abortController.abort();
            state.abortController = null;
        }
        state.isStreaming = false;
        Chat.endStreaming();
    }

    // ─── Model Picker ───
    async function loadModels() {
        const select = document.getElementById('model-select');
        if (!select) return;

        try {
            const data = await api.get('/api/models');
            const models = data.models || [];
            const current = data.current_model || '';

            if (models.length === 0) {
                select.innerHTML = '<option value="">No models found</option>';
                return;
            }

            select.innerHTML = models.map(m => {
                const name = m.name;
                const size = formatSize(m.size);
                const param = m.parameter_size ? ` · ${m.parameter_size}` : '';
                const isSelected = name === current ||
                    name.split(':')[0] === current.split(':')[0];
                return `<option value="${name}" ${isSelected ? 'selected' : ''}>${name}${param} (${size})</option>`;
            }).join('');

        } catch (e) {
            console.error('Failed to load models:', e);
            select.innerHTML = '<option value="">Failed to load</option>';
        }
    }

    function formatSize(bytes) {
        if (!bytes) return '?';
        const gb = bytes / (1024 ** 3);
        if (gb >= 1) return `${gb.toFixed(1)} GB`;
        const mb = bytes / (1024 ** 2);
        return `${mb.toFixed(0)} MB`;
    }

    function initModelSelector() {
        const select = document.getElementById('model-select');
        if (!select) return;

        select.addEventListener('change', async () => {
            const model = select.value;
            if (!model) return;

            select.classList.add('switching');
            showToast(`Switching to ${model}...`, 'info', 2000);

            try {
                const result = await api.post('/api/models/switch', { model });
                if (result.model_loaded) {
                    showToast(`Now using ${result.model_name}`, 'success');
                } else {
                    showToast(`Model set to ${result.model_name} (may need pulling)`, 'warning');
                }
                await checkStatus();
            } catch (e) {
                showToast('Failed to switch model: ' + e.message, 'error');
            } finally {
                select.classList.remove('switching');
            }
        });
    }

    // ─── Embedding Model Picker ───
    async function loadEmbeddingModels() {
        const select = document.getElementById('embed-model-select');
        if (!select) return;

        try {
            const data = await api.get('/api/embeddings/models');
            const models = data.models || [];
            const current = data.current_model || '';

            if (models.length === 0) {
                select.innerHTML = '<option value="">No models found</option>';
                return;
            }

            select.innerHTML = models.map(m => {
                const name = m.name;
                const size = formatSize(m.size);
                const badge = m.mode === 'sentence-transformers' ? ' [ST]' : '';
                const isSelected = name === current ||
                    name.split(':')[0] === current.split(':')[0];
                return `<option value="${name}" data-mode="${m.mode}" ${isSelected ? 'selected' : ''}>${name}${badge} (${size})</option>`;
            }).join('');

        } catch (e) {
            console.error('Failed to load embedding models:', e);
            select.innerHTML = '<option value="">Failed to load</option>';
        }
    }

    function initEmbeddingSelector() {
        const select = document.getElementById('embed-model-select');
        if (!select) return;

        select.addEventListener('change', async () => {
            const model = select.value;
            if (!model) return;

            const selectedOption = select.options[select.selectedIndex];
            const mode = selectedOption?.dataset?.mode || 'ollama';

            select.classList.add('switching');
            showToast(`Switching embeddings to ${model}...`, 'info', 3000);

            try {
                const result = await api.post('/api/embeddings/switch', { model, mode });
                if (result.success) {
                    const msg = result.needs_reindex
                        ? `Switched to ${result.model_name} (dim: ${result.dim}). Index rebuilt — re-scan references to re-index.`
                        : `Embeddings now using ${result.model_name}`;
                    showToast(msg, result.needs_reindex ? 'warning' : 'success', 5000);
                } else {
                    showToast('Switch failed: ' + (result.error || 'Unknown error'), 'error');
                }
                await checkStatus();
            } catch (e) {
                showToast('Failed to switch embedding model: ' + e.message, 'error');
            } finally {
                select.classList.remove('switching');
            }
        });
    }

    // ─── Init ───
    async function init() {
        initShortcuts();
        initModelSelector();
        initEmbeddingSelector();
        await checkStatus();
        await loadModels();
        await loadEmbeddingModels();
        await Sidebar.loadConversations();

        // Welcome hint clicks
        document.querySelectorAll('.hint-card').forEach(card => {
            card.addEventListener('click', () => {
                const hint = card.dataset.hint;
                if (hint) {
                    document.getElementById('message-input').value = hint;
                    Chat.handleSend();
                }
            });
        });

        // Periodic status check
        setInterval(checkStatus, 30000);

        // Scan references button
        document.getElementById('btn-scan-refs')?.addEventListener('click', async () => {
            showToast('Scanning references directory...', 'info');
            try {
                const result = await api.post('/api/references/scan');
                showToast(`Indexed ${result.indexed} documents`, 'success');
                await checkStatus();
            } catch (e) {
                showToast('Scan failed: ' + e.message, 'error');
            }
        });
    }

    // Start when DOM ready
    document.addEventListener('DOMContentLoaded', init);

    return {
        state,
        api,
        showToast,
        newChat,
        stopStreaming,
        checkStatus,
        loadModels,
        loadEmbeddingModels,
    };
})();

