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

        streamChat(message, conversationId, attachment) {
            const formData = new FormData();
            formData.append('message', message);
            if (attachment) formData.append('attachment', attachment);

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

    // ─── Init ───
    async function init() {
        initShortcuts();
        await checkStatus();
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
    };
})();
