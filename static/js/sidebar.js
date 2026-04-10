/**
 * Conversation sidebar controller.
 * Handles listing, searching, renaming, and deleting conversations.
 */

const Sidebar = (() => {
    const listEl = () => document.getElementById('conversation-list');
    const searchEl = () => document.getElementById('search-conversations');

    // ─── Load & Render ───
    async function loadConversations() {
        try {
            App.state.conversations = await App.api.get('/api/conversations');
            render();
        } catch (e) {
            console.error('Failed to load conversations:', e);
        }
    }

    function render(filter = '') {
        const el = listEl();
        if (!el) return;

        const filtered = filter
            ? App.state.conversations.filter(c =>
                c.title.toLowerCase().includes(filter.toLowerCase()))
            : App.state.conversations;

        if (filtered.length === 0) {
            el.innerHTML = `<div style="padding:20px;text-align:center;color:var(--text-tertiary);font-size:0.82rem;">
                ${filter ? 'No matching chats' : 'No conversations yet'}
            </div>`;
            return;
        }

        // Group by date
        const groups = groupByDate(filtered);
        let html = '';

        for (const [label, convs] of Object.entries(groups)) {
            html += `<div class="conv-group-label">${label}</div>`;
            for (const c of convs) {
                const isActive = c.id === App.state.currentConversationId;
                html += `
                    <div class="conv-item ${isActive ? 'active' : ''}" data-id="${c.id}">
                        <span class="conv-item-title">${escHtml(c.title)}</span>
                        <div class="conv-item-actions">
                            <button class="conv-action-btn rename" data-id="${c.id}" title="Rename">✏️</button>
                            <button class="conv-action-btn delete" data-id="${c.id}" title="Delete">🗑️</button>
                        </div>
                    </div>`;
            }
        }

        el.innerHTML = html;

        // Bind events
        el.querySelectorAll('.conv-item').forEach(item => {
            item.addEventListener('click', (e) => {
                if (e.target.closest('.conv-action-btn')) return;
                selectConversation(item.dataset.id);
            });
        });

        el.querySelectorAll('.conv-action-btn.rename').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                startRename(btn.dataset.id);
            });
        });

        el.querySelectorAll('.conv-action-btn.delete').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                deleteConversation(btn.dataset.id);
            });
        });
    }

    function groupByDate(conversations) {
        const now = new Date();
        const today = new Date(now.getFullYear(), now.getMonth(), now.getDate());
        const yesterday = new Date(today); yesterday.setDate(today.getDate() - 1);
        const weekAgo = new Date(today); weekAgo.setDate(today.getDate() - 7);
        const monthAgo = new Date(today); monthAgo.setDate(today.getDate() - 30);

        const groups = {};

        for (const c of conversations) {
            const d = new Date(c.updated_at);
            let label;
            if (d >= today) label = 'Today';
            else if (d >= yesterday) label = 'Yesterday';
            else if (d >= weekAgo) label = 'Previous 7 Days';
            else if (d >= monthAgo) label = 'Previous 30 Days';
            else label = 'Older';

            if (!groups[label]) groups[label] = [];
            groups[label].push(c);
        }

        return groups;
    }

    // ─── Select ───
    async function selectConversation(id) {
        App.state.currentConversationId = id;
        setActive(id);

        try {
            const conv = await App.api.get(`/api/conversations/${id}`);
            Chat.loadConversation(conv);
        } catch (e) {
            App.showToast('Failed to load conversation', 'error');
        }
    }

    function setActive(id) {
        listEl()?.querySelectorAll('.conv-item').forEach(item => {
            item.classList.toggle('active', item.dataset.id === id);
        });
    }

    // ─── Rename ───
    function startRename(id) {
        const item = listEl()?.querySelector(`.conv-item[data-id="${id}"]`);
        if (!item) return;

        const titleEl = item.querySelector('.conv-item-title');
        const currentTitle = titleEl.textContent;

        const input = document.createElement('input');
        input.className = 'conv-rename-input';
        input.value = currentTitle;
        titleEl.replaceWith(input);
        input.focus();
        input.select();

        const finish = async () => {
            const newTitle = input.value.trim() || currentTitle;
            input.replaceWith(titleEl);
            titleEl.textContent = newTitle;

            if (newTitle !== currentTitle) {
                try {
                    await App.api.put(`/api/conversations/${id}`, { title: newTitle });
                    await loadConversations();
                } catch (e) {
                    titleEl.textContent = currentTitle;
                    App.showToast('Rename failed', 'error');
                }
            }
        };

        input.addEventListener('blur', finish);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') input.blur();
            if (e.key === 'Escape') {
                input.value = currentTitle;
                input.blur();
            }
        });
    }

    // ─── Delete ───
    async function deleteConversation(id) {
        if (!confirm('Delete this conversation?')) return;
        try {
            await App.api.delete(`/api/conversations/${id}`);
            if (App.state.currentConversationId === id) {
                App.newChat();
            }
            await loadConversations();
            App.showToast('Conversation deleted', 'info');
        } catch (e) {
            App.showToast('Delete failed', 'error');
        }
    }

    // ─── Search ───
    function initSearch() {
        const el = searchEl();
        if (el) {
            el.addEventListener('input', () => render(el.value));
        }
    }

    // ─── Toggle Sidebar ───
    function initToggle() {
        const sidebar = document.getElementById('sidebar');
        const toggleBtn = document.getElementById('btn-toggle-sidebar');
        const openBtn = document.getElementById('btn-open-sidebar');
        const newChatBtn = document.getElementById('btn-new-chat');

        toggleBtn?.addEventListener('click', () => {
            sidebar.classList.add('collapsed');
            openBtn?.classList.remove('hidden');
        });

        openBtn?.addEventListener('click', () => {
            sidebar.classList.remove('collapsed');
            openBtn.classList.add('hidden');
        });

        newChatBtn?.addEventListener('click', App.newChat);
    }

    function escHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    // ─── Init ───
    document.addEventListener('DOMContentLoaded', () => {
        initSearch();
        initToggle();
    });

    return {
        loadConversations,
        render,
        setActive,
        selectConversation,
    };
})();
