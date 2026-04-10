/**
 * Knowledge Base controller.
 * Manages the directory-style view of indexed documents and chunks.
 */
const KnowledgeBase = (() => {
    // ─── State ───
    const state = {
        documents: [],
        expandedDocs: new Set(),
    };

    // ─── Elements ───
    const docListEl = () => document.getElementById('kb-document-list');
    const statsEl = () => document.getElementById('kb-stats');

    // ─── Load Data ───
    async function init() {
        await refresh();
    }

    async function refresh() {
        try {
            await Promise.all([
                loadStats(),
                loadDocuments()
            ]);
        } catch (e) {
            console.error('KB refresh failed:', e);
        }
    }

    async function loadStats() {
        try {
            const stats = await App.api.get('/api/documents/status');
            renderStats(stats);
        } catch (e) {
            console.error('Failed to load KB stats:', e);
        }
    }

    async function loadDocuments() {
        try {
            state.documents = await App.api.get('/api/documents');
            renderDocumentList();
        } catch (e) {
            console.error('Failed to load documents:', e);
        }
    }

    // ─── Rendering ───
    function renderStats(stats) {
        const el = statsEl();
        if (!el) return;

        el.innerHTML = `
            <div class="stat-item">
                <span class="stat-value">${stats.total_documents}</span>
                <span class="stat-label">Files</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${stats.total_chunks}</span>
                <span class="stat-label">Total Chunks</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${stats.embedding_model}</span>
                <span class="stat-label">Model</span>
            </div>
            <div class="stat-item">
                <span class="stat-value">${stats.embedding_dim}</span>
                <span class="stat-label">Dim</span>
            </div>
        `;
    }

    function renderDocumentList() {
        const el = docListEl();
        if (!el) return;

        if (state.documents.length === 0) {
            el.innerHTML = `<div style="text-align:center;padding:40px;color:var(--text-tertiary);">No documents indexed yet.</div>`;
            return;
        }

        el.innerHTML = state.documents.map(doc => renderDocItem(doc)).join('');

        // Bind events
        el.querySelectorAll('.kb-doc-header').forEach(header => {
            header.addEventListener('click', () => {
                const docId = header.parentElement.dataset.id;
                toggleDocument(docId);
            });
        });
    }

    function renderDocItem(doc) {
        const isExpanded = state.expandedDocs.has(doc.id);
        const size = formatBytes(doc.file_size);
        const date = new Date(doc.indexed_at).toLocaleDateString();

        return `
            <div class="kb-doc-item ${isExpanded ? 'expanded' : ''}" data-id="${doc.id}">
                <div class="kb-doc-header">
                    <div class="kb-doc-icon">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <path d="M13 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V9z"/>
                            <polyline points="13 2 13 9 20 9"/>
                        </svg>
                    </div>
                    <div class="kb-doc-info">
                        <span class="kb-doc-name">${esc(doc.filename)}</span>
                        <span class="kb-doc-path">${esc(doc.filepath)}</span>
                        <span class="kb-doc-meta">${doc.chunk_count} segments · ${size} · Indexed ${date}</span>
                    </div>
                    <div class="kb-doc-chevron">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="6 9 12 15 18 9"/>
                        </svg>
                    </div>
                </div>
                <div class="kb-chunks-list" id="chunks-${doc.id}">
                    ${isExpanded ? '<div class="loading-chunks">Loading segments...</div>' : ''}
                </div>
            </div>
        `;
    }

    async function toggleDocument(docId) {
        const item = document.querySelector(`.kb-doc-item[data-id="${docId}"]`);
        if (!item) return;

        if (state.expandedDocs.has(docId)) {
            state.expandedDocs.delete(docId);
            item.classList.remove('expanded');
        } else {
            state.expandedDocs.add(docId);
            item.classList.add('expanded');
            await loadChunks(docId);
        }
    }

    async function loadChunks(docId) {
        const listEl = document.getElementById(`chunks-${docId}`);
        if (!listEl) return;

        try {
            const data = await App.api.get(`/api/documents/${docId}/chunks`);
            renderChunks(listEl, data.chunks);
        } catch (e) {
            listEl.innerHTML = `<div class="error">Failed to load chunks: ${e.message}</div>`;
        }
    }

    function renderChunks(container, chunks) {
        if (!chunks || chunks.length === 0) {
            container.innerHTML = `<div style="color:var(--text-tertiary);font-size:0.8rem;">No chunks found in index for this file.</div>`;
            return;
        }

        container.innerHTML = chunks.map(c => `
            <div class="kb-chunk-card">
                <div class="kb-chunk-header">
                    <span class="kb-chunk-index">Segment #${c.chunk_index + 1}</span>
                    <span class="kb-chunk-length">${c.text.length} chars</span>
                </div>
                <div class="kb-chunk-text">${esc(c.text)}</div>
            </div>
        `).join('');
    }

    // ─── Helpers ───
    function formatBytes(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    }

    function esc(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    return {
        init,
        refresh,
    };
})();
