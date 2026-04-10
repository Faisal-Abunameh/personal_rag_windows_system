/**
 * File upload and attachment handler.
 * Supports click-to-upload and drag-and-drop.
 */
const Upload = (() => {
    const fileInput = () => document.getElementById('file-input');
    const attachBtn = () => document.getElementById('btn-attach');
    const previewEl = () => document.getElementById('attachment-preview');

    const SUPPORTED = new Set([
        '.pdf', '.docx', '.doc', '.pptx', '.ppt', '.xlsx', '.xls',
        '.html', '.htm', '.csv', '.json', '.xml', '.txt', '.md',
        '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff',
        '.zip', '.epub',
    ]);

    function init() {
        // Click to attach
        attachBtn()?.addEventListener('click', () => fileInput()?.click());

        // File input change
        fileInput()?.addEventListener('change', (e) => {
            handleFiles(e.target.files);
            e.target.value = ''; // Reset for re-selecting same file
        });

        // Drag and drop
        initDragDrop();
    }

    function initDragDrop() {
        const overlay = document.createElement('div');
        overlay.className = 'drag-overlay';
        overlay.innerHTML = '<div class="drag-overlay-text">📎 Drop files to attach</div>';
        document.body.appendChild(overlay);

        let dragCounter = 0;

        document.addEventListener('dragenter', (e) => {
            e.preventDefault();
            dragCounter++;
            overlay.classList.add('active');
        });

        document.addEventListener('dragleave', (e) => {
            e.preventDefault();
            dragCounter--;
            if (dragCounter <= 0) {
                dragCounter = 0;
                overlay.classList.remove('active');
            }
        });

        document.addEventListener('dragover', (e) => {
            e.preventDefault();
        });

        document.addEventListener('drop', (e) => {
            e.preventDefault();
            dragCounter = 0;
            overlay.classList.remove('active');
            handleFiles(e.dataTransfer.files);
        });
    }

    function handleFiles(fileList) {
        if (!fileList || fileList.length === 0) return;

        for (const file of fileList) {
            const ext = '.' + file.name.split('.').pop().toLowerCase();
            if (!SUPPORTED.has(ext)) {
                App.showToast(`Unsupported file type: ${ext}`, 'warning');
                continue;
            }
            App.state.attachments.push(file);
        }

        renderPreview();
    }

    function renderPreview() {
        const el = previewEl();
        if (!el) return;

        if (App.state.attachments.length === 0) {
            el.classList.add('hidden');
            el.innerHTML = '';
            return;
        }

        el.classList.remove('hidden');
        el.innerHTML = App.state.attachments.map((file, i) => `
            <div class="attachment-chip">
                <span>📎 ${escHtml(file.name)}</span>
                <button class="remove-attach" data-index="${i}" title="Remove">✕</button>
            </div>
        `).join('');

        el.querySelectorAll('.remove-attach').forEach(btn => {
            btn.addEventListener('click', () => {
                App.state.attachments.splice(parseInt(btn.dataset.index), 1);
                renderPreview();
            });
        });
    }

    function getAndClearAttachments() {
        const files = [...App.state.attachments];
        App.state.attachments = [];
        renderPreview();
        return files;
    }

    function escHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    document.addEventListener('DOMContentLoaded', init);

    return {
        handleFiles,
        getAndClearAttachments,
    };
})();
