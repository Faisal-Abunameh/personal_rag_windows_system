/**
 * Settings controller.
 * Manages .env configuration through the UI.
 */
const Settings = (() => {
    let config = {
        values: {},
        metadata: {}
    };

    const elements = {
        drawer: document.getElementById('settings-drawer'),
        content: document.getElementById('settings-content'),
        btnSettings: document.getElementById('btn-settings'),
        btnClose: document.getElementById('btn-close-settings'),
        btnSave: document.getElementById('btn-save-settings')
    };

    async function loadSettings() {
        try {
            // Fetch models and settings in parallel
            const [settings, llmModels, embedModels] = await Promise.all([
                App.api.get('/api/settings'),
                App.api.get('/api/models'),
                App.api.get('/api/embeddings/models')
            ]);

            config = settings;
            
            // Inject dynamic options into metadata
            if (config.metadata.LLM_MODEL) {
                config.metadata.LLM_MODEL.options = (llmModels.models || []).map(m => m.name);
            }
            if (config.metadata.EMBEDDING_MODEL) {
                config.metadata.EMBEDDING_MODEL.options = (embedModels.models || []).map(m => m.name);
            }

            renderSettings();
        } catch (e) {
            console.error('Failed to load settings:', e);
            App.showToast('Failed to load settings', 'error');
        }
    }

    function renderSettings() {
        const categories = {};

        // Group by category
        for (const [key, meta] of Object.entries(config.metadata)) {
            const cat = meta.category || 'General';
            if (!categories[cat]) categories[cat] = [];
            categories[cat].push({ key, ...meta, value: config.values[key] });
        }

        elements.content.innerHTML = Object.entries(categories).map(([name, items]) => `
            <div class="settings-group">
                <div class="settings-group-title">${name}</div>
                ${items.map(item => renderSettingItem(item)).join('')}
            </div>
        `).join('');

        // Attach event listeners for real-time slider sync
        attachInputListeners();
    }

    function renderSettingItem(item) {
        const isNumeric = item.type === 'int' || item.type === 'float';
        const showRecommended = item.recommended !== undefined && !item.hide_recommended;
        const recommended = showRecommended ? `<span class="setting-recommended">Recommended: ${item.recommended}</span>` : '';

        let control = '';
        if (isNumeric) {
            const percent = ((item.value - (item.min || 0)) / ((item.max || 100) - (item.min || 0)) * 100);
            control = `
                <div class="slider-container" style="--percent: ${percent}%">
                    <div class="setting-control-row">
                        <input type="range" 
                               class="volume-slider" 
                               id="setting-slider-${item.key}" 
                               value="${item.value}" 
                               min="${item.min || 0}" 
                               max="${item.max || 100}" 
                               step="${item.step || 1}">
                        <div class="value-bubble">${item.value}</div>
                        <input type="number" 
                               class="setting-input-number" 
                               id="setting-input-${item.key}" 
                               value="${item.value}"
                               step="${item.step || 1}">
                    </div>
                </div>
            `;
        } else if (item.type === 'bool') {
            control = `
                <select class="setting-input-text" id="setting-input-${item.key}">
                    <option value="true" ${String(item.value) === 'true' ? 'selected' : ''}>Enabled</option>
                    <option value="false" ${String(item.value) === 'false' ? 'selected' : ''}>Disabled</option>
                </select>
            `;
        } else if (item.type === 'choice' || item.type === 'select') {
            control = `
                <select class="setting-input-text" id="setting-input-${item.key}">
                    ${(item.options || []).map(opt => `<option value="${opt}" ${opt === item.value ? 'selected' : ''}>${opt}</option>`).join('')}
                </select>
            `;
        } else {
            control = `<input type="text" class="setting-input-text" id="setting-input-${item.key}" value="${item.value || ''}">`;
        }

        return `
            <div class="setting-item">
                <div class="setting-label-row">
                    <label class="setting-label" for="setting-input-${item.key}">${item.key.replace(/_/g, ' ')}</label>
                    ${recommended}
                </div>
                ${control}
                <div class="setting-description">${item.description || ''}</div>
            </div>
        `;
    }

    function attachInputListeners() {
        for (const [key, meta] of Object.entries(config.metadata)) {
            const isNumeric = meta.type === 'int' || meta.type === 'float';
            if (isNumeric) {
                const slider = document.getElementById(`setting-slider-${key}`);
                const input = document.getElementById(`setting-input-${key}`);

                slider.addEventListener('input', () => {
                    input.value = slider.value;
                    const container = slider.closest('.slider-container');
                    const bubble = container.querySelector('.value-bubble');
                    const min = parseFloat(meta.min ?? 0);
                    const max = parseFloat(meta.max ?? 100);
                    const percent = ((slider.value - min) / (max - min)) * 100;
                    container.style.setProperty('--percent', `${percent}%`);
                    bubble.textContent = slider.value;
                });

                input.addEventListener('change', () => {
                    let val = parseFloat(input.value);
                    const min = parseFloat(meta.min ?? 0);
                    const max = parseFloat(meta.max ?? 100);

                    if (val < min) val = min;
                    if (val > max) val = max;

                    input.value = val;
                    slider.value = val;
                    
                    const container = slider.closest('.slider-container');
                    const bubble = container.querySelector('.value-bubble');
                    const percent = ((val - min) / (max - min)) * 100;
                    container.style.setProperty('--percent', `${percent}%`);
                    bubble.textContent = val;
                });
            }
        }
    }

    async function saveSettings() {
        elements.btnSave.disabled = true;
        elements.btnSave.textContent = 'Saving...';

        let successCount = 0;
        let totalCount = 0;

        try {
            for (const [key, meta] of Object.entries(config.metadata)) {
                const input = document.getElementById(`setting-input-${key}`);
                if (!input) continue;

                const newValue = input.value;
                const oldValue = String(config.values[key]);

                if (newValue !== oldValue) {
                    totalCount++;
                    const result = await App.api.post('/api/settings', { key, value: newValue });
                    if (result.success) successCount++;
                }
            }

            if (totalCount > 0) {
                if (successCount === totalCount) {
                    App.showToast('Settings saved successfully', 'success');
                    await loadSettings(); // Refresh local state
                } else {
                    App.showToast(`Saved ${successCount}/${totalCount} settings`, 'warning');
                }
            } else {
                App.showToast('No changes detected', 'info');
            }
        } catch (e) {
            console.error('Save failed:', e);
            App.showToast('Failed to save some settings', 'error');
        } finally {
            elements.btnSave.disabled = false;
            elements.btnSave.textContent = 'Save Changes';
            elements.drawer.classList.add('hidden');
        }
    }

    function toggle() {
        const isHidden = elements.drawer.classList.contains('hidden');
        if (isHidden) {
            loadSettings();
            elements.drawer.classList.remove('hidden');
        } else {
            elements.drawer.classList.add('hidden');
        }
    }

    // Init
    elements.btnSettings.addEventListener('click', toggle);
    elements.btnClose.addEventListener('click', () => elements.drawer.classList.add('hidden'));
    elements.btnSave.addEventListener('click', saveSettings);

    return {
        loadSettings,
        toggle
    };
})();
