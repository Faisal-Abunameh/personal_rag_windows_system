/**
 * Chat UI controller.
 * Handles message rendering, tree-based branching, regenerate logic, and timing display.
 */
const Chat = (() => {
    const messagesEl = () => document.getElementById('messages');
    const inputEl = () => document.getElementById('message-input');
    const sendBtn = () => document.getElementById('btn-send');
    const stopBtn = () => document.getElementById('btn-stop');
    const welcomeEl = () => document.getElementById('welcome-screen');
    const chatContainer = () => document.getElementById('chat-container');

    let currentStreamEl = null;
    let streamedText = '';
    let currentConversation = null;
    let messageMap = {}; // msgId -> msgObject
    let childrenMap = {}; // parentId -> [childIds]

    function init() {
        const input = inputEl();
        const send = sendBtn();

        input?.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
            send.disabled = !input.value.trim();
        });

        input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!App.state.isStreaming && input.value.trim()) {
                    handleSend();
                }
            }
        });

        send?.addEventListener('click', handleSend);
        stopBtn()?.addEventListener('click', App.stopStreaming);

        const btnWebSearch = document.getElementById('btn-web-search');
        if (btnWebSearch) {
            btnWebSearch.addEventListener('click', () => {
                btnWebSearch.classList.toggle('active');
            });
        }
    }

    async function handleSend(userMsg = null, regenerateParentId = null) {
        if (App.state.isStreaming) return;

        const input = inputEl();
        let message = userMsg || input?.value.trim();
        const attachment = Upload.getAndClearAttachments()[0] || null;
        
        // If regenerating, we retrieve the content from the parent user message
        if (regenerateParentId && !message) {
            const parentMsg = messageMap[regenerateParentId];
            if (parentMsg) message = parentMsg.content;
        }

        if (!message && !attachment && !regenerateParentId) return;

        const webSearch = document.getElementById('btn-web-search')?.classList.contains('active') || false;

        if (welcomeEl()) welcomeEl().style.display = 'none';

        if (input && !regenerateParentId) {
            input.value = '';
            input.style.height = 'auto';
        }
        sendBtn().disabled = true;

        startStreaming();

        try {
            const resp = await App.api.streamChat(
                message,
                App.state.currentConversationId,
                attachment,
                webSearch,
                regenerateParentId // Specifically passing this as parent_id
            );

            if (!resp.ok) throw new Error(`Server error: ${resp.status}`);

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleStreamEvent(data);
                    } catch (e) {}
                }
            }
        } catch (e) {
            if (e.name === 'AbortError') {
                appendToStream('\n\n*[Generation stopped]*');
            } else {
                console.error('Stream error:', e);
                App.showToast('Error: ' + e.message, 'error');
                appendToStream('\n\n⚠️ ' + e.message);
            }
        } finally {
            endStreaming();
        }
    }

    function handleStreamEvent(data) {
        switch (data.type) {
            case 'token':
                appendToStream(data.content);
                break;
            case 'sources':
                addSources(data.sources);
                break;
            case 'info':
                addInfoMessage(data.content);
                break;
            case 'error':
                App.showToast(data.content, 'error');
                appendToStream('\n\n⚠️ ' + data.content);
                break;
            case 'done':
                if (data.conversation_id) {
                    App.state.currentConversationId = data.conversation_id;
                    Sidebar.loadConversations();
                    Sidebar.setActive(data.conversation_id);
                }
                // Refresh to update tree and add regenerate buttons/timing
                if (App.state.currentConversationId) {
                    App.api.get(`/api/conversations/${App.state.currentConversationId}`)
                        .then(conv => loadConversation(conv, data.message_id))
                        .catch(err => console.error("Sync failed:", err));
                }
                break;
        }
    }

    function startStreaming() {
        App.state.isStreaming = true;
        sendBtn()?.classList.add('hidden');
        stopBtn()?.classList.remove('hidden');

        const msgEl = createMessageElement('assistant', '');
        const bodyEl = msgEl.querySelector('.message-body');
        const typing = document.createElement('div');
        typing.className = 'typing-indicator';
        typing.innerHTML = '<div class="typing-dot"></div><div class="typing-dot"></div><div class="typing-dot"></div>';
        bodyEl.appendChild(typing);

        messagesEl()?.appendChild(msgEl);
        currentStreamEl = bodyEl;
        streamedText = '';
        scrollToBottom();
    }

    function appendToStream(text) {
        if (!currentStreamEl) return;
        const typing = currentStreamEl.querySelector('.typing-indicator');
        if (typing) typing.remove();
        streamedText += text;
        currentStreamEl.innerHTML = MarkdownRenderer.render(streamedText);
        scrollToBottom();
    }

    function endStreaming() {
        App.state.isStreaming = false;
        sendBtn()?.classList.remove('hidden');
        stopBtn()?.classList.add('hidden');
        currentStreamEl = null;
        streamedText = '';
        inputEl()?.focus();
    }

    function loadConversation(conv, targetLeafId = null) {
        currentConversation = conv;
        messageMap = {};
        childrenMap = {};
        
        conv.messages.forEach(m => {
            messageMap[m.id] = m;
            const pid = m.parent_id || 'root';
            if (!childrenMap[pid]) childrenMap[pid] = [];
            childrenMap[pid].push(m.id);
        });

        renderAll();
    }

    function renderAll() {
        const el = messagesEl();
        if (!el) return;
        el.innerHTML = '';

        if (!currentConversation || currentConversation.messages.length === 0) {
            showWelcome();
            return;
        }

        if (welcomeEl()) welcomeEl().style.display = 'none';

        // Recursive DFS traversal to linearize the tree for "Option A"
        function visit(pid) {
            const ids = childrenMap[pid] || [];
            // Sort siblings by creation time
            ids.sort((a, b) => {
                const msgA = messageMap[a];
                const msgB = messageMap[b];
                return new Date(msgA.created_at) - new Date(msgB.created_at);
            });

            ids.forEach(id => {
                const m = messageMap[id];
                const msgEl = createMessageElement(m.role, m.content, m);
                
                if (m.role === 'assistant' && m.generation_time) {
                    const header = msgEl.querySelector('.message-header');
                    const timeSpan = document.createElement('span');
                    timeSpan.className = 'processing-time';
                    timeSpan.textContent = `(${m.generation_time}s)`;
                    header.appendChild(timeSpan);
                }

                addMessageActions(msgEl, m.content, m.role, m.id);
                el.appendChild(msgEl);

                if (m.sources && m.sources.length > 0) {
                    addSourcesToEl(msgEl, m.sources);
                }

                // Visit children (this handles the "under each other" requirement for regenerations)
                visit(id);
            });
        }

        visit('root');
        scrollToBottom();
    }

    function createMessageElement(role, content, messageObj = null) {
        const msg = document.createElement('div');
        msg.className = 'message';
        if (messageObj) msg.dataset.id = messageObj.id;

        const avatarLabel = role === 'user' ? 'Y' : 'A';
        const avatarClass = role === 'user' ? 'user' : 'assistant';
        const roleLabel = role === 'user' ? 'You' : 'Local LLM';
        const bodyHtml = role === 'user'
            ? `<p>${MarkdownRenderer.escapeHtml(content)}</p>`
            : (content ? MarkdownRenderer.render(content) : '');

        msg.innerHTML = `
            <div class="message-header">
                <div class="message-avatar ${avatarClass}">${avatarLabel}</div>
                <span class="message-role">${roleLabel}</span>
            </div>
            <div class="message-body">${bodyHtml}</div>
        `;
        return msg;
    }

    function addMessageActions(msgEl, text, role, messageId = null) {
        const actions = document.createElement('div');
        actions.className = 'message-actions';
        
        let actionsHtml = `
            <button class="msg-action-btn copy-msg" title="Copy">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
                Copy
            </button>
        `;

        if (role === 'assistant' && messageId) {
            actionsHtml += `
                <button class="msg-action-btn regenerate" title="Regenerate">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                        <path d="M23 4v6h-6"/><path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                    </svg>
                    Regenerate
                </button>
            `;
        }

        actions.innerHTML = actionsHtml;
        actions.querySelector('.copy-msg')?.addEventListener('click', () => {
            navigator.clipboard.writeText(text);
            App.showToast('Copied to clipboard', 'success', 2000);
        });
        actions.querySelector('.regenerate')?.addEventListener('click', () => {
            const m = messageMap[messageId];
            if (m && m.parent_id) {
                // To regenerate assistant, we send null message + the original user message's ID as parent_id
                handleSend(null, m.parent_id);
            }
        });

        msgEl.appendChild(actions);
    }

    function addInfoMessage(text) {
        const el = document.createElement('div');
        el.className = 'info-message';
        el.textContent = text;
        messagesEl()?.appendChild(el);
        scrollToBottom();
    }

    function addSources(sources) {
        if (!sources || sources.length === 0) return;
        addSourcesToEl(null, sources);
    }

    function addSourcesToEl(parentEl, sources) {
        const container = document.createElement('div');
        container.className = 'sources-container';
        const toggle = document.createElement('button');
        toggle.className = 'sources-toggle';
        toggle.innerHTML = `<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="6 9 12 15 18 9"/></svg> ${sources.length} source${sources.length > 1 ? 's' : ''} found`;
        const list = document.createElement('div');
        list.className = 'sources-list';
        sources.forEach(s => {
            const chip = document.createElement('div');
            chip.className = 'source-chip';
            chip.innerHTML = `<div style="flex:1;min-width:0;"><div class="source-filename">${escHtml(s.filename)}</div><div class="source-text">${escHtml(s.chunk_text)}</div></div><span class="source-score">${(s.relevance_score * 100).toFixed(0)}%</span>`;
            list.appendChild(chip);
        });
        toggle.addEventListener('click', () => {
            toggle.classList.toggle('expanded');
            list.classList.toggle('visible');
        });
        container.appendChild(toggle);
        container.appendChild(list);
        if (parentEl) parentEl.appendChild(container);
        else messagesEl()?.appendChild(container);
        scrollToBottom();
    }

    function showWelcome() {
        const el = messagesEl();
        if (!el) return;
        el.innerHTML = `<div id="welcome-screen" class="welcome-screen">
            <h1 class="welcome-title">Local LLM</h1>
            <p id="welcome-subtitle" class="welcome-subtitle">Powered by Ollama &bull; FAISS &bull; Web Search</p>
            <div class="welcome-hints">
                <div class="hint-card" data-hint="Summarize my local documents">📄<span>Summarize docs</span></div>
                <div class="hint-card" data-hint="What is Apple's stock price today?">🌍<span>Stock Update</span></div>
                <div class="hint-card" data-hint="Analyze and compare the references">📊<span>Comparison</span></div>
            </div>
        </div>`;
        document.querySelectorAll('.hint-card').forEach(card => card.addEventListener('click', () => {
            inputEl().value = card.dataset.hint;
            handleSend();
        }));
    }

    function scrollToBottom() {
        const container = chatContainer();
        if (container) container.scrollTop = container.scrollHeight;
    }

    function escHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }

    function clear() {
        messagesEl() && (messagesEl().innerHTML = '');
        showWelcome();
        currentConversation = null;
        messageMap = {};
        childrenMap = {};
        activeLeafId = null;
    }

    document.addEventListener('DOMContentLoaded', init);

    return { init, handleSend, loadConversation, clear, endStreaming };
})();
