/**
 * Chat UI controller.
 * Handles message rendering, SSE streaming, sources, and auto-scroll.
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

    // ─── Init ───
    function init() {
        const input = inputEl();
        const send = sendBtn();

        // Auto-resize textarea
        input?.addEventListener('input', () => {
            input.style.height = 'auto';
            input.style.height = Math.min(input.scrollHeight, 200) + 'px';
            send.disabled = !input.value.trim();
        });

        // Send on Enter (without Shift)
        input?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                if (!App.state.isStreaming && input.value.trim()) {
                    handleSend();
                }
            }
        });

        // Send button
        send?.addEventListener('click', handleSend);

        // Stop button
        stopBtn()?.addEventListener('click', App.stopStreaming);
    }

    // ─── Send Message ───
    async function handleSend() {
        const input = inputEl();
        const message = input?.value.trim();
        if (!message || App.state.isStreaming) return;

        // Hide welcome screen
        const welcome = welcomeEl();
        if (welcome) welcome.style.display = 'none';

        // Get attachment
        const attachments = Upload.getAndClearAttachments();
        const attachment = attachments.length > 0 ? attachments[0] : null;

        // Add user message to UI
        addMessage('user', message);

        // Clear input
        input.value = '';
        input.style.height = 'auto';
        sendBtn().disabled = true;

        // Start streaming state
        startStreaming();

        try {
            const resp = await App.api.streamChat(
                message,
                App.state.currentConversationId,
                attachment
            );

            if (!resp.ok) {
                throw new Error(`Server error: ${resp.status}`);
            }

            // Read SSE stream
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });

                // Process complete SSE events
                const lines = buffer.split('\n');
                buffer = lines.pop(); // Keep incomplete line in buffer

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const data = JSON.parse(line.slice(6));
                        handleStreamEvent(data);
                    } catch (e) {
                        // Skip malformed events
                    }
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

    // ─── Handle Stream Events ───
    function handleStreamEvent(data) {
        switch (data.type) {
            case 'token':
                appendToStream(data.content);
                break;

            case 'sources':
                addSources(data.sources);
                break;

            case 'done':
                if (data.conversation_id) {
                    App.state.currentConversationId = data.conversation_id;
                    Sidebar.loadConversations();
                    Sidebar.setActive(data.conversation_id);
                }
                break;

            case 'info':
                addInfoMessage(data.content);
                break;

            case 'error':
                App.showToast(data.content, 'error');
                appendToStream('\n\n⚠️ ' + data.content);
                break;
        }
    }

    // ─── Streaming Controls ───
    function startStreaming() {
        App.state.isStreaming = true;
        sendBtn()?.classList.add('hidden');
        stopBtn()?.classList.remove('hidden');

        // Add typing indicator + assistant message shell
        const msgEl = createMessageElement('assistant', '');
        const bodyEl = msgEl.querySelector('.message-body');

        // Typing indicator
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

        // Remove typing indicator
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

        // Remove any remaining typing indicator
        if (currentStreamEl) {
            const typing = currentStreamEl.querySelector('.typing-indicator');
            if (typing) typing.remove();

            // Add copy button for the whole message
            const parent = currentStreamEl.closest('.message');
            if (parent) {
                addMessageActions(parent, streamedText);
            }
        }

        currentStreamEl = null;
        streamedText = '';
        inputEl()?.focus();
    }

    // ─── Message Rendering ───
    function addMessage(role, content) {
        const msgEl = createMessageElement(role, content);
        if (role === 'user') {
            addMessageActions(msgEl, content);
        }
        messagesEl()?.appendChild(msgEl);
        scrollToBottom();
    }

    function createMessageElement(role, content) {
        const msg = document.createElement('div');
        msg.className = 'message';

        const avatarLabel = role === 'user' ? 'You' : 'AI';
        const avatarClass = role === 'user' ? 'user' : 'assistant';
        const roleLabel = role === 'user' ? 'You' : 'Local LLM';
        const bodyHtml = role === 'user'
            ? `<p>${MarkdownRenderer.escapeHtml(content)}</p>`
            : (content ? MarkdownRenderer.render(content) : '');

        msg.innerHTML = `
            <div class="message-header">
                <div class="message-avatar ${avatarClass}">${avatarLabel.charAt(0)}</div>
                <span class="message-role">${roleLabel}</span>
            </div>
            <div class="message-body">${bodyHtml}</div>
        `;

        return msg;
    }

    function addMessageActions(msgEl, text) {
        const actions = document.createElement('div');
        actions.className = 'message-actions';
        actions.innerHTML = `
            <button class="msg-action-btn copy-msg" title="Copy">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2" ry="2"/>
                    <path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1"/>
                </svg>
                Copy
            </button>
        `;
        actions.querySelector('.copy-msg')?.addEventListener('click', () => {
            navigator.clipboard.writeText(text);
            App.showToast('Copied to clipboard', 'success', 2000);
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

        const container = document.createElement('div');
        container.className = 'sources-container';

        const toggle = document.createElement('button');
        toggle.className = 'sources-toggle';
        toggle.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="6 9 12 15 18 9"/>
            </svg>
            ${sources.length} source${sources.length > 1 ? 's' : ''} found
        `;

        const list = document.createElement('div');
        list.className = 'sources-list';

        sources.forEach(s => {
            const chip = document.createElement('div');
            chip.className = 'source-chip';
            chip.innerHTML = `
                <div style="flex:1;min-width:0;">
                    <div class="source-filename">${escHtml(s.filename)}</div>
                    <div class="source-text">${escHtml(s.chunk_text)}</div>
                </div>
                <span class="source-score">${(s.relevance_score * 100).toFixed(0)}%</span>
            `;
            list.appendChild(chip);
        });

        toggle.addEventListener('click', () => {
            toggle.classList.toggle('expanded');
            list.classList.toggle('visible');
        });

        container.appendChild(toggle);
        container.appendChild(list);
        messagesEl()?.appendChild(container);
        scrollToBottom();
    }

    // ─── Load Existing Conversation ───
    function loadConversation(conv) {
        clear();
        document.getElementById('welcome-screen').style.display = 'none';

        conv.messages.forEach(msg => {
            const el = createMessageElement(msg.role, msg.content);
            if (msg.sources && msg.sources.length > 0) {
                // Add sources after assistant message
                messagesEl()?.appendChild(el);
                addSources(msg.sources);
            } else {
                messagesEl()?.appendChild(el);
            }
        });

        scrollToBottom();
    }

    // ─── Utils ───
    function clear() {
        const el = messagesEl();
        if (!el) return;
        el.innerHTML = `<div id="welcome-screen" class="welcome-screen">
            <div class="welcome-icon">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="url(#gradient)" stroke-width="1.5">
                    <defs>
                        <linearGradient id="gradient" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stop-color="#a78bfa"/>
                            <stop offset="100%" stop-color="#6366f1"/>
                        </linearGradient>
                    </defs>
                    <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5"/>
                </svg>
            </div>
            <h1 class="welcome-title">Local LLM</h1>
            <p class="welcome-subtitle">Powered by Gemma 4 &bull; FAISS &bull; NeMo Retriever</p>
            <div class="welcome-hints">
                <div class="hint-card" data-hint="Summarize the key findings from my documents">
                    <span class="hint-icon">📄</span><span>Summarize documents</span>
                </div>
                <div class="hint-card" data-hint="What are the main topics covered in my references?">
                    <span class="hint-icon">🔍</span><span>Search knowledge base</span>
                </div>
                <div class="hint-card" data-hint="Compare and contrast the information in my uploaded files">
                    <span class="hint-icon">📊</span><span>Analyze & compare</span>
                </div>
                <div class="hint-card" data-hint="Help me draft a report based on my documents">
                    <span class="hint-icon">✍️</span><span>Draft from references</span>
                </div>
            </div>
        </div>`;
        // Re-bind hint clicks
        document.querySelectorAll('.hint-card').forEach(card => {
            card.addEventListener('click', () => {
                const hint = card.dataset.hint;
                if (hint) {
                    inputEl().value = hint;
                    handleSend();
                }
            });
        });
    }

    function scrollToBottom() {
        const container = chatContainer();
        if (!container) return;
        requestAnimationFrame(() => {
            container.scrollTop = container.scrollHeight;
        });
    }

    function escHtml(text) {
        const div = document.createElement('div');
        div.textContent = text || '';
        return div.innerHTML;
    }

    document.addEventListener('DOMContentLoaded', init);

    return {
        handleSend,
        loadConversation,
        clear,
        endStreaming,
        addMessage,
    };
})();
