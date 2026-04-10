/**
 * Markdown rendering utilities.
 * Uses marked.js + highlight.js for rich content display.
 */
const MarkdownRenderer = (() => {
    // Configure marked
    const renderer = new marked.Renderer();

    // Add copy button to code blocks
    renderer.code = function (code, language) {
        const langClass = language ? `language-${language}` : '';
        const highlighted = language && hljs.getLanguage(language)
            ? hljs.highlight(code, { language }).value
            : escapeHtml(code);
        return `<pre><code class="${langClass}">${highlighted}</code><button class="code-copy-btn" onclick="MarkdownRenderer.copyCode(this)">Copy</button></pre>`;
    };

    // External links open in new tab
    renderer.link = function (href, title, text) {
        const titleAttr = title ? ` title="${title}"` : '';
        return `<a href="${href}"${titleAttr} target="_blank" rel="noopener">${text}</a>`;
    };

    marked.setOptions({
        renderer,
        gfm: true,
        breaks: true,
        smartypants: true,
    });

    function escapeHtml(text) {
        const div = document.createElement('div');
        div.textContent = text;
        return div.innerHTML;
    }

    function render(markdown) {
        if (!markdown) return '';
        try {
            return marked.parse(markdown);
        } catch (e) {
            console.error('Markdown parse error:', e);
            return `<p>${escapeHtml(markdown)}</p>`;
        }
    }

    function copyCode(btn) {
        const code = btn.parentElement.querySelector('code');
        if (!code) return;
        navigator.clipboard.writeText(code.textContent).then(() => {
            const orig = btn.textContent;
            btn.textContent = 'Copied!';
            setTimeout(() => { btn.textContent = orig; }, 1500);
        });
    }

    return { render, copyCode, escapeHtml };
})();
