class Textbox {
    constructor(identifier, options = {}) {
        this.identifier = identifier;
        this.options = {
            align: 'left',                // left, right, center, justify-full, justify-left, justify-center, justify-right
            break: 'normal',              // normal, hyphen, no, character-wrap, controlled
            lineHeight: 1.1,
            markdown: false,
            content: ''
        };
        this.setOptions(options);

        this.wrapper = Q('<div>', { class: 'textbox_wrapper' }).get(0);
        this.wrapper.id = identifier;
        this.contentEl = Q('<div>', { class: 'textbox_content' }).get(0);
        this.wrapper.appendChild(this.contentEl);

        this.applyTypography();
        this.renderContent();
    }

    setOptions(opts = {}) {
        if (!opts || typeof opts !== 'object') return;
        // normalize keys from schema (line-height may come with hyphen)
        const normalized = { ...opts };
        if (normalized['line-height'] !== undefined) normalized.lineHeight = normalized['line-height'];
        if (normalized['break'] !== undefined) normalized.break = normalized['break'];
        this.options = { ...this.options, ...normalized };
    }

    applyTypography() {
        // Reset classes
        this.contentEl.className = 'textbox_content';

        // Alignment classes
        const alignMap = {
            'left': 'align-left',
            'right': 'align-right',
            'center': 'align-center',
            'justify-full': 'align-justify-full',
            'justify-left': 'align-justify-left',
            'justify-center': 'align-justify-center',
            'justify-right': 'align-justify-right'
        };
        const alignClass = alignMap[this.options.align] || 'align-left';
        Q(this.contentEl).addClass(alignClass);

        // Breaking classes
        const breakMap = {
            'normal': 'break-normal',
            'hyphen': 'break-hyphen',
            'no': 'break-no',
            'character-wrap': 'break-character-wrap',
            'controlled': 'break-controlled'
        };
        const breakClass = breakMap[this.options.break] || 'break-normal';
        Q(this.contentEl).addClass(breakClass);

        // Line height
        if (this.options.lineHeight) {
            Q(this.contentEl).css('line-height', String(this.options.lineHeight));
        }
    }

    escapeHtml(str) {
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    // Minimal markdown renderer for the supported features only
    renderMarkdown(md) {
        if (!md) return '';

        // Normalize newlines
        md = md.replace(/\r\n?/g, '\n');

        // Handle fenced code blocks first
        const codeBlocks = [];
        md = md.replace(/```([a-zA-Z0-9_-]*)\n([\s\S]*?)```/g, (m, lang, code) => {
            const idx = codeBlocks.length;
            codeBlocks.push({ lang: lang || '', code });
            return `{{{CODEBLOCK_${idx}}}}`;
        });

        // Tables (simple): header | header\n|---|---|\nrows...
        const lines = md.split('\n');
        const out = [];
        let i = 0;
        while (i < lines.length) {
            const line = lines[i];
            const next = lines[i + 1] || '';
            if (/^\s*\|?(.+\|.+)\|?\s*$/.test(line) && /^\s*\|?\s*:?-{3,}.*\|.*-+:?\s*\|?\s*$/.test(next)) {
                // parse table
                const headers = line.trim().replace(/^\||\|$/g, '').split('|').map(s => s.trim());
                i += 2; // skip header + separator
                const rows = [];
                while (i < lines.length && /^\s*\|?(.+\|.+)\|?\s*$/.test(lines[i])) {
                    const row = lines[i].trim().replace(/^\||\|$/g, '').split('|').map(s => s.trim());
                    rows.push(row);
                    i++;
                }
                // build table html
                let tableHtml = '<table class="textbox_table"><thead><tr>';
                headers.forEach(h => { tableHtml += `<th>${this.inlineMarkdown(h)}</th>`; });
                tableHtml += '</tr></thead><tbody>';
                rows.forEach(r => {
                    tableHtml += '<tr>' + r.map(c => `<td>${this.inlineMarkdown(c)}</td>`).join('') + '</tr>';
                });
                tableHtml += '</tbody></table>';
                out.push(tableHtml);
                continue;
            }

            // Horizontal rule
            if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
                out.push('<hr>');
                i++;
                continue;
            }

            // Headings #..######
            const h = line.match(/^(#{1,6})\s+(.*)$/);
            if (h) {
                const level = h[1].length;
                out.push(`<h${level}>${this.inlineMarkdown(h[2])}</h${level}>`);
                i++;
                continue;
            }

            // Blockquote
            if (/^>\s+/.test(line)) {
                const quote = line.replace(/^>\s+/, '');
                out.push(`<blockquote>${this.inlineMarkdown(quote)}</blockquote>`);
                i++;
                continue;
            }

            // Lists (ordered / unordered)
            if (/^\s*([-*])\s+/.test(line)) {
                const items = [];
                while (i < lines.length && /^\s*([-*])\s+/.test(lines[i])) {
                    items.push(lines[i].replace(/^\s*[-*]\s+/, ''));
                    i++;
                }
                out.push('<ul>' + items.map(it => `<li>${this.inlineMarkdown(it)}</li>`).join('') + '</ul>');
                continue;
            }
            if (/^\s*\d+\.\s+/.test(line)) {
                const items = [];
                while (i < lines.length && /^\s*\d+\.\s+/.test(lines[i])) {
                    items.push(lines[i].replace(/^\s*\d+\.\s+/, ''));
                    i++;
                }
                out.push('<ol>' + items.map(it => `<li>${this.inlineMarkdown(it)}</li>`).join('') + '</ol>');
                continue;
            }

            // Paragraph or blank
            if (line.trim() === '') {
                out.push('');
            } else {
                out.push(`<p>${this.inlineMarkdown(line)}</p>`);
            }
            i++;
        }

        let html = out.join('\n');
        // Replace code blocks placeholders
        html = html.replace(/\{\{\{CODEBLOCK_(\d+)\}\}\}/g, (m, idx) => {
            const cb = codeBlocks[Number(idx)];
            const codeEsc = this.escapeHtml(cb.code);
            const langClass = cb.lang ? ` class="language-${cb.lang}"` : '';
            return `<pre><code${langClass}>${codeEsc}</code></pre>`;
        });
        return html;
    }

    inlineMarkdown(text) {
        if (!text) return '';
        // Escape first, then re-insert inline code blocks unescaped inside <code>
        // Handle inline code `...`
        const codeSpans = [];
        text = text.replace(/`([^`]+)`/g, (m, code) => {
            const i = codeSpans.length;
            codeSpans.push(code);
            return `{{{C_${i}}}}`;
        });
        let s = this.escapeHtml(text);
        // Images ![alt](url)
        s = s.replace(/!\[([^\]]*)\]\(([^\)]+)\)/g, '<img alt="$1" src="$2">');
        // Links [text](url)
        s = s.replace(/\[([^\]]+)\]\(([^\)]+)\)/g, '<a href="$2" target="_blank" rel="noopener noreferrer">$1<\/a>');
        // Bold+italic ***text***
        s = s.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1<\/em><\/strong>');
        // Bold **text**
        s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1<\/strong>');
        // Italic *text*
        s = s.replace(/\*([^*]+)\*/g, '<em>$1<\/em>');
        // Strikethrough ~~text~~
        s = s.replace(/~~([^~]+)~~/g, '<del>$1<\/del>');
        // Put back inline code
        s = s.replace(/\{\{\{C_(\d+)\}\}\}/g, (m, idx) => `<code>${this.escapeHtml(codeSpans[Number(idx)])}<\/code>`);
        return s;
    }

    renderContent() {
        const content = this.options.content || '';
        if (this.options.markdown) {
            const html = this.renderMarkdown(content);
            this.contentEl.innerHTML = html;
        } else {
            this.contentEl.textContent = content;
        }
    }

    // Only set is required; supports string (content) or object (options update)
    set(valueOrOptions) {
        if (typeof valueOrOptions === 'string') {
            this.options.content = valueOrOptions;
        } else if (valueOrOptions && typeof valueOrOptions === 'object') {
            this.setOptions(valueOrOptions);
            this.applyTypography();
        }
        this.renderContent();
    }

    getElement() {
        return this.wrapper;
    }
}
