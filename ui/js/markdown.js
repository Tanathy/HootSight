(function(){
  const root = window;
  const hs = root.Hootsight || (root.Hootsight = {});

  function escapeHtml(value){
    return value
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function sanitizeUrl(raw){
    if(!raw) return '#';
    const trimmed = raw.trim();
    if(!trimmed) return '#';
    if(/^https?:\/\//i.test(trimmed)) return trimmed;
    if(trimmed.startsWith('#') || trimmed.startsWith('/') || trimmed.startsWith('mailto:') || trimmed.startsWith('tel:')){
      return trimmed;
    }
    return '#';
  }

  function formatInline(text, options = {}){
    if(typeof text !== 'string' || text.length === 0){
      return '';
    }
    const { skipLinks = false } = options;

    const codeSpans = [];
    let working = text.replace(/`([^`]+)`/g, (match, code) => {
      const token = `@@CODE${codeSpans.length}@@`;
      codeSpans.push(`<code>${escapeHtml(code)}</code>`);
      return token;
    });

    working = escapeHtml(working);

    working = working.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    working = working.replace(/__([^_]+)__/g, '<strong>$1</strong>');
    working = working.replace(/\*([^*]+)\*/g, '<em>$1</em>');
    working = working.replace(/_([^_]+)_/g, '<em>$1</em>');
    working = working.replace(/~~([^~]+)~~/g, '<del>$1</del>');

    if(!skipLinks){
      working = working.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
        const safeUrl = sanitizeUrl(url);
        const labelHtml = formatInline(label, { skipLinks: true });
        return `<a href="${safeUrl}" target="_blank" rel="noopener noreferrer">${labelHtml}</a>`;
      });
    }

    codeSpans.forEach((html, index) => {
      const token = `@@CODE${index}@@`;
      working = working.replace(new RegExp(token, 'g'), html);
    });

    return working;
  }

  function renderMarkdown(markdown){
    if(typeof markdown !== 'string'){
      return '';
    }

    const normalized = markdown.replace(/\r\n/g, '\n');
    const lines = normalized.split('\n');
    const html = [];
    let inUnorderedList = false;
    let inOrderedList = false;
    let inCodeBlock = false;
    const codeLines = [];

    function closeLists(){
      if(inUnorderedList){
        html.push('</ul>');
        inUnorderedList = false;
      }
      if(inOrderedList){
        html.push('</ol>');
        inOrderedList = false;
      }
    }

    lines.forEach((line) => {
      const trimmed = line.trim();

      if(/^```/.test(trimmed)){
        if(inCodeBlock){
          html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
          codeLines.length = 0;
          inCodeBlock = false;
        } else {
          closeLists();
          inCodeBlock = true;
          codeLines.length = 0;
        }
        return;
      }

      if(inCodeBlock){
        codeLines.push(line);
        return;
      }

      if(trimmed.length === 0){
        closeLists();
        html.push('<div class="md-gap"></div>');
        return;
      }

      const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
      if(headingMatch){
        closeLists();
        const level = headingMatch[1].length;
        const content = formatInline(headingMatch[2]);
        html.push(`<h${level}>${content}</h${level}>`);
        return;
      }

      const orderedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
      if(orderedMatch){
        if(inUnorderedList){
          html.push('</ul>');
          inUnorderedList = false;
        }
        if(!inOrderedList){
          html.push('<ol>');
          inOrderedList = true;
        }
        html.push(`<li>${formatInline(orderedMatch[2])}</li>`);
        return;
      }

      const unorderedMatch = trimmed.match(/^[-*+]\s+(.*)$/);
      if(unorderedMatch){
        if(inOrderedList){
          html.push('</ol>');
          inOrderedList = false;
        }
        if(!inUnorderedList){
          html.push('<ul>');
          inUnorderedList = true;
        }
        html.push(`<li>${formatInline(unorderedMatch[1])}</li>`);
        return;
      }

      const quoteMatch = trimmed.match(/^>\s+(.*)$/);
      if(quoteMatch){
        closeLists();
        html.push(`<blockquote>${formatInline(quoteMatch[1])}</blockquote>`);
        return;
      }

      closeLists();
      html.push(`<p>${formatInline(trimmed)}</p>`);
    });

    if(inCodeBlock){
      html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
    }
    closeLists();

    return html.join('');
  }

  hs.markdown = {
    render: renderMarkdown
  };
})();
