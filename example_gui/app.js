/**
 * SharePoint Chatbot - Dual Panel JavaScript
 * Handles two completely independent chat panels
 */

// ============================================================================
// Panel State Management - Each panel has its own isolated state
// ============================================================================

const panelState = {
    left: {
        threadId: null,
        messages: [],
        isLoading: false
    },
    right: {
        threadId: null,
        messages: [],
        isLoading: false
    }
};

// Global data
let availableModels = [];
let availableProfiles = [];

// ============================================================================
// Initialization
// ============================================================================

document.addEventListener('DOMContentLoaded', async () => {
    console.log('🚀 Initializing Dual Chat Interface...');
    
    // Load models and profiles
    await Promise.all([
        loadModels(),
        loadProfiles()
    ]);
    
    // Create new threads for both panels
    await Promise.all([
        newThread('left'),
        newThread('right')
    ]);
    
    // Set up keyboard shortcuts
    setupKeyboardShortcuts();
    
    console.log('✅ Initialization complete');
});

// ============================================================================
// API Calls
// ============================================================================

async function loadModels() {
    try {
        const response = await fetch('/api/models');
        const data = await response.json();
        availableModels = data.models;
        
        // Populate both dropdowns
        ['left', 'right'].forEach(panel => {
            const select = document.getElementById(`model-${panel}`);
            select.innerHTML = availableModels.map(m => 
                `<option value="${m.id}">${m.name}</option>`
            ).join('');
        });
        
        console.log(`📦 Loaded ${availableModels.length} models`);
    } catch (error) {
        console.error('Failed to load models:', error);
        showError('left', 'Failed to load models');
        showError('right', 'Failed to load models');
    }
}

async function loadProfiles() {
    try {
        const response = await fetch('/api/profiles');
        const data = await response.json();
        availableProfiles = data.profiles;
        
        // Populate both dropdowns
        ['left', 'right'].forEach(panel => {
            const select = document.getElementById(`profile-${panel}`);
            select.innerHTML = availableProfiles.map(p => 
                `<option value="${p.id}" title="${p.description}">${p.name}</option>`
            ).join('');
        });
        
        console.log(`👤 Loaded ${availableProfiles.length} profiles`);
    } catch (error) {
        console.error('Failed to load profiles:', error);
        showError('left', 'Failed to load profiles');
        showError('right', 'Failed to load profiles');
    }
}

async function newThread(panel) {
    try {
        const response = await fetch('/api/thread/new', { method: 'POST' });
        const data = await response.json();
        
        panelState[panel].threadId = data.thread_id;
        panelState[panel].messages = [];
        
        // Clear messages area
        const messagesEl = document.getElementById(`messages-${panel}`);
        messagesEl.innerHTML = `
            <div class="welcome-message">
                <div class="welcome-icon">💬</div>
                <h3>Chat ${panel === 'left' ? 'A' : 'B'} Ready</h3>
                <p>New thread created. Select a model and profile, then start chatting!</p>
                <small class="thread-id">Thread: ${data.thread_id.substring(0, 8)}...</small>
            </div>
        `;
        
        // Clear inputs
        document.getElementById(`input-${panel}`).value = '';
        document.getElementById(`knowledge-${panel}`).value = '';
        document.getElementById(`additional-${panel}`).value = '';
        
        console.log(`🔄 New thread for ${panel}: ${data.thread_id}`);
    } catch (error) {
        console.error(`Failed to create thread for ${panel}:`, error);
        showError(panel, 'Failed to create new thread');
    }
}

async function sendMessage(panel) {
    const input = document.getElementById(`input-${panel}`);
    const message = input.value.trim();
    
    if (!message) return;
    if (panelState[panel].isLoading) return;
    
    const model = document.getElementById(`model-${panel}`).value;
    const profile = document.getElementById(`profile-${panel}`).value;
    const knowledge = document.getElementById(`knowledge-${panel}`).value.trim();
    const additional = document.getElementById(`additional-${panel}`).value.trim();
    
    if (!model || !profile) {
        showError(panel, 'Please select a model and profile first');
        return;
    }
    
    // Ensure we have a thread
    if (!panelState[panel].threadId) {
        await newThread(panel);
    }
    
    // Show user message
    addMessage(panel, 'user', message);
    input.value = '';
    
    // Show loading
    panelState[panel].isLoading = true;
    const loadingId = showLoading(panel);
    
    try {
        const response = await fetch('/api/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: message,
                thread_id: panelState[panel].threadId,
                model: model,
                profile: profile,
                knowledge_history: knowledge,
                additional_info: additional
            })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP ${response.status}: ${response.statusText}`);
        }
        
        const data = await response.json();
        
        // Remove loading
        removeLoading(panel, loadingId);
        
        // Show assistant response
        addMessage(panel, 'assistant', data.response, data.sources);
        
        // Show token usage
        console.log(`📊 ${panel} tokens:`, data.token_usage);
        
    } catch (error) {
        console.error(`Chat error for ${panel}:`, error);
        removeLoading(panel, loadingId);
        showError(panel, `Error: ${error.message}`);
    } finally {
        panelState[panel].isLoading = false;
    }
}

// ============================================================================
// UI Helpers
// ============================================================================

function addMessage(panel, role, content, sources = null) {
    const messagesEl = document.getElementById(`messages-${panel}`);
    
    // Remove welcome message if present
    const welcome = messagesEl.querySelector('.welcome-message');
    if (welcome) welcome.remove();
    
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${role}`;
    
    const avatar = role === 'user' ? '👤' : '🤖';
    const roleLabel = role === 'user' ? 'You' : 'Assistant';
    
    // Format content with basic markdown support
    const formattedContent = formatMarkdown(content);
    
    let html = `
        <div class="message-header">
            <span class="message-avatar">${avatar}</span>
            <span class="message-role">${roleLabel}</span>
            <span class="message-time">${new Date().toLocaleTimeString()}</span>
        </div>
        <div class="message-content">${formattedContent}</div>
    `;
    
    // Add sources if available
    if (sources && sources.length > 0) {
        html += `
            <div class="message-sources">
                <span class="sources-label">📚 Sources:</span>
                <div class="sources-list">
                    ${sources.map(s => `
                        <button class="source-btn" onclick="openDocument('${s.md_file || ''}', '${escapeHtml(s.name)}')" 
                                ${!s.md_file ? 'disabled' : ''}>
                            ${escapeHtml(s.name)}
                        </button>
                    `).join('')}
                </div>
            </div>
        `;
    }
    
    messageDiv.innerHTML = html;
    messagesEl.appendChild(messageDiv);
    
    // Scroll to bottom
    messagesEl.scrollTop = messagesEl.scrollHeight;
    
    // Store message
    panelState[panel].messages.push({ role, content, sources });
}

function showLoading(panel) {
    const messagesEl = document.getElementById(`messages-${panel}`);
    const loadingId = `loading-${Date.now()}`;
    
    const loadingDiv = document.createElement('div');
    loadingDiv.className = 'message message-loading';
    loadingDiv.id = loadingId;
    loadingDiv.innerHTML = `
        <div class="message-header">
            <span class="message-avatar">🤖</span>
            <span class="message-role">Assistant</span>
        </div>
        <div class="message-content">
            <div class="loading-dots">
                <span></span><span></span><span></span>
            </div>
            <span class="loading-text">Thinking...</span>
        </div>
    `;
    
    messagesEl.appendChild(loadingDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
    
    return loadingId;
}

function removeLoading(panel, loadingId) {
    const loadingEl = document.getElementById(loadingId);
    if (loadingEl) loadingEl.remove();
}

function showError(panel, message) {
    const messagesEl = document.getElementById(`messages-${panel}`);
    
    const errorDiv = document.createElement('div');
    errorDiv.className = 'message message-error';
    errorDiv.innerHTML = `
        <div class="message-header">
            <span class="message-avatar">⚠️</span>
            <span class="message-role">Error</span>
        </div>
        <div class="message-content">${escapeHtml(message)}</div>
    `;
    
    messagesEl.appendChild(errorDiv);
    messagesEl.scrollTop = messagesEl.scrollHeight;
}

// ============================================================================
// Document Viewer
// ============================================================================

// Cache for available documents (for link resolution)
let availableDocuments = null;

async function loadAvailableDocuments() {
    if (availableDocuments) return availableDocuments;
    
    try {
        const response = await fetch('/api/documents');
        const data = await response.json();
        availableDocuments = data.documents || [];
        return availableDocuments;
    } catch (error) {
        console.error('Failed to load documents list:', error);
        return [];
    }
}

function findMatchingDocument(linkText) {
    if (!availableDocuments || !linkText) return null;
    
    // Clean up the link text
    const cleanText = linkText.trim().toLowerCase();
    
    // Try to find a matching document
    for (const doc of availableDocuments) {
        const docName = doc.name.toLowerCase();
        
        // Exact match (ignoring case and underscores vs spaces)
        const normalizedDocName = docName.replace(/_/g, ' ').replace(/-/g, ' ');
        const normalizedLinkText = cleanText.replace(/_/g, ' ').replace(/-/g, ' ');
        
        if (normalizedDocName.includes(normalizedLinkText) || normalizedLinkText.includes(normalizedDocName)) {
            return doc.filename;
        }
        
        // Try matching key parts - "USV - Something" -> look for "Something" in doc name
        const linkParts = cleanText.split(/[-–—]/);
        const mainPart = linkParts.length > 1 ? linkParts.slice(1).join('-').trim() : cleanText;
        
        if (mainPart.length > 3 && normalizedDocName.includes(mainPart)) {
            return doc.filename;
        }
    }
    
    return null;
}

function linkifyDocumentReferences(html) {
    if (!availableDocuments || availableDocuments.length === 0) return html;
    
    // Pattern to match document references that aren't already links
    // Matches things like "USV - Something" or "Archived - USV - Something" on their own line or after certain tags
    const patterns = [
        // Lines that look like document references (USV, PSV, Archived, etc.)
        /(?<![">])((Archived\s*[-–—]\s*)?(USV|PSV|MSV|Communication)\s*[-–—]\s*[A-Za-z][A-Za-z0-9\s,.'()#\-–—]+?)(?=<\/p>|<\/li>|<br>|$)/gi,
        // Also match "Awards voting for Members#Jasmin" style
        /(?<![">])([A-Z][A-Za-z]+\s+[a-z]+\s+for\s+[A-Za-z]+(?:#[A-Za-z]+)?)(?=<\/p>|<\/li>|<br>|$)/gi,
    ];
    
    let result = html;
    
    patterns.forEach(pattern => {
        result = result.replace(pattern, (match, text) => {
            // Don't process if it's already inside an anchor tag
            if (match.includes('<a ') || match.includes('href=')) return match;
            
            const docFile = findMatchingDocument(text);
            if (docFile) {
                return `<a href="#" class="doc-link" onclick="openDocument('${docFile}', '${escapeHtml(text)}'); return false;">${text}</a>`;
            }
            return match;
        });
    });
    
    return result;
}

async function openDocument(filename, displayName) {
    if (!filename) {
        alert('Document not available');
        return;
    }
    
    try {
        // Load available documents for link resolution
        await loadAvailableDocuments();
        
        const response = await fetch(`/api/document/${filename}`);
        if (!response.ok) throw new Error('Document not found');
        
        const data = await response.json();
        
        // Set modal title
        document.getElementById('doc-modal-title').textContent = displayName;
        
        // Convert markdown to HTML - formatMarkdownFull handles image paths via formatInline
        let htmlContent = formatMarkdownFull(data.content);
        
        // Post-process: make document references clickable
        htmlContent = linkifyDocumentReferences(htmlContent);
        
        // Set modal body
        document.getElementById('doc-modal-body').innerHTML = htmlContent;
        
        // Show modal
        document.getElementById('doc-modal').classList.add('active');
        
    } catch (error) {
        console.error('Failed to load document:', error);
        alert(`Failed to load document: ${error.message}`);
    }
}

function closeDocModal(event) {
    if (event && event.target.id !== 'doc-modal') return;
    document.getElementById('doc-modal').classList.remove('active');
}

// ============================================================================
// Markdown Formatting - Robust tokenized approach
// ============================================================================

function escapeHtml(text) {
    if (typeof text !== 'string') return '';
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#39;');
}

function sanitizeUrl(raw) {
    if (!raw) return '#';
    const trimmed = raw.trim();
    if (!trimmed) return '#';
    
    // Block dangerous protocols (file://, javascript:, vbscript:, data:)
    if (/^(file|javascript|vbscript|data):/i.test(trimmed)) {
        return '#';
    }
    
    if (/^https?:\/\//i.test(trimmed)) return trimmed;
    if (trimmed.startsWith('#') || trimmed.startsWith('/') || trimmed.startsWith('mailto:') || trimmed.startsWith('tel:')) {
        return trimmed;
    }
    if (/^[A-Za-z0-9._~!$&'()*+,;=@\/-]+$/.test(trimmed)) {
        return trimmed;
    }
    return '#';
}

function formatInline(text, options = {}) {
    if (typeof text !== 'string' || text.length === 0) {
        return '';
    }
    const { skipLinks = false, baseImagePath = '/docs/' } = options;

    const imageTokens = [];
    const codeSpans = [];
    
    // Protect images first
    let working = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, (match, alt, url) => {
        const token = `@@IMG${imageTokens.length}@@`;
        let safeUrl = sanitizeUrl(url);
        
        // Add base path for relative URLs
        if (safeUrl && safeUrl !== '#' && !safeUrl.startsWith('http') && !safeUrl.startsWith('/')) {
            safeUrl = baseImagePath + safeUrl;
        }
        
        const altHtml = escapeHtml(alt || '');
        if (!safeUrl || safeUrl === '#') {
            imageTokens.push({ token, html: altHtml });
        } else {
            imageTokens.push({ token, html: `<img src="${safeUrl}" alt="${altHtml}" class="doc-image" loading="lazy">` });
        }
        return token;
    });

    // Protect inline code
    working = working.replace(/`([^`]+)`/g, (match, code) => {
        const token = `@@CODE${codeSpans.length}@@`;
        codeSpans.push(`<code>${escapeHtml(code)}</code>`);
        return token;
    });

    // Now escape HTML
    working = escapeHtml(working);

    // Bold and italic - asterisks only (underscores break filenames with _)
    working = working.replace(/\*\*\*([^*]+)\*\*\*/g, '<strong><em>$1</em></strong>');
    working = working.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
    working = working.replace(/\*([^*\n]+)\*/g, '<em>$1</em>');
    
    // Strikethrough
    working = working.replace(/~~([^~]+)~~/g, '<del>$1</del>');

    // Links (if not skipped)
    if (!skipLinks) {
        working = working.replace(/\[([^\]]+)\]\(([^)]+)\)/g, (match, label, url) => {
            const safeUrl = sanitizeUrl(url);
            if (!safeUrl || safeUrl === '#') {
                return formatInline(label, { skipLinks: true, baseImagePath });
            }
            const labelHtml = formatInline(label, { skipLinks: true, baseImagePath });
            const isExternal = /^https?:\/\//i.test(safeUrl);
            const attrs = [`href="${safeUrl}"`];
            if (isExternal) {
                attrs.push('target="_blank"', 'rel="noopener noreferrer"');
            }
            return `<a ${attrs.join(' ')}>${labelHtml}</a>`;
        });
    }

    // Restore protected code spans
    codeSpans.forEach((html, index) => {
        const token = `@@CODE${index}@@`;
        working = working.replace(new RegExp(token, 'g'), html);
    });

    // Restore protected images
    imageTokens.forEach(({ token, html }) => {
        working = working.replace(new RegExp(token, 'g'), html);
    });

    return working;
}

function isTableRow(line) {
    const trimmed = line.trim();
    return /^\|.+\|$/.test(trimmed) || /^\|.+\|.+/.test(trimmed);
}

function parseTableRow(line) {
    const trimmed = line.trim();
    const cells = trimmed.split('|').map(c => c.trim()).filter(c => c.length > 0);
    return cells;
}

function isTableSeparator(line) {
    const trimmed = line.trim();
    if (!/^\|[\s\-:|]+\|$/.test(trimmed)) return false;
    const cells = trimmed.split('|').map(c => c.trim()).filter(c => c.length > 0);
    return cells.every(c => /^:?-+:?$/.test(c));
}

function parseTableAlignment(line) {
    const trimmed = line.trim();
    const cells = trimmed.split('|').map(c => c.trim()).filter(c => c.length > 0);
    return cells.map(c => {
        if (c.startsWith(':') && c.endsWith(':')) return 'center';
        if (c.endsWith(':')) return 'right';
        if (c.startsWith(':')) return 'left';
        return 'left';
    });
}

function formatMarkdown(text) {
    // Full markdown for chat messages (AI responses can have lists, bold, etc.)
    if (!text) return '';
    
    // For simple one-liners without markdown, just escape and return
    const hasMarkdown = /[*_`#\-\[\]|>]/.test(text) || text.includes('\n');
    if (!hasMarkdown) {
        return escapeHtml(text);
    }
    
    // Use full markdown parser for complex content
    return formatMarkdownFull(text);
}

function formatMarkdownFull(markdown) {
    if (typeof markdown !== 'string') {
        return '';
    }

    const normalized = markdown.replace(/\r\n/g, '\n');
    const lines = normalized.split('\n');
    const html = [];
    let inUnorderedList = false;
    let inOrderedList = false;
    let inCodeBlock = false;
    let inTable = false;
    let tableAlignment = [];
    const codeLines = [];

    function closeLists() {
        if (inUnorderedList) {
            html.push('</ul>');
            inUnorderedList = false;
        }
        if (inOrderedList) {
            html.push('</ol>');
            inOrderedList = false;
        }
    }

    function closeTable() {
        if (inTable) {
            html.push('</tbody></table>');
            inTable = false;
            tableAlignment = [];
        }
    }

    lines.forEach((line, index) => {
        const trimmed = line.trim();

        // Code block toggle
        if (/^```/.test(trimmed)) {
            if (inCodeBlock) {
                closeTable();
                closeLists();
                html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
                codeLines.length = 0;
                inCodeBlock = false;
            } else {
                closeTable();
                closeLists();
                inCodeBlock = true;
                codeLines.length = 0;
            }
            return;
        }

        if (inCodeBlock) {
            codeLines.push(line);
            return;
        }

        // Empty line
        if (trimmed.length === 0) {
            closeTable();
            closeLists();
            return;
        }

        // Horizontal rule
        if (/^(-{3,}|\*{3,}|_{3,})$/.test(trimmed)) {
            closeTable();
            closeLists();
            html.push('<hr>');
            return;
        }

        // Table start
        if (isTableRow(trimmed) && !inTable) {
            const cells = parseTableRow(trimmed);
            const nextLine = index + 1 < lines.length ? lines[index + 1] : '';
            if (isTableSeparator(nextLine)) {
                tableAlignment = parseTableAlignment(nextLine);
                closeLists();
                html.push('<table class="doc-table"><thead><tr>');
                cells.forEach((cell, i) => {
                    const align = tableAlignment[i] || 'left';
                    html.push(`<th style="text-align:${align}">${formatInline(cell)}</th>`);
                });
                html.push('</tr></thead><tbody>');
                inTable = true;
                return;
            }
        }

        // Table row
        if (isTableRow(trimmed) && inTable) {
            const cells = parseTableRow(trimmed);
            html.push('<tr>');
            cells.forEach((cell, i) => {
                const align = tableAlignment[i] || 'left';
                html.push(`<td style="text-align:${align}">${formatInline(cell)}</td>`);
            });
            html.push('</tr>');
            return;
        }

        // Table separator (skip)
        if (isTableSeparator(trimmed)) {
            return;
        }

        // End table if not a table row
        if (inTable && !isTableRow(trimmed)) {
            closeTable();
        }

        // Headings
        const headingMatch = trimmed.match(/^(#{1,6})\s+(.*)$/);
        if (headingMatch) {
            closeTable();
            closeLists();
            const level = headingMatch[1].length;
            const content = formatInline(headingMatch[2]);
            html.push(`<h${level}>${content}</h${level}>`);
            return;
        }

        // Ordered list
        const orderedMatch = trimmed.match(/^(\d+)\.\s+(.*)$/);
        if (orderedMatch) {
            closeTable();
            if (inUnorderedList) {
                html.push('</ul>');
                inUnorderedList = false;
            }
            if (!inOrderedList) {
                html.push('<ol>');
                inOrderedList = true;
            }
            html.push(`<li>${formatInline(orderedMatch[2])}</li>`);
            return;
        }

        // Unordered list
        const unorderedMatch = trimmed.match(/^[-*+]\s+(.*)$/);
        if (unorderedMatch) {
            closeTable();
            if (inOrderedList) {
                html.push('</ol>');
                inOrderedList = false;
            }
            if (!inUnorderedList) {
                html.push('<ul>');
                inUnorderedList = true;
            }
            html.push(`<li>${formatInline(unorderedMatch[1])}</li>`);
            return;
        }

        // Blockquote
        const quoteMatch = trimmed.match(/^>\s+(.*)$/);
        if (quoteMatch) {
            closeTable();
            closeLists();
            html.push(`<blockquote>${formatInline(quoteMatch[1])}</blockquote>`);
            return;
        }

        // Regular paragraph
        closeTable();
        closeLists();
        html.push(`<p>${formatInline(trimmed)}</p>`);
    });

    // Close any remaining blocks
    if (inCodeBlock) {
        html.push(`<pre><code>${escapeHtml(codeLines.join('\n'))}</code></pre>`);
    }
    closeTable();
    closeLists();

    return html.join('');
}

// ============================================================================
// Keyboard Shortcuts
// ============================================================================

function setupKeyboardShortcuts() {
    // Enter to send (with panel detection)
    ['left', 'right'].forEach(panel => {
        const input = document.getElementById(`input-${panel}`);
        input.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                sendMessage(panel);
            }
        });
    });
    
    // Escape to close modal
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            closeDocModal();
        }
    });
}

// ============================================================================
// Export for debugging
// ============================================================================

window.panelState = panelState;
window.sendMessage = sendMessage;
window.newThread = newThread;
window.openDocument = openDocument;
window.closeDocModal = closeDocModal;
