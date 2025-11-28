class PromptField {
    constructor(identifier, value = "", title = "", description = "") {
        this.identifier = identifier;
        this.promptWrapper = Q('<div>', { class: 'prompt_wrapper' }).get(0);
        
        // History management (undo/redo)
        this.history = [];
        this.historyIndex = -1;
        this.maxHistory = 25;
        this.isUndoRedoAction = false;
        
        if (title) {
            const heading = Q('<h3>', { class: 'inputs_title', text: title }).get(0);
            this.promptWrapper.appendChild(heading);
        }
        if (description) {
            const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: description }).get(0);
            this.promptWrapper.appendChild(descriptionHeading);
        }
        
        // Action buttons container
        this.actionsContainer = Q('<div>', { class: 'prompt_actions' }).get(0);
        this.promptWrapper.appendChild(this.actionsContainer);
        this.setupActionButtons();
        
        this.promptContent = Q('<div>', { class: 'prompt_content' }).get(0);
        this.hiddenInput = Q('<input>', { class: 'hidden_input' }).get(0);
        this.hiddenInput.type = "hidden";
        this.hiddenInput.value = value;
        this.hiddenInput.setAttribute("id", identifier);
        
        this.promptWrapper.append(this.promptContent, this.hiddenInput);
        this.initializeTags(value);
        this.setupMainInput();
        this.setupDragAndDrop();
        this.setupKeyboardShortcuts();
        
        // Save initial state to history
        this.saveToHistory();
    }
    
    setupActionButtons() {
        // Copy to clipboard button
        const copyBtn = Q('<button>', { class: 'prompt_action_btn', 'data-tooltip': 'ui.clipboard.copy' }).get(0);
        copyBtn.innerHTML = window.UI_ICONS?.prompt?.copy || '📋';
        Q(copyBtn).on('click', () => this.copyToClipboard());
        
        // Paste from clipboard button
        const pasteBtn = Q('<button>', { class: 'prompt_action_btn', 'data-tooltip': 'ui.clipboard.paste' }).get(0);
        pasteBtn.innerHTML = window.UI_ICONS?.prompt?.paste || '📥';
        Q(pasteBtn).on('click', () => this.pasteFromClipboard());
        
        // Clear all button
        const clearBtn = Q('<button>', { class: 'prompt_action_btn prompt_action_danger', 'data-tooltip': 'ui.clipboard.clear' }).get(0);
        clearBtn.innerHTML = window.UI_ICONS?.prompt?.delete || '🗑️';
        Q(clearBtn).on('click', () => this.clearAll());
        
        // Undo button
        const undoBtn = Q('<button>', { class: 'prompt_action_btn', 'data-tooltip': 'ui.clipboard.undo' }).get(0);
        undoBtn.innerHTML = window.UI_ICONS?.prompt?.undo || '↶';
        Q(undoBtn).on('click', () => this.undo());
        this.undoBtn = undoBtn;
        
        // Redo button
        const redoBtn = Q('<button>', { class: 'prompt_action_btn', 'data-tooltip': 'ui.clipboard.redo' }).get(0);
        redoBtn.innerHTML = window.UI_ICONS?.prompt?.redo || '↷';
        Q(redoBtn).on('click', () => this.redo());
        this.redoBtn = redoBtn;
        
        this.actionsContainer.append(copyBtn, pasteBtn, clearBtn, undoBtn, redoBtn);
        this.updateActionButtons();
    }
    
    updateActionButtons() {
        // Update undo/redo button states
        if (this.undoBtn) {
            this.undoBtn.disabled = this.historyIndex <= 0;
        }
        if (this.redoBtn) {
            this.redoBtn.disabled = this.historyIndex >= this.history.length - 1;
        }
    }
    
    setupKeyboardShortcuts() {
        Q(this.promptContent).on('keydown', (e) => {
            // Only handle if prompt content is focused or a child is focused
            if (!this.promptContent.contains(document.activeElement)) return;
            
            // Ctrl+Z - Undo
            if (e.ctrlKey && e.key === 'z') {
                e.preventDefault();
                this.undo();
            }
            
            // Ctrl+Y - Redo
            if (e.ctrlKey && e.key === 'y') {
                e.preventDefault();
                this.redo();
            }
        });
    }
    
    saveToHistory() {
        if (this.isUndoRedoAction) return;
        
        const currentValue = this.get();
        
        // Don't save if same as current history state
        if (this.historyIndex >= 0 && this.history[this.historyIndex] === currentValue) {
            return;
        }
        
        // Remove any redo history if we're not at the end
        if (this.historyIndex < this.history.length - 1) {
            this.history = this.history.slice(0, this.historyIndex + 1);
        }
        
        // Add new state
        this.history.push(currentValue);
        
        // Limit history size
        if (this.history.length > this.maxHistory) {
            this.history.shift();
        } else {
            this.historyIndex++;
        }
        
        this.updateActionButtons();
    }
    
    undo() {
        if (this.historyIndex > 0) {
            this.historyIndex--;
            this.isUndoRedoAction = true;
            this.set(this.history[this.historyIndex]);
            this.isUndoRedoAction = false;
            this.updateActionButtons();
        }
    }
    
    redo() {
        if (this.historyIndex < this.history.length - 1) {
            this.historyIndex++;
            this.isUndoRedoAction = true;
            this.set(this.history[this.historyIndex]);
            this.isUndoRedoAction = false;
            this.updateActionButtons();
        }
    }
    
    async copyToClipboard() {
        const value = this.get();
        try {
            await navigator.clipboard.writeText(value);
            console.log('Prompt copied to clipboard');
        } catch (err) {
            console.error('Failed to copy to clipboard:', err);
        }
    }
    
    async pasteFromClipboard() {
        try {
            const text = await navigator.clipboard.readText();
            if (text) {
                this.set(text);
                console.log('Prompt pasted from clipboard');
            }
        } catch (err) {
            console.error('Failed to paste from clipboard:', err);
        }
    }
    
    clearAll() {
        if (confirm('Are you sure you want to clear all tags?')) {
            this.set('');
        }
    }
    
    /**
     * Parse egy prompt stringet tagekre
     * Felismeri: normál szavakat, (súlyozásokat), BREAK-eket, <lora:...> tageket
     */
    parsePrompt(promptText) {
        if (!promptText || !promptText.trim()) return [];
        
        const tags = [];
        let currentPos = 0;
        const text = promptText.trim();
        
        while (currentPos < text.length) {
            // Skip whitespace (de NE vessző - csak szóközöket)
            // Vesszők a tag separátorok!
            while (currentPos < text.length && /[ \t\r\n]/.test(text[currentPos])) {
                currentPos++;
            }
            
            // Ha vesszőt találtunk, ugrd át és folytass
            if (text[currentPos] === ',') {
                currentPos++;
                while (currentPos < text.length && /[ \t\r\n]/.test(text[currentPos])) {
                    currentPos++;
                }
                continue;
            }
            
            if (currentPos >= text.length) break;
            
            // BREAK check - speciális szó
            if (text.substring(currentPos, currentPos + 5).toUpperCase() === 'BREAK') {
                tags.push({
                    text: 'BREAK',
                    weight: 1.0,
                    type: 'break'
                });
                currentPos += 5;
                continue;
            }
            
            // LoRA/embedding/hypernet check: <lora:name> vagy <lora:name:weight>
            // Támogatott típusok: lora, lyco, ti, embedding, hypernet
            if (text[currentPos] === '<') {
                const endPos = text.indexOf('>', currentPos);
                if (endPos !== -1) {
                    const content = text.substring(currentPos + 1, endPos);
                    const parts = content.split(':');
                    
                    if (parts.length >= 2) {
                        const networkType = parts[0].toLowerCase(); // lora, lyco, ti, embedding, hypernet
                        const networkName = parts[1];
                        const networkWeight = parts.length >= 3 ? parseFloat(parts[2]) : 1.0;
                        
                        // Extra network típusok
                        const validTypes = ['lora', 'lyco', 'ti', 'embedding', 'hypernet'];
                        if (validTypes.includes(networkType)) {
                            tags.push({
                                text: `${networkType}:${networkName}`,
                                weight: networkWeight,
                                type: 'network',
                                networkType: networkType, // Melyik típusú network
                                fullText: text.substring(currentPos, endPos + 1)
                            });
                            currentPos = endPos + 1;
                            continue;
                        }
                    }
                }
            }
            
            // Zárójelezett súlyozás check: (text) vagy ((text)) vagy (text:1.3)
            // VAGY alternation check: {cat|dog}
            if (text[currentPos] === '(' || text[currentPos] === '{') {
                const openChar = text[currentPos];
                const closeChar = openChar === '(' ? ')' : '}';
                let depth = 0;
                let start = currentPos;
                
                // Számoljuk meg a nyitó zárójeleket
                while (currentPos < text.length && text[currentPos] === openChar) {
                    depth++;
                    currentPos++;
                }
                
                // Keressük meg a tartalmat és a megfelelő számú záró zárójelet
                let parenContent = '';
                let closingDepth = 0;
                let foundClosing = true;
                
                while (currentPos < text.length && closingDepth < depth) {
                    if (text[currentPos] === closeChar) {
                        closingDepth++;
                        if (closingDepth < depth) {
                            parenContent += text[currentPos];
                        }
                    } else {
                        parenContent += text[currentPos];
                    }
                    currentPos++;
                }
                
                // Ellenőrizzük, hogy minden zárójel le van-e zárva
                if (closingDepth !== depth) {
                    foundClosing = false;
                    currentPos = start + depth;
                    parenContent = '';
                    while (currentPos < text.length && 
                           text[currentPos] !== ',' && 
                           !['(', ')', '{', '}', '<'].includes(text[currentPos]) &&
                           !/\s/.test(text[currentPos])) {
                        parenContent += text[currentPos];
                        currentPos++;
                    }
                }
                
                // Ha van tartalom, dolgozzuk fel
                if (parenContent.trim()) {
                    // ALTERNATION check: {cat|dog} vagy {cat OR dog}
                    if (openChar === '{' && depth === 1 && (parenContent.includes('|') || parenContent.includes(' OR '))) {
                        // Ez egy alternation tag
                        tags.push({
                            text: parenContent.trim(),
                            weight: 1.0,
                            type: 'alternation'
                        });
                    } else {
                        // Súlyozás feldolgozása
                        this.processWeightedContent(parenContent, depth, tags);
                    }
                }
                continue;
            }
            
            // Tag a következő vesszőig (szóközöket is tartalmazhat!)
            // Ez az új megközelítés: csak a VESSZŐK a separátorok
            let tagEnd = currentPos;
            let depth = 0;
            
            while (tagEnd < text.length) {
                const char = text[tagEnd];
                
                // Nyomon követjük a zárójeleket
                if (char === '(' || char === '{' || char === '<') depth++;
                else if (char === ')' || char === '}' || char === '>') depth--;
                
                // Ha zárójel nélkül vagyunk és vesszőt találunk, akkor tag vége
                if (depth === 0 && char === ',') break;
                
                tagEnd++;
            }
            
            if (tagEnd > currentPos) {
                const rawTag = text.substring(currentPos, tagEnd).trim();
                
                if (rawTag) {
                    // Ellenőrizzük, hogy tartalmaz-e " OR " mintát
                    if (rawTag.includes(' OR ')) {
                        // Ez egy alternation tag (de kapcsos zárójel nélkül írták)
                        tags.push({
                            text: rawTag.replace(/\sOR\s/g, '|'), // Normalizálás | karakterre
                            weight: 1.0,
                            type: 'alternation'
                        });
                    } else {
                        tags.push({
                            text: rawTag,
                            weight: 1.0,
                            type: 'normal'
                        });
                    }
                }
                currentPos = tagEnd;
            } else {
                currentPos++;
            }
        }
        
        return tags;
    }
    
    /**
     * Súlyozott tartalom feldolgozása (helper a parsePrompt-hoz)
     */
    processWeightedContent(parenContent, depth, tags) {
        // Parse weight: (text:1.3) formátum
        const weightMatch = parenContent.match(/^(.+):(\d+(?:\.\d+)?)$/);
        if (weightMatch) {
            // Ha vesszővel van elválasztva több elem, azokat külön kezeljük
            const items = weightMatch[1].split(',').map(s => s.trim()).filter(Boolean);
            const explicitWeight = parseFloat(weightMatch[2]);
            
            items.forEach(item => {
                // Tisztítás: távolítsuk el a maradék zárójeleket
                const cleanItem = item.replace(/[(){}]/g, '').trim();
                if (cleanItem) {
                    tags.push({
                        text: cleanItem,
                        weight: explicitWeight,
                        type: 'emphasis',
                        originalDepth: depth
                    });
                }
            });
        } else {
            // Implicit súlyozás: () = 1.1, (()) = 1.21 (1.1^2)
            const implicitWeight = Math.pow(1.1, depth);
            
            // Ha vesszővel van elválasztva több elem, azokat külön kezeljük
            const items = parenContent.split(',').map(s => s.trim()).filter(Boolean);
            
            items.forEach(item => {
                // Tisztítás: távolítsuk el a maradék zárójeleket
                const cleanItem = item.replace(/[(){}]/g, '').trim();
                if (cleanItem) {
                    tags.push({
                        text: cleanItem,
                        weight: Math.round(implicitWeight * 100) / 100,
                        type: 'emphasis',
                        originalDepth: depth
                    });
                }
            });
        }
    }
    
    initializeTags(value) {
        if (value) {
            const parsedTags = this.parsePrompt(value);
            
            // Chain detection: csoportosítjuk az egymás melletti azonos súlyú tageket
            const processedTags = [];
            for (let i = 0; i < parsedTags.length; i++) {
                const tag = parsedTags[i];
                
                // Nézzük meg van-e előtte/utána azonos súlyú tag
                const prevTag = i > 0 ? parsedTags[i - 1] : null;
                const nextTag = i < parsedTags.length - 1 ? parsedTags[i + 1] : null;
                
                const hasChainBefore = prevTag && 
                    prevTag.weight === tag.weight && 
                    prevTag.type === tag.type &&
                    tag.weight !== 1.0 &&
                    tag.type !== 'break' &&
                    tag.type !== 'network' &&
                    tag.type !== 'alternation';
                    
                const hasChainAfter = nextTag && 
                    nextTag.weight === tag.weight && 
                    nextTag.type === tag.type &&
                    tag.weight !== 1.0 &&
                    tag.type !== 'break' &&
                    tag.type !== 'network' &&
                    tag.type !== 'alternation';
                
                processedTags.push({
                    ...tag,
                    inChain: hasChainBefore || hasChainAfter,
                    chainStart: !hasChainBefore && hasChainAfter,
                    chainEnd: hasChainBefore && !hasChainAfter
                });
            }
            
            // Tagek hozzáadása
            processedTags.forEach(tagData => {
                this.addTag(tagData.text, tagData.weight, tagData.type, tagData);
            });
            
            // Chain státusz ellenőrzése beillesztés után
            this.recheckChainStatus();
        }
    }
    
    addTag(text, weight = 1.0, type = 'normal', metadata = {}) {
        // Tisztítás: zárójelek eltávolítása a szövegből (de NEM network/BREAK/ALTERNATION tageknél)
        const cleanText = (type === 'network' || type === 'break' || type === 'alternation') 
            ? text.trim() 
            : text.replace(/[(){}]/g, '').trim();
        
        // Üres szöveg ellenőrzés
        if (!cleanText) return;
        
        // Duplikátum ellenőrzés
        const existingTags = Array.from(this.promptContent.querySelectorAll(".prompt_tag"));
        if (existingTags.some(tag => tag.dataset.text === cleanText && tag.dataset.type === type)) return;
        
        const tag = Q('<div>', { class: 'prompt_tag' }).get(0);
        tag.dataset.text = cleanText;
        tag.dataset.weight = weight;
        tag.dataset.type = type;
        tag.draggable = true;
        
        // Network típus specifikus metadata
        if (type === 'network' && metadata.networkType) {
            tag.dataset.networkType = metadata.networkType;
        }
        
        // Típus szerinti class hozzáadása
        if (type === 'emphasis') {
            Q(tag).addClass('type-emphasis');
        } else if (type === 'break') {
            Q(tag).addClass('type-break');
        } else if (type === 'network') {
            Q(tag).addClass('type-network');
            // Network típus specifikus class (lora, lyco, ti, embedding, hypernet)
            if (metadata.networkType) {
                Q(tag).addClass(`network-${metadata.networkType}`);
            }
        } else if (type === 'alternation') {
            Q(tag).addClass('type-alternation');
        }
        
        // Chain jelölés
        if (metadata.inChain) {
            Q(tag).addClass('chain-linked');
        }
        
        // Tag tartalom - BREAK, ALTERNATION és TI/EMBEDDING kivételével mindenhol mutatjuk a weight kontrollt
        const networkType = metadata.networkType || '';
        const hasWeight = type !== 'break' && 
                         type !== 'alternation' && 
                         !(type === 'network' && (networkType === 'ti' || networkType === 'embedding'));
        
        if (type === 'break') {
            // BREAK tag - nincs weight
            Q(tag).html(`
                <span class="tag_text">${cleanText}</span>
                <div class="tag_controls">
                    <button class="remove_btn">×</button>
                </div>
            `);
        } else if (type === 'alternation') {
            // ALTERNATION tag - nincs weight, de megjelenítjük | helyett " OR "
            const displayText = cleanText.replace(/\|/g, ' OR ');
            Q(tag).html(`
                <span class="tag_text">${displayText}</span>
                <div class="tag_controls">
                    <button class="remove_btn">×</button>
                </div>
            `);
        } else if (type === 'network' && (networkType === 'ti' || networkType === 'embedding')) {
            // TI/Embedding tag - nincs weight
            Q(tag).html(`
                <span class="tag_text">${cleanText}</span>
                <div class="tag_controls">
                    <button class="remove_btn">×</button>
                </div>
            `);
        } else {
            // Minden más tag - van weight kontroll
            Q(tag).html(`
                <span class="tag_text">${cleanText}</span>
                <div class="tag_controls">
                    <button class="weight_btn decrease">-</button>
                    <span class="weight_display">${weight.toFixed(1)}</span>
                    <button class="weight_btn increase">+</button>
                    <button class="remove_btn">×</button>
                </div>
            `);
        }
        
        this.setupTagEvents(tag);
        
        // Beszúrás a main input elé
        const mainInput = this.promptContent.querySelector(".main_input");
        if (mainInput) {
            this.promptContent.insertBefore(tag, mainInput);
        } else {
            this.promptContent.appendChild(tag);
        }
        
        this.updateHiddenInput();
    }
    
    setupTagEvents(tag) {
        const textSpan = tag.querySelector(".tag_text");
        const decreaseBtn = tag.querySelector(".decrease");
        const increaseBtn = tag.querySelector(".increase");
        const removeBtn = tag.querySelector(".remove_btn");
        const weightDisplay = tag.querySelector(".weight_display");
        
        // Weight kontrollok (csak ha van súly megjelenítés)
        if (decreaseBtn && increaseBtn && weightDisplay) {
            Q(decreaseBtn).on("click", (e) => {
                e.stopPropagation();
                const weight = Math.max(0.1, Math.round((parseFloat(tag.dataset.weight) - 0.1) * 10) / 10);
                tag.dataset.weight = weight;
                Q(weightDisplay).text(weight.toFixed(1));
                this.updateHiddenInput();
                this.recheckChainStatus(); // Újraellenőrzés weight változás után
            });
            
            Q(increaseBtn).on("click", (e) => {
                e.stopPropagation();
                const weight = Math.min(2.0, Math.round((parseFloat(tag.dataset.weight) + 0.1) * 10) / 10);
                tag.dataset.weight = weight;
                Q(weightDisplay).text(weight.toFixed(1));
                this.updateHiddenInput();
                this.recheckChainStatus(); // Újraellenőrzés weight változás után
            });
        }
        
        // Törlés
        Q(removeBtn).on("click", (e) => {
            e.stopPropagation();
            tag.remove();
            this.recheckChainStatus(); // Újraellenőrzés törlés után
            this.updateHiddenInput();
        });
        
        // Szerkesztés - csak normál és emphasis tagekhez
        const tagType = tag.dataset.type;
        if (tagType === 'normal' || tagType === 'emphasis') {
            Q(textSpan).on("dblclick", (e) => {
                e.stopPropagation();
                this.editTag(tag, textSpan);
            });
        }
    }
    
    editTag(tag, textSpan) {
        const currentText = tag.dataset.text;
    const input = Q('<input>', { class: 'tag_edit_input' }).get(0);
        input.value = currentText;
        input.style.width = (currentText.length + 2) + "ch";
        
    Q(textSpan).css('display', 'none');
        tag.insertBefore(input, textSpan);
        input.focus();
        input.select();
        
        const finishEdit = () => {
            const newText = input.value.trim();
            if (newText && newText !== currentText) {
                tag.dataset.text = newText;
                Q(textSpan).text(newText);
                this.updateHiddenInput();
            }
            input.remove();
            Q(textSpan).css('display', '');
        };
        
    Q(input).on("blur", finishEdit);
    Q(input).on("keydown", (e) => {
            if (e.key === "Enter") {
                finishEdit();
            } else if (e.key === "Escape") {
                input.remove();
                Q(textSpan).css('display', '');
            }
        });
        
        // Auto-resize
    Q(input).on("input", () => {
            input.style.width = Math.max(input.value.length + 2, 6) + "ch";
        });
    }
    
    setupMainInput() {
        const mainInput = Q('<input>', { class: 'main_input' }).get(0);
        mainInput.placeholder = typeof window.APP_BOOTSTRAP !== 'undefined' && typeof window.APP_BOOTSTRAP.lang === 'function' 
            ? window.APP_BOOTSTRAP.lang('ui.promptfield.add_tags_placeholder')
            : "Add prompt tags...";
        this.promptContent.appendChild(mainInput);
        
        Q(mainInput).on("keydown", (e) => {
            if (e.key === "Enter" || e.key === ",") {
                e.preventDefault();
                const text = mainInput.value.trim();
                if (text) {
                    // Parse az input szöveget
                    const parsedTags = this.parsePrompt(text);
                    parsedTags.forEach(tagData => {
                        this.addTag(tagData.text, tagData.weight, tagData.type, tagData);
                    });
                    mainInput.value = "";
                    // Chain státusz ellenőrzése új tag hozzáadása után
                    this.recheckChainStatus();
                }
            }
        });
        
        // Click focus
        Q(this.promptContent).on("click", (e) => {
            if (e.target === this.promptContent) {
                mainInput.focus();
            }
        });
    }
    
    setupDragAndDrop() {
        let draggedElement = null;
        
    Q(this.promptContent).on("dragstart", (e) => {
            if (Q(e.target).hasClass("prompt_tag")) {
                draggedElement = e.target;
                Q(e.target).addClass("dragging");
                e.dataTransfer.effectAllowed = "move";
            }
        });
        
    Q(this.promptContent).on("dragend", (e) => {
            if (Q(e.target).hasClass("prompt_tag")) {
                Q(e.target).removeClass("dragging");
                draggedElement = null;
                // Cleanup all drag indicators
                this.promptContent.querySelectorAll(".drop_indicator").forEach(el => el.remove());
            }
        });
        
    Q(this.promptContent).on("dragover", (e) => {
            e.preventDefault();
            if (!draggedElement) return;
            
            const target = e.target.closest(".prompt_tag");
            if (target && target !== draggedElement) {
                this.showDropIndicator(target, e.clientX);
            }
        });
        
    Q(this.promptContent).on("drop", (e) => {
            e.preventDefault();
            if (!draggedElement) return;
            
            const target = e.target.closest(".prompt_tag");
            if (target && target !== draggedElement) {
                const rect = target.getBoundingClientRect();
                const midpoint = rect.left + rect.width / 2;
                
                if (e.clientX < midpoint) {
                    target.parentNode.insertBefore(draggedElement, target);
                } else {
                    target.parentNode.insertBefore(draggedElement, target.nextSibling);
                }
                
                this.recheckChainStatus(); // Újraellenőrzés drag után
                this.updateHiddenInput();
            }
        });
    }
    
    showDropIndicator(target, clientX) {
        // Töröljük a régi indikátorokat
        this.promptContent.querySelectorAll(".drop_indicator").forEach(el => el.remove());
        
        const rect = target.getBoundingClientRect();
        const midpoint = rect.left + rect.width / 2;
    const indicator = Q('<div>', { class: 'drop_indicator' }).get(0);
        
        if (clientX < midpoint) {
            target.parentNode.insertBefore(indicator, target);
        } else {
            target.parentNode.insertBefore(indicator, target.nextSibling);
        }
    }
    
    /**
     * Chain státusz újraellenőrzése (drag után vagy weight változásakor)
     */
    recheckChainStatus() {
        const tags = Array.from(this.promptContent.querySelectorAll(".prompt_tag"));
        
        tags.forEach((tag, i) => {
            const weight = parseFloat(tag.dataset.weight);
            const type = tag.dataset.type;
            
            // Csak súlyozott tageket ellenőrizzük (nem BREAK, network, alternation)
            if (type === 'break' || type === 'network' || type === 'alternation' || weight === 1.0) {
                Q(tag).removeClass('chain-linked');
                return;
            }
            
            // Előző és következő tag
            const prevTag = i > 0 ? tags[i - 1] : null;
            const nextTag = i < tags.length - 1 ? tags[i + 1] : null;
            
            const hasChainBefore = prevTag && 
                parseFloat(prevTag.dataset.weight) === weight && 
                prevTag.dataset.type === type &&
                prevTag.dataset.type !== 'break' &&
                prevTag.dataset.type !== 'network' &&
                prevTag.dataset.type !== 'alternation';
                
            const hasChainAfter = nextTag && 
                parseFloat(nextTag.dataset.weight) === weight && 
                nextTag.dataset.type === type &&
                nextTag.dataset.type !== 'break' &&
                nextTag.dataset.type !== 'network' &&
                nextTag.dataset.type !== 'alternation';
            
            // Chain class alkalmazása
            if (hasChainBefore || hasChainAfter) {
                Q(tag).addClass('chain-linked');
            } else {
                Q(tag).removeClass('chain-linked');
            }
        });
    }
    
    updateHiddenInput() {
        const tags = Array.from(this.promptContent.querySelectorAll(".prompt_tag"));
        const values = [];
        
        for (let i = 0; i < tags.length; i++) {
            const tag = tags[i];
            const text = tag.dataset.text;
            const weight = parseFloat(tag.dataset.weight);
            const type = tag.dataset.type;
            
            let formatted = '';
            
            if (type === 'break') {
                formatted = 'BREAK';
            } else if (type === 'network') {
                // Network formátum: <type:name:weight>
                const networkType = tag.dataset.networkType || 'lora';
                
                // TI és embedding típusoknak nincs weight-je
                if (networkType === 'ti' || networkType === 'embedding') {
                    formatted = `<${text}>`;
                } else {
                    // lora, lyco, hypernet - van weight
                    if (weight === 1.0) {
                        formatted = `<${text}>`;
                    } else {
                        formatted = `<${text}:${weight.toFixed(1)}>`;
                    }
                }
            } else if (type === 'alternation') {
                // Alternation formátum: {cat|dog}
                formatted = `{${text}}`;
            } else {
                // Normál és emphasis tagek
                if (weight === 1.0) {
                    formatted = text;
                } else {
                    // Chain optimization: ha egymás mellett vannak azonos súlyú tagek
                    const prevTag = i > 0 ? tags[i - 1] : null;
                    const nextTag = i < tags.length - 1 ? tags[i + 1] : null;
                    
                    const prevSameWeight = prevTag && 
                        parseFloat(prevTag.dataset.weight) === weight &&
                        prevTag.dataset.type !== 'break' &&
                        prevTag.dataset.type !== 'network' &&
                        prevTag.dataset.type !== 'alternation';
                    const nextSameWeight = nextTag && 
                        parseFloat(nextTag.dataset.weight) === weight &&
                        nextTag.dataset.type !== 'break' &&
                        nextTag.dataset.type !== 'network' &&
                        nextTag.dataset.type !== 'alternation';
                    
                    // Ha chain részben van, akkor egyszerűen csak a szöveget
                    if (prevSameWeight || nextSameWeight) {
                        // Chain elején zárójeleket nyitunk, végén zárjuk
                        if (!prevSameWeight && nextSameWeight) {
                            // Chain kezdete - nyitjuk a zárójelet
                            const depth = this.weightToDepth(weight);
                            formatted = '('.repeat(depth) + text;
                        } else if (prevSameWeight && !nextSameWeight) {
                            // Chain vége - zárjuk a zárójelet
                            const depth = this.weightToDepth(weight);
                            formatted = text + ')'.repeat(depth);
                        } else {
                            // Chain közepén
                            formatted = text;
                        }
                    } else {
                        // Egyedülálló tag - teljes súlyozás
                        formatted = `(${text}:${weight.toFixed(1)})`;
                    }
                }
            }
            
            values.push(formatted);
        }
        
        this.hiddenInput.value = values.join(', ');
        Q(this.hiddenInput).trigger("change");
        
        // Save to history after changes
        this.saveToHistory();
    }
    
    /**
     * Weight értékből kiszámolja hány zárójelbe kell tenni
     * 1.1 = 1 zárójel, 1.21 = 2 zárójel, stb.
     */
    weightToDepth(weight) {
        if (weight <= 1.0) return 0;
        return Math.round(Math.log(weight) / Math.log(1.1));
    }
    
    get() {
        return this.hiddenInput.value;
    }
    
    set(value) {
        // Normalize arrays to comma-delimited string
        if (Array.isArray(value)) {
            value = value.filter(Boolean).join(', ');
        }
        Q(this.promptContent).empty();
        this.hiddenInput.value = value || '';
        this.initializeTags(value);
        this.setupMainInput();
    }
    
    getElement() {
        return this.promptWrapper;
    }
}
