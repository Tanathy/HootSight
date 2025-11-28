/**
 * Card widget - container for other widgets/content
 */
class Card {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            title: options.title || '',
            subtitle: options.subtitle || '',
            description: options.description || '',
            footer: options.footer || '',
            actions: options.actions || [],
            content: options.content || null
        };
        
        this._element = document.createElement('div');
        this._element.className = 'card';
        this._element.id = `card-${this.id}`;
        
        this._build();
    }
    
    _build() {
        if (this.options.title || this.options.subtitle || this.options.actions.length) {
            this._header = document.createElement('div');
            this._header.className = 'card-header';
            
            const titleGroup = document.createElement('div');
            titleGroup.className = 'card-title-group';
            
            this._titleEl = document.createElement('div');
            this._titleEl.className = 'card-title';
            this._titleEl.textContent = this.options.title;
            if (this.options.title) {
                titleGroup.appendChild(this._titleEl);
            }
            
            if (this.options.subtitle) {
                this._subtitleEl = document.createElement('div');
                this._subtitleEl.className = 'card-subtitle';
                this._subtitleEl.textContent = this.options.subtitle;
                titleGroup.appendChild(this._subtitleEl);
            }
            
            if (titleGroup.children.length) {
                this._header.appendChild(titleGroup);
            }
            
            if (this.options.actions.length) {
                this._actionsEl = document.createElement('div');
                this._actionsEl.className = 'card-actions';
                this.options.actions.forEach(action => this.addAction(action));
                this._header.appendChild(this._actionsEl);
            }
            
            this._element.appendChild(this._header);
        }
        
        this._body = document.createElement('div');
        this._body.className = 'card-body';
        this._element.appendChild(this._body);
        
        if (this.options.content) {
            this.addContent(this.options.content);
        }
        
        this._footer = document.createElement('div');
        this._footer.className = 'card-footer';
        this._footer.style.display = 'none';
        if (this.options.footer) {
            this.setFooter(this.options.footer);
        }
        this._element.appendChild(this._footer);
    }
    
    addContent(content) {
        if (!content) return;
        if (content instanceof Node) {
            this._body.appendChild(content);
        } else if (typeof content === 'string') {
            const paragraph = document.createElement('div');
            paragraph.textContent = content;
            this._body.appendChild(paragraph);
        } else if (content.getElement) {
            this._body.appendChild(content.getElement());
        }
    }
    
    addWidget(widgetInstance) {
        if (widgetInstance && widgetInstance.getElement) {
            this._body.appendChild(widgetInstance.getElement());
        }
        return this;
    }
    
    setTitle(title = '') {
        if (!this._titleEl) {
            this._titleEl = document.createElement('div');
            this._titleEl.className = 'card-title';
            this._header = this._header || this._createHeader();
            this._header.insertBefore(this._titleEl, this._header.firstChild);
        }
        this._titleEl.textContent = title;
    }
    
    setSubtitle(subtitle = '') {
        if (!this._subtitleEl) {
            this._subtitleEl = document.createElement('div');
            this._subtitleEl.className = 'card-subtitle';
            this._header = this._header || this._createHeader();
            this._header.insertBefore(this._subtitleEl, this._header.firstChild.nextSibling);
        }
        this._subtitleEl.textContent = subtitle;
    }
    
    _createHeader() {
        const header = document.createElement('div');
        header.className = 'card-header';
        this._element.insertBefore(header, this._body);
        return header;
    }
    
    clearBody() {
        this._body.innerHTML = '';
    }
    
    getBodyElement() {
        return this._body;
    }
    
    setFooter(content) {
        if (content instanceof Node) {
            this._footer.innerHTML = '';
            this._footer.appendChild(content);
        } else {
            this._footer.textContent = content;
        }
        this._footer.style.display = content ? '' : 'none';
    }
    
    addAction(action) {
        this._actionsEl = this._actionsEl || document.createElement('div');
        this._actionsEl.className = 'card-actions';
        if (!this._header) {
            this._header = this._createHeader();
        }
        if (!this._actionsEl.isConnected) {
            this._header.appendChild(this._actionsEl);
        }
        const button = document.createElement('button');
        button.className = action.className || 'btn btn-secondary';
        button.textContent = action.label || 'Action';
        if (typeof action.onClick === 'function') {
            button.addEventListener('click', action.onClick);
        }
        this._actionsEl.appendChild(button);
    }
    
    getElement() {
        return this._element;
    }
    
    static fromSchema(id, schema = {}) {
        const ui = schema.ui || {};
        return new Card(id, {
            title: schema.title || ui.title || id,
            subtitle: ui.subtitle || '',
            footer: ui.footer || ''
        });
    }
}
