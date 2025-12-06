/**
 * Card widget - container for other widgets/content
 */
class Card {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            title: options.title || '',
            titleLangKey: options.titleLangKey || null,
            subtitle: options.subtitle || '',
            subtitleLangKey: options.subtitleLangKey || null,
            description: options.description || '',
            descriptionLangKey: options.descriptionLangKey || null,
            footer: options.footer || '',
            footerLangKey: options.footerLangKey || null,
            actions: options.actions || [],
            content: options.content || null
        };
        
        this._element = Q('<div>', { class: 'card', id: `card-${this.id}` }).get();
        this._build();
    }
    
    _build() {
        if (this.options.title || this.options.subtitle || this.options.actions.length) {
            this._header = Q('<div>', { class: 'card-header' }).get();
            
            const titleGroup = Q('<div>', { class: 'card-title-group' }).get();
            
            this._titleEl = Q('<div>', { class: 'card-title', text: this.options.title }).get();
            if (this.options.titleLangKey) {
                this._titleEl.setAttribute('data-lang-key', this.options.titleLangKey);
            }
            if (this.options.title) {
                Q(titleGroup).append(this._titleEl);
            }
            
            if (this.options.subtitle) {
                this._subtitleEl = Q('<div>', { class: 'card-subtitle', text: this.options.subtitle }).get();
                if (this.options.subtitleLangKey) {
                    this._subtitleEl.setAttribute('data-lang-key', this.options.subtitleLangKey);
                }
                Q(titleGroup).append(this._subtitleEl);
            }
            
            if (titleGroup.children.length) {
                Q(this._header).append(titleGroup);
            }
            
            if (this.options.actions.length) {
                this._actionsEl = Q('<div>', { class: 'card-actions' }).get();
                this.options.actions.forEach(action => this.addAction(action));
                Q(this._header).append(this._actionsEl);
            }
            
            Q(this._element).append(this._header);
        }
        
        this._body = Q('<div>', { class: 'card-body' }).get();
        Q(this._element).append(this._body);
        
        if (this.options.content) {
            this.addContent(this.options.content);
        }
        
        this._footer = Q('<div>', { class: 'card-footer' }).get();
        if (this.options.footerLangKey) {
            this._footer.setAttribute('data-lang-key', this.options.footerLangKey);
        }
        Q(this._footer).hide();
        if (this.options.footer) {
            this.setFooter(this.options.footer);
        }
        Q(this._element).append(this._footer);
    }
    
    addContent(content) {
        if (!content) return;
        if (content instanceof Node) {
            Q(this._body).append(content);
        } else if (typeof content === 'string') {
            Q(this._body).append(Q('<div>', { text: content }).get());
        } else if (content.getElement) {
            Q(this._body).append(content.getElement());
        }
    }
    
    addWidget(widgetInstance) {
        if (widgetInstance && widgetInstance.getElement) {
            Q(this._body).append(widgetInstance.getElement());
        }
        return this;
    }
    
    setTitle(title = '') {
        if (!this._titleEl) {
            this._titleEl = Q('<div>', { class: 'card-title' }).get();
            this._header = this._header || this._createHeader();
            this._header.insertBefore(this._titleEl, this._header.firstChild);
        }
        Q(this._titleEl).text(title);
    }
    
    setSubtitle(subtitle = '') {
        if (!this._subtitleEl) {
            this._subtitleEl = Q('<div>', { class: 'card-subtitle' }).get();
            this._header = this._header || this._createHeader();
            this._header.insertBefore(this._subtitleEl, this._header.firstChild.nextSibling);
        }
        Q(this._subtitleEl).text(subtitle);
    }
    
    _createHeader() {
        const header = Q('<div>', { class: 'card-header' }).get();
        this._element.insertBefore(header, this._body);
        return header;
    }
    
    clearBody() {
        Q(this._body).empty();
    }
    
    getBodyElement() {
        return this._body;
    }
    
    setFooter(content) {
        if (content instanceof Node) {
            Q(this._footer).empty();
            Q(this._footer).append(content);
        } else {
            Q(this._footer).text(content);
        }
        Q(this._footer).css('display', content ? '' : 'none');
    }
    
    addAction(action) {
        if (!this._actionsEl) {
            this._actionsEl = Q('<div>', { class: 'card-actions' }).get();
        }
        if (!this._header) {
            this._header = this._createHeader();
        }
        if (!this._actionsEl.isConnected) {
            Q(this._header).append(this._actionsEl);
        }
        const button = Q('<button>', {
            class: action.className || 'btn btn-secondary',
            text: action.label || 'Action'
        }).get();
        if (typeof action.onClick === 'function') {
            Q(button).on('click', action.onClick);
        }
        Q(this._actionsEl).append(button);
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
