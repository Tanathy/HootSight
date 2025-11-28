/**
 * HootSight - Modal/Dialog Widget
 * Custom modal dialogs to replace native browser alert/confirm/prompt
 * Uses ActionButton and TextInput widgets internally
 * 
 * Usage:
 *   // Alert (simple message)
 *   await Modal.alert('Operation complete!');
 *   await Modal.alert('Error occurred', 'Error');
 *   
 *   // Confirm (yes/no)
 *   const confirmed = await Modal.confirm('Are you sure?');
 *   const confirmed = await Modal.confirm('Delete this item?', 'Confirm Delete');
 *   
 *   // Prompt (text input)
 *   const name = await Modal.prompt('Enter folder name:', 'New Folder');
 *   const name = await Modal.prompt('Enter name:', 'Rename', 'default value');
 *   
 *   // Custom modal
 *   Modal.show({
 *       title: 'Custom Modal',
 *       content: '<p>Custom HTML content</p>',
 *       buttons: [
 *           { label: 'Cancel', variant: 'ghost', action: () => false },
 *           { label: 'Save', variant: 'primary', action: () => true }
 *       ]
 *   });
 */

const Modal = {
    _overlay: null,
    _container: null,
    _resolveCallback: null,
    _isOpen: false,
    _promptInput: null,

    /**
     * Initialize the modal system
     */
    init: function() {
        if (this._overlay) return;

        this._overlay = Q('<div>', { class: 'modal-overlay' }).get(0);
        this._container = Q('<div>', { class: 'modal-container' }).get(0);
        this._overlay.appendChild(this._container);
        
        document.body.appendChild(this._overlay);

        Q(this._overlay).on('click', (e) => {
            if (e.target === this._overlay) {
                this._close(null);
            }
        });

        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this._isOpen) {
                this._close(null);
            }
        });
    },

    /**
     * Show alert dialog
     * @param {string} message - Message to display
     * @param {string} [title=''] - Optional title
     * @returns {Promise<void>}
     */
    alert: function(message, title = '') {
        return this.show({
            title: title,
            content: `<p class="modal-message">${this._escapeHtml(message)}</p>`,
            buttons: [
                { label: 'OK', variant: 'primary', action: () => true, autoFocus: true }
            ]
        });
    },

    /**
     * Show confirm dialog
     * @param {string} message - Message to display
     * @param {string} [title=''] - Optional title
     * @returns {Promise<boolean>}
     */
    confirm: function(message, title = '') {
        return this.show({
            title: title,
            content: `<p class="modal-message">${this._escapeHtml(message)}</p>`,
            buttons: [
                { label: lang('common.cancel') || 'Cancel', variant: 'ghost', action: () => false },
                { label: 'OK', variant: 'primary', action: () => true, autoFocus: true }
            ]
        });
    },

    /**
     * Show prompt dialog
     * @param {string} message - Message/label to display
     * @param {string} [title=''] - Optional title
     * @param {string} [defaultValue=''] - Default input value
     * @returns {Promise<string|null>} - Input value or null if cancelled
     */
    prompt: function(message, title = '', defaultValue = '') {
        const inputId = 'modal-prompt-' + Date.now();
        
        // Create TextInput widget
        this._promptInput = new TextInput(inputId, {
            label: message,
            default: defaultValue,
            placeholder: ''
        });

        return this.show({
            title: title,
            contentElement: this._promptInput.getElement(),
            buttons: [
                { label: lang('common.cancel') || 'Cancel', variant: 'ghost', action: () => null },
                { label: 'OK', variant: 'primary', action: () => this._promptInput.get(), autoFocus: false }
            ],
            onOpen: () => {
                this._promptInput.focus();
                // Enter key submits
                this._promptInput.input.addEventListener('keydown', (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        this._close(this._promptInput.get());
                    }
                });
            }
        });
    },

    /**
     * Show custom modal
     * @param {Object} options
     * @param {string} [options.title] - Modal title
     * @param {string} [options.content] - HTML content string
     * @param {HTMLElement} [options.contentElement] - DOM element as content
     * @param {Array} options.buttons - Button definitions [{label, variant, action, autoFocus}]
     * @param {Function} [options.onOpen] - Callback after modal opens
     * @returns {Promise<any>}
     */
    show: function(options) {
        this.init();
        
        return new Promise((resolve) => {
            this._resolveCallback = resolve;
            this._container.innerHTML = '';
            
            // Header (if title)
            if (options.title) {
                const header = Q('<div>', { class: 'modal-header' }).get(0);
                const title = Q('<h3>', { class: 'modal-title', text: options.title }).get(0);
                header.appendChild(title);
                
                const closeBtn = new ActionButton('modal-close', {
                    label: '\u00D7',
                    className: 'modal-close',
                    onClick: () => this._close(null)
                });
                header.appendChild(closeBtn.getElement());
                
                this._container.appendChild(header);
            }
            
            // Body
            const body = Q('<div>', { class: 'modal-body' }).get(0);
            if (options.contentElement) {
                body.appendChild(options.contentElement);
            } else if (options.content) {
                body.innerHTML = options.content;
            }
            this._container.appendChild(body);
            
            // Footer with buttons (using ActionButton widgets)
            if (options.buttons && options.buttons.length > 0) {
                const footer = Q('<div>', { class: 'modal-footer' }).get(0);
                
                let autoFocusBtn = null;
                options.buttons.forEach((btnDef, idx) => {
                    const variantClass = btnDef.variant ? `btn-${btnDef.variant}` : 'btn-secondary';
                    const btn = new ActionButton(`modal-btn-${idx}`, {
                        label: btnDef.label,
                        className: `btn ${variantClass}`,
                        onClick: () => {
                            const result = typeof btnDef.action === 'function' ? btnDef.action() : btnDef.action;
                            this._close(result);
                        }
                    });
                    
                    if (btnDef.autoFocus) {
                        autoFocusBtn = btn.getElement();
                    }
                    
                    footer.appendChild(btn.getElement());
                });
                
                this._container.appendChild(footer);
                
                if (autoFocusBtn) {
                    setTimeout(() => autoFocusBtn.focus(), 10);
                }
            }
            
            // Show modal
            Q(this._overlay).addClass('visible');
            this._isOpen = true;
            
            if (options.onOpen) {
                setTimeout(options.onOpen, 10);
            }
        });
    },

    /**
     * Close modal and resolve promise
     */
    _close: function(result) {
        if (!this._isOpen) return;
        
        Q(this._overlay).removeClass('visible');
        this._isOpen = false;
        this._promptInput = null;
        
        if (this._resolveCallback) {
            this._resolveCallback(result);
            this._resolveCallback = null;
        }
    },

    /**
     * Hide modal (alias for external use)
     */
    hide: function() {
        this._close(null);
    },

    /**
     * Escape HTML for safe insertion
     */
    _escapeHtml: function(str) {
        if (!str) return '';
        return String(str)
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#039;');
    }
};

// Auto-initialize on load
document.addEventListener('DOMContentLoaded', () => Modal.init());
