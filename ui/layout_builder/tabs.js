/**
 * HootSight - Tabs Container
 * Horizontal tab component built with JavaScript
 * 
 * Usage:
 *   const tabs = new Tabs('my-tabs');
 *   tabs.addTab('tab1', 'First Tab', contentElement1);
 *   tabs.addTab('tab2', 'Second Tab', contentElement2);
 *   container.appendChild(tabs.getElement());
 *   tabs.activate('tab1');
 */

class Tabs {
    constructor(id) {
        this.id = id;
        this.tabs = [];
        this.activeTab = null;
        this._changeCallbacks = [];
        
        // Build DOM structure
        this.element = Q('<div>', { class: 'tabs', id: id }).get(0);
        this.header = Q('<div>', { class: 'tabs-header' }).get(0);
        this.content = Q('<div>', { class: 'tabs-content' }).get(0);
        
        Q(this.element).append(this.header);
        Q(this.element).append(this.content);
    }
    
    /**
     * Add a tab
     * @param {string} tabId - Unique tab identifier
     * @param {string} label - Tab button label
     * @param {HTMLElement|null} content - Tab panel content (optional, can add later)
     * @param {Object} options - Optional settings (e.g., langKey for localization)
     * @returns {Tabs} - Returns self for chaining
     */
    addTab(tabId, label, content = null, options = {}) {
        // Create tab button
        const button = Q('<button>', { 
            class: 'tab-button',
            text: label
        }).get(0);
        button.dataset.tab = tabId;
        
        // Add lang key for live translation
        if (options.langKey) {
            button.setAttribute('data-lang-key', options.langKey);
        }
        
        // Create tab panel
        const panel = Q('<div>', { class: 'tab-panel' }).get(0);
        panel.dataset.tab = tabId;
        
        if (content) {
            if (content instanceof HTMLElement) {
                Q(panel).append(content);
            } else if (typeof content === 'string') {
                Q(panel).html(content);
            }
        }
        
        // Event listener
        Q(button).on('click', () => this.activate(tabId));
        
        // Store tab info
        this.tabs.push({ id: tabId, label, button, panel, langKey: options.langKey });
        
        // Add to DOM
        Q(this.header).append(button);
        Q(this.content).append(panel);
        
        // Activate first tab by default
        if (this.tabs.length === 1) {
            this.activate(tabId);
        }
        
        return this;
    }
    
    /**
     * Remove a tab
     * @param {string} tabId - Tab identifier to remove
     * @returns {Tabs}
     */
    removeTab(tabId) {
        const index = this.tabs.findIndex(t => t.id === tabId);
        if (index === -1) return this;
        
        const tab = this.tabs[index];
        tab.button.remove();
        tab.panel.remove();
        this.tabs.splice(index, 1);
        
        // If removed tab was active, activate another
        if (this.activeTab === tabId && this.tabs.length > 0) {
            this.activate(this.tabs[0].id);
        }
        
        return this;
    }
    
    /**
     * Activate a tab
     * @param {string} tabId - Tab identifier to activate
     * @returns {Tabs}
     */
    activate(tabId) {
        const tab = this.tabs.find(t => t.id === tabId);
        if (!tab) return this;
        
        // Deactivate all
        this.tabs.forEach(t => {
            Q(t.button).removeClass('active');
            Q(t.panel).removeClass('active');
        });
        
        // Activate selected
        Q(tab.button).addClass('active');
        Q(tab.panel).addClass('active');
        
        const previousTab = this.activeTab;
        this.activeTab = tabId;
        
        // Trigger callbacks
        if (previousTab !== tabId) {
            this._changeCallbacks.forEach(cb => cb(tabId, previousTab));
        }
        
        return this;
    }
    
    /**
     * Get the active tab ID
     * @returns {string|null}
     */
    getActive() {
        return this.activeTab;
    }
    
    /**
     * Get a tab's panel element
     * @param {string} tabId - Tab identifier
     * @returns {HTMLElement|null}
     */
    getPanel(tabId) {
        const tab = this.tabs.find(t => t.id === tabId);
        return tab ? tab.panel : null;
    }
    
    /**
     * Set tab content
     * @param {string} tabId - Tab identifier
     * @param {HTMLElement|string} content - New content
     * @returns {Tabs}
     */
    setContent(tabId, content) {
        const panel = this.getPanel(tabId);
        if (!panel) return this;
        
        Q(panel).empty();
        if (content instanceof HTMLElement) {
            Q(panel).append(content);
        } else if (typeof content === 'string') {
            Q(panel).html(content);
        }
        
        return this;
    }

    /**
     * Update tab label
     * @param {string} tabId - Tab identifier
     * @param {string} label - New label
     * @returns {Tabs}
     */
    setLabel(tabId, label) {
        const tab = this.tabs.find(t => t.id === tabId);
        if (tab) {
            tab.label = label;
            Q(tab.button).text(label);
        }
        return this;
    }
    
    /**
     * Register change callback
     * @param {Function} callback - Called with (newTabId, previousTabId)
     * @returns {Tabs}
     */
    onChange(callback) {
        this._changeCallbacks.push(callback);
        return this;
    }
    
    /**
     * Get the DOM element
     * @returns {HTMLElement}
     */
    getElement() {
        return this.element;
    }
    
    /**
     * Get all tab IDs
     * @returns {string[]}
     */
    getTabIds() {
        return this.tabs.map(t => t.id);
    }
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Tabs;
}
