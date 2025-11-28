/**
 * HootSight - Pages
 * Page definitions and content structure
 */

const Pages = {
    
    /**
     * Page registry
     */
    _pages: {},
    
    /**
     * Register a page
     * @param {string} name - Page name (matches nav data-page)
     * @param {Object} config - Page configuration
     */
    register(name, config) {
        this._pages[name] = config;
    },
    
    /**
     * Get page config
     * @param {string} name
     * @returns {Object|null}
     */
    get(name) {
        return this._pages[name] || null;
    },
    
    /**
     * Render page content
     * @param {string} name - Page name
     * @param {HTMLElement} container - Container to render into
     */
    render(name, container) {
        const page = this.get(name);
        if (!page) {
            container.innerHTML = '<div class="heading"><h2 class="heading-title">Page Not Found</h2></div>';
            return;
        }
        
        // Clear container
        container.innerHTML = '';
        
        // Build page content
        if (page.build && typeof page.build === 'function') {
            page.build(container);
        }
    }
};

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Pages;
}
