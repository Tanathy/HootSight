/**
 * HootSight - Navigation Module
 * Dynamically generates sidebar navigation from configuration
 */

const Navigation = {
    /**
     * Navigation items configuration
     * order: display order (lower = higher)
     * page: page identifier for routing
     * langKey: localization key for label
     */
    items: [
        { order: 1, page: 'projects', langKey: 'nav.projects' },
        { order: 2, page: 'training', langKey: 'nav.training_setup' },
        { order: 3, page: 'dataset', langKey: 'nav.dataset' },
        { order: 4, page: 'inference', langKey: 'nav.inference' },
        { order: 5, page: 'settings', langKey: 'nav.settings' }
    ],

    /**
     * Build and render navigation items
     * @param {string} containerId - ID of the nav container
     */
    build: function(containerId = 'sidebar-nav') {
        const container = document.getElementById(containerId);
        if (!container) {
            console.error('Navigation container not found:', containerId);
            return;
        }

        // Clear existing items
        container.innerHTML = '';

        // Sort by order
        const sorted = [...this.items].sort((a, b) => a.order - b.order);

        // Create nav items
        sorted.forEach((item, index) => {
            const navItem = document.createElement('div');
            navItem.className = 'nav-item';
            navItem.dataset.page = item.page;

            // First item is active by default
            if (index === 0) {
                navItem.classList.add('active');
            }

            const label = document.createElement('span');
            label.className = 'nav-item-label';
            label.textContent = lang(item.langKey);

            navItem.appendChild(label);
            container.appendChild(navItem);
        });
    },

    /**
     * Add a navigation item
     * @param {Object} item - { order, page, langKey }
     */
    addItem: function(item) {
        if (!item.page || !item.langKey) {
            console.error('Navigation item requires page and langKey');
            return;
        }
        item.order = item.order ?? this.items.length + 1;
        this.items.push(item);
    },

    /**
     * Remove a navigation item by page
     * @param {string} page - Page identifier
     */
    removeItem: function(page) {
        this.items = this.items.filter(item => item.page !== page);
    },

    /**
     * Get the default (first) page
     * @returns {string} - Page identifier
     */
    getDefaultPage: function() {
        const sorted = [...this.items].sort((a, b) => a.order - b.order);
        return sorted.length > 0 ? sorted[0].page : 'projects';
    },

    /**
     * Navigate to a page (triggers click on nav item)
     * @param {string} pageName - Page identifier
     */
    navigateTo: function(pageName) {
        const navItem = document.querySelector(`.nav-item[data-page="${pageName}"]`);
        if (navItem) {
            navItem.click();
        }
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Navigation;
}
