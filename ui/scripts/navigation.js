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
        { order: 4, page: 'performance', langKey: 'nav.performance' },
        { order: 5, page: 'heatmap', langKey: 'nav.heatmap' },
        { order: 6, page: 'updates', langKey: 'nav.updates' }
    ],

    /**
     * Language dropdown element reference
     */
    _langDropdown: null,

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
            label.setAttribute('data-lang-key', item.langKey);

            navItem.appendChild(label);
            container.appendChild(navItem);
        });

        // Build language selector in sidebar footer
        this._buildLanguageSelector();
    },

    /**
     * Build the language selector dropdown in sidebar footer
     * Uses the Dropdown widget for consistent styling
     */
    _buildLanguageSelector: function() {
        const sidebar = document.querySelector('.sidebar');
        if (!sidebar) return;

        // Remove existing footer if any
        const existingFooter = sidebar.querySelector('.sidebar-footer');
        if (existingFooter) {
            existingFooter.remove();
        }

        // Get available languages
        const activeCode = Lang.getActiveLanguage();
        const languages = Lang.getAvailableLanguages();

        if (!languages || languages.length === 0) {
            return; // No languages available
        }

        // Build options and labels for Dropdown widget
        const options = languages.map(l => l.code);
        const optionLabels = {};
        languages.forEach(l => {
            optionLabels[l.code] = l.name;
        });

        // Create footer container
        const footer = document.createElement('div');
        footer.className = 'sidebar-footer';

        // Create Dropdown widget
        this._langDropdown = new Dropdown('language-selector', {
            label: lang('ui.language_select_title'),
            labelLangKey: 'ui.language_select_title',
            description: '',
            options: options,
            optionLabels: optionLabels,
            default: activeCode
        });

        // Handle language change
        this._langDropdown.onChange(async (newCode) => {
            const success = await Lang.switchLanguage(newCode);
            if (!success) {
                // Revert to previous value on failure
                this._langDropdown.set(activeCode);
            }
        });

        footer.appendChild(this._langDropdown.getElement());
        sidebar.appendChild(footer);
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
