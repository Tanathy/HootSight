/**
 * HootSight - Context Menu Widget
 * Global right-click context menu with dynamic actions
 * 
 * Usage:
 *   // Register context for specific elements
 *   ContextMenu.register('.image-card', (element, event) => [
 *       { label: 'View', icon: 'view.svg', action: () => viewImage(element) },
 *       { type: 'separator' },
 *       { label: 'Delete', icon: 'trash.svg', action: () => deleteImage(element), danger: true },
 *       { label: 'Submenu', icon: 'folder.svg', children: [
 *           { label: 'Option 1', action: () => {} },
 *           { label: 'Option 2', action: () => {} }
 *       ]}
 *   ]);
 *   
 *   // Show programmatically
 *   ContextMenu.show(x, y, menuItems);
 *   
 *   // Hide
 *   ContextMenu.hide();
 */

const ContextMenu = {
    _container: null,
    _submenuContainer: null,
    _registrations: [],
    _activeElement: null,
    _isVisible: false,

    /**
     * Initialize the context menu system
     */
    init: function() {
        if (this._container) return;

        // Create main menu container
        this._container = Q('<div>', { class: 'context-menu' }).get(0);
        document.body.appendChild(this._container);

        // Create submenu container
        this._submenuContainer = Q('<div>', { class: 'context-menu context-submenu' }).get(0);
        document.body.appendChild(this._submenuContainer);

        // Global click handler to close menu
        document.addEventListener('click', (e) => {
            if (!this._container.contains(e.target) && !this._submenuContainer.contains(e.target)) {
                this.hide();
            }
        });

        // Global right-click handler
        document.addEventListener('contextmenu', (e) => this._handleContextMenu(e));

        // Close on escape
        document.addEventListener('keydown', (e) => {
            if (e.key === 'Escape' && this._isVisible) {
                this.hide();
            }
        });

        // Close on scroll
        document.addEventListener('scroll', () => this.hide(), true);
    },

    /**
     * Register context menu for a selector
     * @param {string} selector - CSS selector
     * @param {Function} menuBuilder - Function(element, event) returning menu items array
     */
    register: function(selector, menuBuilder) {
        this._registrations.push({ selector, menuBuilder });
    },

    /**
     * Unregister context menu for a selector
     * @param {string} selector - CSS selector to remove
     */
    unregister: function(selector) {
        this._registrations = this._registrations.filter(r => r.selector !== selector);
    },

    /**
     * Handle context menu event
     * @param {MouseEvent} e
     */
    _handleContextMenu: function(e) {
        // Find matching registration
        for (const reg of this._registrations) {
            const element = e.target.closest(reg.selector);
            if (element) {
                e.preventDefault();
                const items = reg.menuBuilder(element, e);
                if (items && items.length > 0) {
                    this._activeElement = element;
                    this.show(e.clientX, e.clientY, items);
                }
                return;
            }
        }
    },

    /**
     * Show context menu at position
     * @param {number} x - X coordinate
     * @param {number} y - Y coordinate
     * @param {Array} items - Menu items
     */
    show: function(x, y, items) {
        if (!this._container) this.init();

        this._container.innerHTML = '';
        this._submenuContainer.innerHTML = '';
        this._submenuContainer.style.display = 'none';

        items.forEach((item, index) => {
            const menuItem = this._createMenuItem(item, index);
            this._container.appendChild(menuItem);
        });

        // Position menu
        this._container.style.display = 'block';
        this._isVisible = true;

        // Adjust position if menu goes off screen
        const rect = this._container.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let finalX = x;
        let finalY = y;

        if (x + rect.width > viewportWidth) {
            finalX = viewportWidth - rect.width - 8;
        }
        if (y + rect.height > viewportHeight) {
            finalY = viewportHeight - rect.height - 8;
        }

        this._container.style.left = finalX + 'px';
        this._container.style.top = finalY + 'px';
    },

    /**
     * Hide context menu
     */
    hide: function() {
        if (this._container) {
            this._container.style.display = 'none';
        }
        if (this._submenuContainer) {
            this._submenuContainer.style.display = 'none';
        }
        this._isVisible = false;
        this._activeElement = null;
    },

    /**
     * Create a menu item element
     * @param {Object} item - Menu item config
     * @param {number} index - Item index
     * @returns {HTMLElement}
     */
    _createMenuItem: function(item, index) {
        // Separator
        if (item.type === 'separator') {
            return Q('<div>', { class: 'context-menu-separator' }).get(0);
        }

        const menuItem = Q('<div>', { class: 'context-menu-item' }).get(0);

        // Disabled state
        if (item.disabled) {
            menuItem.classList.add('disabled');
        }

        // Danger state (red color)
        if (item.danger) {
            menuItem.classList.add('danger');
        }

        // Icon
        if (item.icon) {
            const icon = Q('<img>', { 
                src: `/static/icons/${item.icon}`, 
                class: 'context-menu-icon',
                alt: ''
            }).get(0);
            menuItem.appendChild(icon);
        } else {
            // Spacer for alignment when no icon
            const spacer = Q('<span>', { class: 'context-menu-icon-spacer' }).get(0);
            menuItem.appendChild(spacer);
        }

        // Label
        const label = Q('<span>', { class: 'context-menu-label', text: item.label }).get(0);
        menuItem.appendChild(label);

        // Shortcut hint
        if (item.shortcut) {
            const shortcut = Q('<span>', { class: 'context-menu-shortcut', text: item.shortcut }).get(0);
            menuItem.appendChild(shortcut);
        }

        // Submenu arrow
        if (item.children && item.children.length > 0) {
            const arrow = Q('<span>', { class: 'context-menu-arrow', text: '\u25B6' }).get(0);
            menuItem.appendChild(arrow);
            menuItem.classList.add('has-submenu');

            // Show submenu on hover
            Q(menuItem).on('mouseenter', () => {
                this._showSubmenu(menuItem, item.children);
            });
        }

        // Click handler
        if (item.action && !item.disabled) {
            Q(menuItem).on('click', (e) => {
                e.stopPropagation();
                this.hide();
                item.action(this._activeElement);
            });
        }

        return menuItem;
    },

    /**
     * Show submenu
     * @param {HTMLElement} parentItem - Parent menu item element
     * @param {Array} items - Submenu items
     */
    _showSubmenu: function(parentItem, items) {
        this._submenuContainer.innerHTML = '';

        items.forEach((item, index) => {
            const menuItem = this._createMenuItem(item, index);
            this._submenuContainer.appendChild(menuItem);
        });

        // Position submenu
        const parentRect = parentItem.getBoundingClientRect();
        const menuRect = this._container.getBoundingClientRect();

        this._submenuContainer.style.display = 'block';

        const submenuRect = this._submenuContainer.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let x = parentRect.right;
        let y = parentRect.top;

        // Flip to left if not enough space on right
        if (x + submenuRect.width > viewportWidth) {
            x = menuRect.left - submenuRect.width;
        }

        // Adjust vertical position
        if (y + submenuRect.height > viewportHeight) {
            y = viewportHeight - submenuRect.height - 8;
        }

        this._submenuContainer.style.left = x + 'px';
        this._submenuContainer.style.top = y + 'px';
    },

    /**
     * Get the currently active element (element that was right-clicked)
     * @returns {HTMLElement|null}
     */
    getActiveElement: function() {
        return this._activeElement;
    }
};

// Auto-initialize on DOM ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => ContextMenu.init());
} else {
    ContextMenu.init();
}

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = ContextMenu;
}
