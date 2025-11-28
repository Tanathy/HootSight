class Box {
    constructor(container, config) {
        this.container = container;
        this.config = config;
        this.element = null;
        this.titleElement = null;
        this.contentElement = null;
        this.collapseButton = null;
        this.popupButton = null;
        this.isCollapsed = config.collapsed || false;
        this.floatingWindow = null;
        this.isPoppedOut = false;

        this.init();
    }

    init() {
        this.createElement();
        this.attachToContainer();
    }

    createElement() {
        this.element = Q('<div>', { class: 'box-component' }).get(0);
        this.element.id = this.config.id;

        if (this.config.collapsible) {
            Q(this.element).addClass('collapsible');
        }

        if (this.isCollapsed) {
            Q(this.element).addClass('collapsed');
        }

        // Apply colors if specified
        if (this.config.colors) {
            if (this.config.colors.startsWith('--')) {
                Q(this.element).css('backgroundColor', `var(${this.config.colors})`);
            } else {
                Q(this.element).css('backgroundColor', this.config.colors);
            }
        }

        if (this.config.title) {
            const titleWrapper = Q('<div>', { class: 'box-title-wrapper' }).get(0);
            
            // Create left container for module icon and title
            const titleLeft = Q('<div>', { class: 'box-title-left' }).get(0);
            
            // Debug: log isModule flag
            if (this.config.id === 'hires-box' || this.config.id === 'styles-manager-box') {
                console.log(`🔍 Box ${this.config.id} isModule:`, this.config.isModule);
            }
            
            // Add module icon if this box is from a module
            if (this.config.isModule) {
                console.log(`✅ Adding module icon to ${this.config.id}`);
                const moduleIcon = Q('<span>', { class: 'box-module-icon' }).get(0);
                Q(moduleIcon).html(window.UI_ICONS.module);
                Q(titleLeft).append(moduleIcon);
            }
            
            this.titleElement = Q('<h2>', { class: 'section_title', text: this.config.title }).get(0);
            Q(titleLeft).append(this.titleElement);
            Q(titleWrapper).append(titleLeft);

            // Add buttons container for collapse and popup buttons
            const buttonsContainer = Q('<div>', { class: 'box-title-buttons' }).get(0);

            // Add popup button if enabled
            if (this.config.popup) {
                this.popupButton = Q('<button>', { class: 'box-popup-button' }).get(0);
                Q(this.popupButton).html(window.UI_ICONS.boxPopup);
                Q(this.popupButton).on('click', (e) => {
                    e.stopPropagation();
                    this.togglePopup();
                });
                Q(buttonsContainer).append(this.popupButton);
            }

            // Add collapse button if collapsible
            if (this.config.collapsible) {
                this.collapseButton = Q('<button>', { class: 'box-collapse-button' }).get(0);
                Q(this.collapseButton).html(this.isCollapsed ? '▶' : '▼');
                Q(this.collapseButton).on('click', (e) => {
                    e.stopPropagation();
                    this.toggle();
                });
                Q(buttonsContainer).append(this.collapseButton);
                Q(titleWrapper).addClass('collapsible');
                Q(titleWrapper).on('click', (e) => {
                    if (e.target !== this.collapseButton && e.target !== this.popupButton && !Q(e.target).closest('.box-popup-button').length) {
                        this.toggle();
                    }
                });
            }

            if (this.config.collapsible || this.config.popup) {
                Q(titleWrapper).append(buttonsContainer);
            }

            Q(this.element).append(titleWrapper);
        }

        if (this.config.description) {
            const descElement = Q('<h4>', { class: 'section_description', text: this.config.description }).get(0);
            Q(this.element).append(descElement);
        }

    this.contentElement = Q('<div>', { class: 'box-content' }).get(0);
    // Ensure unique ID for content element to avoid duplicating the outer box ID
    this.contentElement.id = `${this.config.id}-content`;

        // Propagate dynamicwindow intent to descendants via data attribute
        if (this.config.dynamicwindow) {
            this.element.setAttribute('data-dynamicwindow', 'true');
            this.contentElement.setAttribute('data-dynamicwindow', 'true');
        }

        if (this.isCollapsed) {
            Q(this.contentElement).hide();
        }

        Q(this.element).append(this.contentElement);
    }

    attachToContainer() {
        if (this.container && this.element) {
            Q(this.container).append(this.element);
        }
    }

    toggle() {
        if (!this.config.collapsible) return;

        this.isCollapsed = !this.isCollapsed;

        if (this.isCollapsed) {
            Q(this.element).addClass('collapsed');
            Q(this.contentElement).hide();
            if (this.collapseButton) {
                Q(this.collapseButton).html('▶');
            }
        } else {
            Q(this.element).removeClass('collapsed');
            Q(this.contentElement).show();
            if (this.collapseButton) {
                Q(this.collapseButton).html('▼');
            }
        }
    }

    collapse() {
        if (!this.config.collapsible) return;

        this.isCollapsed = true;
    Q(this.element).addClass('collapsed');
    Q(this.contentElement).hide();
        if (this.collapseButton) {
            Q(this.collapseButton).html('▶');
        }
    }

    expand() {
        if (!this.config.collapsible) return;

        this.isCollapsed = false;
    Q(this.element).removeClass('collapsed');
    Q(this.contentElement).show();
        if (this.collapseButton) {
            Q(this.collapseButton).html('▼');
        }
    }

    getElement() {
        return this.element;
    }

    getContentElement() {
        return this.contentElement;
    }

    setContent(content) {
        const contentElement = this.getContentElement();
        if (contentElement) {
            if (typeof content === 'string') {
                Q(contentElement).html(content);
            } else if (content instanceof Element) {
                Q(contentElement).empty();
                Q(contentElement).append(content);
            }
        }
    }

    destroy() {
        // Close floating window if open
        if (this.floatingWindow && this.isPoppedOut) {
            this.floatingWindow.close();
            this.floatingWindow = null;
        }
        
        if (this.element) {
            Q(this.element).remove();
        }
        this.element = null;
    }

    togglePopup() {
        if (this.isPoppedOut) {
            // Return content to box
            this.popIn();
        } else {
            // Move content to floating window
            this.popOut();
        }
    }

    popOut() {
        if (this.isPoppedOut || !this.contentElement) return;

        // Create floating window
        const windowId = `${this.config.id}-popup-window`;

        let storedWindowState = null;
        if (window.storageManager && typeof window.storageManager.get === 'function') {
            storedWindowState = window.storageManager.get('windows', windowId);
        } else if (window.WINDOW_SETTINGS?.windows?.[windowId]) {
            storedWindowState = window.WINDOW_SETTINGS.windows[windowId];
        }

        const windowWidth = storedWindowState?.width || 600;
        const windowHeight = storedWindowState?.height || 400;

        this.floatingWindow = new FloatingWindow(windowId, {
            title: this.config.title || 'Window',
            width: windowWidth,
            height: windowHeight,
            showButtons: false,
            closable: true,
            resizable: true,
            draggable: true,
            onClose: () => {
                this.popIn({ fromWindow: true });
            }
        });

        // Detach content from box and attach to window
        const content = this.contentElement;
        Q(content).detach();
        Q(this.floatingWindow.contentElement).append(content);

        // Hide the original box content area
        Q(this.element).addClass('popped-out');
        
        this.isPoppedOut = true;
        this.floatingWindow.open();

        // Update popup button icon to indicate "pop in"
        if (this.popupButton) {
            Q(this.popupButton).addClass('active');
        }
    }

    popIn({ fromWindow = false } = {}) {
        if (!this.isPoppedOut || !this.contentElement) return;

        // Detach content from window and return to box
        const content = this.contentElement;
        Q(content).detach();
        Q(this.element).append(content);

        // Show the box content area again
        Q(this.element).removeClass('popped-out');

        // Close and cleanup floating window
        const windowRef = this.floatingWindow;
        this.floatingWindow = null;
        this.isPoppedOut = false;

        if (!fromWindow && windowRef) {
            windowRef.close();
        }

        // Update popup button icon back to normal
        if (this.popupButton) {
            Q(this.popupButton).removeClass('active');
        }
    }
}

function createBox(container, config) {
    return new Box(container, config);
}
