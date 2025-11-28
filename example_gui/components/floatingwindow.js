class FloatingWindow {
    constructor(identifier, options = {}) {
        this.identifier = identifier;
        this.options = {
            title: 'Window',
            content: '',
            width: 400,
            height: 300,
            minWidth: 200,
            minHeight: 150,
            resizable: true,
            draggable: true,
            closable: true,
            animate: true,
            animationDuration: 150,
            showButtons: true,
            onApply: null,
            onCancel: null,
            onClose: null,
            ...options
        };

        this.element = null;
        this.titlebarElement = null;
        this.contentElement = null;
        this.footerElement = null;
        this.isOpen = false;
        this.isAnimating = false;
        this.isDragging = false;
        this.isResizing = false;

        this.init();
    }

    init() {
        this.createWindowElement();
        this.setupEventListeners();
        this.loadWindowState();
        
        if (!FloatingWindow.openWindows) {
            FloatingWindow.openWindows = [];
            FloatingWindow.highestZIndex = 1000;
        }
    }

    createWindowElement() {
    this.element = Q('<div>', { class: 'window_container' }).get(0);
    Q(this.element).attr("id", this.identifier);
        Q(this.element).css('width', this.options.width + "px");
        Q(this.element).css('height', this.options.height + "px");
        Q(this.element).css('zIndex', this.options.zIndex || 1000);

        this.titlebarElement = Q('<div>', { class: 'window_titlebar' }).get(0);

    const titleElement = Q('<h2>', { class: 'window_title', text: this.options.title }).get(0);

        const controlsElement = Q('<div>', { class: 'window_controls' }).get(0);

        if (this.options.closable) {
            const closeBtn = Q('<button>', { class: 'window_button window_close' }).get(0);
            const closeIcon = (window.UI_ICONS && window.UI_ICONS.windowClose)
                ? window.UI_ICONS.windowClose
                : '<svg class="window_button_icon" viewBox="0 0 12 12"><path d="M1,1 L11,11 M11,1 L1,11" stroke="currentColor" stroke-width="1.5"/></svg>';
            Q(closeBtn).html(closeIcon);
            Q(controlsElement).append(closeBtn);
        }

        Q(this.titlebarElement).append(titleElement, controlsElement);

    this.contentElement = Q('<div>', { class: 'window_content' }).get(0);

    this.footerElement = Q('<div>', { class: 'window_footer' }).get(0);

        if (this.options.showButtons) {
            // Dynamic footer buttons if provided by schema
            if (Array.isArray(this.options.footerButtons) && this.options.footerButtons.length > 0) {
                this.options.footerButtons.forEach((btn) => {
                    const label = (typeof lang === 'function') ? lang(btn.label || '') : (btn.label || '');
                    const buttonEl = Q('<div>', { class: 'button_wrapper', text: label || '' }).get(0);
                    // Apply variant class if any
                    if (btn.class) {
                        Q(buttonEl).addClass(btn.class);
                    } else if (btn.primary) {
                        Q(buttonEl).addClass('btn-primary');
                    } else if (btn.action === 'cancel') {
                        Q(buttonEl).addClass('btn-secondary');
                    }
                    // Encode action identity
                    buttonEl.setAttribute("data-action", btn.action || '');
                    // Attach click
                    Q(buttonEl).on('click', () => {
                        if (btn.action === 'cancel') {
                            if (typeof this.options.onCancel === 'function') {
                                try { this.options.onCancel(); } catch (e) { console.error(e); this.close(); }
                            } else { this.close(); }
                        } else if (btn.action === 'save' || btn.primary) {
                            if (typeof this.options.onApply === 'function') {
                                try { this.options.onApply(); } catch (e) { console.error(e); }
                            } else { this.close(); }
                        } else if (btn.action) {
                            // First try ComponentsBuilder action execution
                            try {
                                if (window.COMPONENTS_BUILDER && typeof window.COMPONENTS_BUILDER.executeAction === 'function') {
                                    window.COMPONENTS_BUILDER.executeAction(btn.action, { id: `window-btn-${btn.action}`, assign: btn.assign || '' });
                                } else if (typeof window[btn.action] === 'function') {
                                    window[btn.action]();
                                }
                            } catch (e) { 
                                console.error('Footer button action error:', e); 
                                // Try onApply as fallback for primary buttons
                                if (btn.primary && typeof this.options.onApply === 'function') {
                                    try { this.options.onApply(); } catch (e2) { console.error(e2); }
                                }
                            }
                        }
                    });
                    Q(this.footerElement).append(buttonEl);
                });
            } else {
                // Fallback default Cancel/Apply
                const cancelBtn = Q('<div>', { class: 'button_wrapper btn-secondary', text: (typeof lang === 'function') ? lang('ui.button.cancel') : 'Cancel' }).get(0);
                cancelBtn.setAttribute("data-action", "cancel");
                const applyBtn = Q('<div>', { class: 'button_wrapper btn-primary', text: (typeof lang === 'function') ? lang('ui.button.apply') : 'Apply' }).get(0);
                applyBtn.setAttribute("data-action", "apply");
                Q(this.footerElement).append(cancelBtn, applyBtn);
            }
        }

        if (this.options.resizable) {
            const resizeHandles = [
                { class: "window_resize_handle window_resize_n", data: "n" },
                { class: "window_resize_handle window_resize_e", data: "e" },
                { class: "window_resize_handle window_resize_s", data: "s" },
                { class: "window_resize_handle window_resize_w", data: "w" },
                { class: "window_resize_handle window_resize_nw", data: "nw" },
                { class: "window_resize_handle window_resize_ne", data: "ne" },
                { class: "window_resize_handle window_resize_se", data: "se" },
                { class: "window_resize_handle window_resize_sw", data: "sw" }
            ];
            
            resizeHandles.forEach(handle => {
                const handleElement = Q('<div>', { class: handle.class }).get(0);
                handleElement.setAttribute("data-resize", handle.data);
                Q(this.element).append(handleElement);
            });
        }

        Q(this.element).append(this.titlebarElement, this.contentElement, this.footerElement);

        if (this.options.content) {
            this.setContent(this.options.content);
        }
    }

    setupEventListeners() {
    Q(this.element).on("click", (e) => {
            if (e.target.closest(".window_close")) {
                this.close();
            }
        });

    Q(this.footerElement).on("click", (e) => {
            const action = e.target.getAttribute("data-action");
            
            if (action === "cancel") {
                if (this.options.onCancel && typeof this.options.onCancel === "function") {
                    try {
                        this.options.onCancel();
                    } catch (error) {
                        console.error("Error in onCancel callback:", error);
                        this.close();
                    }
                } else {
                    this.close();
                }
            } else if (action === "apply") {
                if (this.options.onApply && typeof this.options.onApply === "function") {
                    try {
                        this.options.onApply();
                    } catch (error) {
                        console.error("Error in onApply callback:", error);
                    }
                } else {
                    // Try global saveSettings as fallback
                    try {
                        if (typeof window.saveSettings === 'function') {
                            window.saveSettings();
                        }
                    } catch (error) {
                        console.error("Error calling saveSettings fallback:", error);
                        this.close();
                    }
                }
            }
        });

        if (this.options.draggable) {
            this.setupDragging();
        }

        if (this.options.resizable) {
            this.setupResizing();
        }

    Q(this.element).on("mousedown", () => this.bringToFront());

    Q(document).on("keydown", (e) => {
            if (e.key === "Escape" && this.isOpen) {
                this.close();
            }
        });

        this.setupWindowResize();
    }

    setupWindowResize() {
        let resizeTimeout;
        const handleWindowResize = () => {
            clearTimeout(resizeTimeout);
            resizeTimeout = setTimeout(() => {
                if (this.isOpen) {
                    this.constrainToViewport();
                }
            }, 500);
        };

    Q(window).on("resize", handleWindowResize);
        
        this.windowResizeHandler = handleWindowResize;
    }

    constrainToViewport() {
        if (!this.element) return;

        const rect = this.element.getBoundingClientRect();
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        let newLeft = parseInt(this.element.style.left, 10) || 0;
        let newTop = parseInt(this.element.style.top, 10) || 0;
        let newWidth = this.element.offsetWidth;
        let newHeight = this.element.offsetHeight;
        let changed = false;

        if (newWidth > viewportWidth) {
            newWidth = Math.max(viewportWidth - 20, this.options.minWidth);
            Q(this.element).css('width', newWidth + "px");
            changed = true;
        }

        if (newHeight > viewportHeight) {
            newHeight = Math.max(viewportHeight - 20, this.options.minHeight);
            Q(this.element).css('height', newHeight + "px");
            changed = true;
        }

        if (newLeft + newWidth > viewportWidth) {
            newLeft = Math.max(0, viewportWidth - newWidth);
            Q(this.element).css('left', newLeft + "px");
            changed = true;
        }

        if (newTop + newHeight > viewportHeight) {
            newTop = Math.max(0, viewportHeight - newHeight);
            Q(this.element).css('top', newTop + "px");
            changed = true;
        }

        if (newLeft < 0) {
            Q(this.element).css('left', "0px");
            changed = true;
        }

        if (newTop < 0) {
            Q(this.element).css('top', "0px");
            changed = true;
        }

        if (changed) {
            this.saveWindowState();
        }
    }

    setupDragging() {
        let startX, startY, startLeft, startTop;

    Q(this.titlebarElement).on("mousedown", (e) => {
            this.isDragging = true;
            startX = e.clientX;
            startY = e.clientY;
            startLeft = parseInt(this.element.style.left, 10) || 0;
            startTop = parseInt(this.element.style.top, 10) || 0;
            
            this.bringToFront();
            
            const handleMouseMove = (e) => {
                if (!this.isDragging) return;
                
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                const newLeft = startLeft + dx;
                const newTop = startTop + dy;
                
                const maxLeft = window.innerWidth - this.element.offsetWidth;
                const maxTop = window.innerHeight - this.element.offsetHeight;
                
                const constrainedLeft = Math.max(0, Math.min(newLeft, maxLeft));
                const constrainedTop = Math.max(0, Math.min(newTop, maxTop));
                
                Q(this.element).css('left', constrainedLeft + "px");
                Q(this.element).css('top', constrainedTop + "px");
            };
            
            const handleMouseUp = () => {
                this.isDragging = false;
                Q(document).off("mousemove", handleMouseMove);
                Q(document).off("mouseup", handleMouseUp);
                this.saveWindowState();
            };
            
            Q(document).on("mousemove", handleMouseMove);
            Q(document).on("mouseup", handleMouseUp);
            
            e.preventDefault();
        });
    }

    setupResizing() {
        let resizeDirection = "";
        let startX, startY, startWidth, startHeight, startLeft, startTop;

    Q(this.element).on("mousedown", (e) => {
            if (!Q(e.target).hasClass("window_resize_handle")) return;
            
            this.isResizing = true;
            resizeDirection = e.target.getAttribute("data-resize");
            
            startX = e.clientX;
            startY = e.clientY;
            startWidth = this.element.offsetWidth;
            startHeight = this.element.offsetHeight;
            startLeft = parseInt(this.element.style.left, 10) || 0;
            startTop = parseInt(this.element.style.top, 10) || 0;
            
            const handleMouseMove = (e) => {
                if (!this.isResizing) return;
                
                const dx = e.clientX - startX;
                const dy = e.clientY - startY;
                
                let newWidth = startWidth;
                let newHeight = startHeight;
                let newLeft = startLeft;
                let newTop = startTop;
                
                if (resizeDirection.includes("e")) {
                    newWidth = startWidth + dx;
                }
                if (resizeDirection.includes("s")) {
                    newHeight = startHeight + dy;
                }
                if (resizeDirection.includes("w")) {
                    newWidth = startWidth - dx;
                    newLeft = startLeft + dx;
                }
                if (resizeDirection.includes("n")) {
                    newHeight = startHeight - dy;
                    newTop = startTop + dy;
                }
                
                if (newWidth < this.options.minWidth) {
                    if (resizeDirection.includes("w")) {
                        newLeft = startLeft + startWidth - this.options.minWidth;
                    }
                    newWidth = this.options.minWidth;
                }
                
                if (newHeight < this.options.minHeight) {
                    if (resizeDirection.includes("n")) {
                        newTop = startTop + startHeight - this.options.minHeight;
                    }
                    newHeight = this.options.minHeight;
                }
                
                const maxLeft = window.innerWidth - newWidth;
                const maxTop = window.innerHeight - newHeight;
                
                newLeft = Math.max(0, Math.min(newLeft, maxLeft));
                newTop = Math.max(0, Math.min(newTop, maxTop));
                
                Q(this.element).css('width', newWidth + "px");
                Q(this.element).css('height', newHeight + "px");
                Q(this.element).css('left', newLeft + "px");
                Q(this.element).css('top', newTop + "px");
            };
            
            const handleMouseUp = () => {
                this.isResizing = false;
                Q(document).off("mousemove", handleMouseMove);
                Q(document).off("mouseup", handleMouseUp);
                this.saveWindowState();
            };
            
            Q(document).on("mousemove", handleMouseMove);
            Q(document).on("mouseup", handleMouseUp);
            
            e.preventDefault();
            e.stopPropagation();
        });
    }

    saveWindowState() {
        const state = {
            left: parseInt(this.element.style.left, 10) || 0,
            top: parseInt(this.element.style.top, 10) || 0,
            width: this.element.offsetWidth,
            height: this.element.offsetHeight
        };
        
        waitForStorageManager((storageManager) => {
            if (storageManager) {
                storageManager.set('windows', this.identifier, state);
            }
        });
    }

    loadWindowState() {
        waitForStorageManager((storageManager) => {
            if (!storageManager) return;
            
            const state = storageManager.get('windows', this.identifier);
            if (state) {
                if (typeof state.width === 'number') {
                    this.options.width = state.width;
                    if (this.element) {
                        Q(this.element).css('width', `${state.width}px`);
                    }
                }

                if (typeof state.height === 'number') {
                    this.options.height = state.height;
                    if (this.element) {
                        Q(this.element).css('height', `${state.height}px`);
                    }
                }

                this.savedPosition = { left: state.left, top: state.top };

                if (this.element && typeof state.left === 'number') {
                    Q(this.element).css('left', `${state.left}px`);
                }

                if (this.element && typeof state.top === 'number') {
                    Q(this.element).css('top', `${state.top}px`);
                }

                if (this.isOpen) {
                    this.restorePosition();
                }
            }
        });
    }

    restorePosition() {
        if (this.savedPosition) {
            const maxLeft = window.innerWidth - this.element.offsetWidth;
            const maxTop = window.innerHeight - this.element.offsetHeight;
            
            const constrainedLeft = Math.max(0, Math.min(this.savedPosition.left, maxLeft));
            const constrainedTop = Math.max(0, Math.min(this.savedPosition.top, maxTop));
            
            Q(this.element).css('left', constrainedLeft + "px");
            Q(this.element).css('top', constrainedTop + "px");
        }
    }

    centerWindow() {
        const windowWidth = this.options.width;
        const windowHeight = this.options.height;
        const viewportWidth = window.innerWidth;
        const viewportHeight = window.innerHeight;

        const left = (viewportWidth - windowWidth) / 2;
        const top = (viewportHeight - windowHeight) / 2;

        Q(this.element).css('left', left + "px");
        Q(this.element).css('top', top + "px");
    }

    bringToFront() {
        const index = FloatingWindow.openWindows.indexOf(this);
        if (index !== -1) {
            FloatingWindow.openWindows.splice(index, 1);
        }
        FloatingWindow.openWindows.push(this);
        this.updateZIndices();
    }

    updateZIndices() {
        const baseZIndex = 1000;
        FloatingWindow.openWindows.forEach((window, index) => {
            Q(window.element).css('zIndex', baseZIndex + index);
        });
        FloatingWindow.highestZIndex = baseZIndex + FloatingWindow.openWindows.length - 1;
    }

    open() {
        if (this.isOpen || this.isAnimating) return this;

        Q(document.body).append(this.element);

        // Ensure state is fresh when reopening
    Q(this.element).removeClass("window_fade_out");
        // Re-attach resize handler if it was removed on previous close
        if (!this.windowResizeHandler) {
            this.setupWindowResize();
        }

        if (this.savedPosition) {
            this.restorePosition();
        } else {
            this.centerWindow();
        }

        if (this.options.animate) {
            this.isAnimating = true;
            
            Q(this.element).css('display', 'flex');
            this.element.offsetHeight;
            
            Q(this.element).addClass("window_show");
            
            setTimeout(() => {
                this.isAnimating = false;
                this.isOpen = true;
            }, this.options.animationDuration);
        } else {
            Q(this.element).css('display', 'flex');
            this.isOpen = true;
        }

        this.bringToFront();
        return this;
    }

    close() {
        if (!this.isOpen || this.isAnimating) return this;

        this.saveWindowState();

        // Call onClose callback if provided
        if (typeof this.options.onClose === 'function') {
            try {
                this.options.onClose();
            } catch (e) {
                console.error('Error in onClose callback:', e);
            }
        }

        if (this.options.animate) {
            this.isAnimating = true;
            
            Q(this.element).removeClass("window_show");
            Q(this.element).addClass("window_fade_out");
            
            setTimeout(() => {
                // Hide without destroying instance
                if (this.element.parentNode) {
                    this.element.parentNode.removeChild(this.element);
                }
                // Keep resize handler to allow re-open constraints; it will be reattached if needed
                this.isAnimating = false;
                this.isOpen = false;
            }, this.options.animationDuration);
        } else {
            if (this.element.parentNode) {
                this.element.parentNode.removeChild(this.element);
            }
            this.isOpen = false;
        }

        // Remove from open stack but keep instance alive for reuse
        const index = FloatingWindow.openWindows.indexOf(this);
        if (index !== -1) {
            FloatingWindow.openWindows.splice(index, 1);
            this.updateZIndices();
        }

        return this;
    }

    cleanup() {
        // Legacy cleanup (not used in hide behavior)
        if (this.element.parentNode) {
            this.element.parentNode.removeChild(this.element);
        }
        const index = FloatingWindow.openWindows.indexOf(this);
        if (index !== -1) {
            FloatingWindow.openWindows.splice(index, 1);
        }
        if (this.windowResizeHandler) {
            Q(window).off("resize", this.windowResizeHandler);
            this.windowResizeHandler = null;
        }
        this.updateZIndices();
    }

    setContent(content) {
        Q(this.contentElement).empty();
        if (typeof content === "string") {
            Q(this.contentElement).html(content);
        } else {
            Q(this.contentElement).append(content);
        }
        return this;
    }

    setTitle(title) {
        this.options.title = title;
    Q(this.titlebarElement.querySelector('.window_title')).text(title);
        return this;
    }

    getElement() {
        return this.element;
    }

    // Compatibility helper for consumers expecting DOM element
    getContentElement() {
        return this.contentElement;
    }

    getContent() {
    return this.contentElement.innerHTML;
    }

    getTitle() {
        return this.options.title;
    }

    isWindowOpen() {
        return this.isOpen;
    }

    // Bring window to front and focus the content for accessibility
    focus() {
        this.bringToFront();
        try {
            this.element?.focus?.();
        } catch (_e) {}
    }
}

FloatingWindow.openWindows = [];
FloatingWindow.highestZIndex = 1000;
