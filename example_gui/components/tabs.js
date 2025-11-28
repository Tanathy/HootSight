class Tabs {
    constructor(identifier, tabsData = {}, extraClassOrOptions = "", maybeOptions = undefined) {
        this.identifier = identifier;
        this.tabsData = tabsData;
        this.activeTab = null;
        this.savedActiveTab = null;
        this.tabContents = {};
        // Back-compat for third arg being extra class or options
        this.options = { sticky: false, stickyOffset: 0 };
        let extraClass = "";
        if (typeof extraClassOrOptions === "string") {
            extraClass = extraClassOrOptions || "";
            if (maybeOptions && typeof maybeOptions === "object") {
                this.options = { ...this.options, ...maybeOptions };
            }
        } else if (extraClassOrOptions && typeof extraClassOrOptions === "object") {
            this.options = { ...this.options, ...extraClassOrOptions };
        }

    this.tabsWrapper = Q('<div>', { class: 'tabs_wrapper' }).get(0);
        this.tabsWrapper.setAttribute("id", identifier);
        
        if (extraClass) {
            Q(this.tabsWrapper).addClass(extraClass);
        }
        
    this.tabsHeader = Q('<div>', { class: 'tabs_header' }).get(0);
    this.tabsContent = Q('<div>', { class: 'tabs_content' }).get(0);
    this.statusArea = Q('<div>', { class: 'tabs_status' }).get(0);
        
        this.tabsWrapper.appendChild(this.tabsHeader);
        this.tabsWrapper.appendChild(this.tabsContent);
        
        this.loadActiveTabState();
        this.createTabs();
        this.setupEventListeners();

        if (this.options.sticky) {
            this.enableStickyHeader();
        }
    }
    
    createTabs() {
        const sortedTabs = Object.entries(this.tabsData).sort((a, b) => a[1].order - b[1].order);
        
        sortedTabs.forEach(([key, tab]) => {
            const tabButton = Q('<div>', { class: 'tab_button' }).get(0);
            Q(tabButton).text(tab.title);
            tabButton.setAttribute("data-tab", key);
            this.tabsHeader.appendChild(tabButton);
            
            this.tabContents[key] = Q('<div>', { class: 'tab_content' }).get(0);
            this.tabContents[key].setAttribute("data-tab", key);
            this.tabContents[key].id = `${key}`;
            this.tabsContent.appendChild(this.tabContents[key]);
        });
        
        this.tabsHeader.appendChild(this.statusArea);
        
        const firstTab = this.savedActiveTab && this.tabsData[this.savedActiveTab] ? 
            this.savedActiveTab : (sortedTabs.length > 0 ? sortedTabs[0][0] : null);
        
        if (firstTab) {
            this.setActiveTab(firstTab);
        }
    }
    
    setupEventListeners() {
        Q(this.tabsHeader).on("click", (event) => {
            if (Q(event.target).hasClass("tab_button")) {
                const tabKey = event.target.getAttribute("data-tab");
                this.setActiveTab(tabKey);
            }
        });
    }

    // Sticky header support: fixes header to viewport top and offsets content accordingly
    enableStickyHeader() {
        if (this._stickyEnabled) return;
        this._stickyEnabled = true;

        Q(this.tabsWrapper).addClass('tabs_sticky_enabled');
        Q(this.tabsHeader).addClass('tabs_header_sticky');

        // Handlers
        this._stickyUpdate = () => this.updateStickyLayout();
        this._stickyRO = new ResizeObserver(() => this.updateStickyLayout());
        this._stickyRO.observe(this.tabsHeader);
        this._stickyRO.observe(this.tabsWrapper);
        const frameHeader = document.querySelector('.frame-tabs-header');
        if (frameHeader) {
            this._stickyRO.observe(frameHeader);
        }
        window.addEventListener('resize', this._stickyUpdate, { passive: true });
        window.addEventListener('scroll', this._stickyUpdate, { passive: true });

        // Initial
        this.updateStickyLayout();
    }

    disableStickyHeader() {
        if (!this._stickyEnabled) return;
        this._stickyEnabled = false;
        Q(this.tabsWrapper).removeClass('tabs_sticky_enabled');
        Q(this.tabsHeader).removeClass('tabs_header_sticky');

        // Cleanup inline styles
        Object.assign(this.tabsHeader.style, { position: '', top: '', left: '', right: '', width: '', zIndex: '' });
        this.tabsContent.style.marginTop = '';

        if (this._stickyRO) {
            this._stickyRO.disconnect();
            this._stickyRO = null;
        }
        window.removeEventListener('resize', this._stickyUpdate);
        window.removeEventListener('scroll', this._stickyUpdate);
        this._stickyUpdate = null;
    }

    updateStickyLayout() {
        if (!this._stickyEnabled) return;
    // Align header width and left to wrapper
    const wrapperRect = this.tabsWrapper.getBoundingClientRect();
    // Determine global offset (e.g., FrameTabs header height)
        const frameHeader = document.querySelector('.frame-tabs-header');
        let frameOffset = 0;
        if (frameHeader) {
            const fcs = window.getComputedStyle(frameHeader);
            const fMarginTop = parseFloat(fcs.marginTop) || 0;
            const fMarginBottom = parseFloat(fcs.marginBottom) || 0;
            frameOffset = frameHeader.offsetHeight + fMarginTop + fMarginBottom;
        }
        const topOffset = frameOffset + (this.options.stickyOffset || 0);
    // Place header at viewport top + frame header height, aligned to wrapper horizontally
    this.tabsHeader.style.position = 'fixed';
    this.tabsHeader.style.top = `${topOffset}px`;
        this.tabsHeader.style.left = `${Math.max(0, wrapperRect.left)}px`;
        this.tabsHeader.style.width = `${wrapperRect.width}px`;
        this.tabsHeader.style.right = '';
    // Keep below main frame header; avoid extremely high z-order
    this.tabsHeader.style.zIndex = '2';

        // Compute outer height (offsetHeight + vertical margins)
        const cs = window.getComputedStyle(this.tabsHeader);
        const marginTop = parseFloat(cs.marginTop) || 0;
        const marginBottom = parseFloat(cs.marginBottom) || 0;
        const outerHeight = (this.tabsHeader.offsetHeight || 0) + marginTop + marginBottom + (this.options.stickyOffset || 0);
        this.tabsContent.style.marginTop = `${outerHeight}px`;
    }
    
    setActiveTab(tabKey) {
        if (this.activeTab) {
            this.saveTabContent(this.activeTab);
        }
        
        this.tabsHeader.querySelectorAll(".tab_button").forEach(button => {
            Q(button).removeClass("active");
        });
        
        this.tabsContent.querySelectorAll(".tab_content").forEach(content => {
            Q(content).removeClass("active");
        });
        
        const activeButton = this.tabsHeader.querySelector(`[data-tab="${tabKey}"]`);
        const activeContent = this.tabsContent.querySelector(`[data-tab="${tabKey}"]`);
        
        if (activeButton && activeContent) {
            Q(activeButton).addClass("active");
            Q(activeContent).addClass("active");
            this.activeTab = tabKey;
            this.restoreTabContent(tabKey);
            this.saveActiveTabState();
        }
    }
    
    saveTabContent(tabKey) {
        const tabContent = this.tabContents[tabKey];
        if (!tabContent) return;
        
        const inputs = tabContent.querySelectorAll("input, textarea, select");
        inputs.forEach(input => {
            if (input.type === "checkbox" || input.type === "radio") {
                input.dataset.savedChecked = input.checked;
            } else {
                input.dataset.savedValue = input.value;
            }
        });
    }
    
    restoreTabContent(tabKey) {
        const tabContent = this.tabContents[tabKey];
        if (!tabContent) return;
        
        const inputs = tabContent.querySelectorAll("input, textarea, select");
        inputs.forEach(input => {
            if (input.type === "checkbox" || input.type === "radio") {
                if (input.dataset.savedChecked !== undefined) {
                    input.checked = input.dataset.savedChecked === "true";
                }
            } else {
                if (input.dataset.savedValue !== undefined) {
                    input.value = input.dataset.savedValue;
                }
            }
        });
    }
    
    addContentToTab(tabKey, content) {
        if (this.tabContents[tabKey]) {
            if (typeof content === "string") {
                Q(this.tabContents[tabKey]).html(content);
            } else {
                Q(this.tabContents[tabKey]).append(content);
            }
        }
    }
    
    clearTabContent(tabKey) {
        if (this.tabContents[tabKey]) {
            Q(this.tabContents[tabKey]).empty();
        }
    }
    
    getElement() {
        return this.tabsWrapper;
    }
    
    getTabContent(tabKey) {
        return this.tabContents[tabKey];
    }
    
    getCurrentTab() {
        return this.activeTab;
    }
    
    getStatusArea() {
        return this.statusArea;
    }
    
    addStatusElement(element) {
        this.statusArea.appendChild(element);
    }
    
    clearStatusArea() {
    Q(this.statusArea).empty();
    }

    saveActiveTabState() {
        if (!this.activeTab) return;
        
        waitForStorageManager((storageManager) => {
            if (!storageManager) return;
            
            const state = {
                activeTab: this.activeTab
            };
            
            storageManager.set('tabs', this.identifier, state);
        });
    }

    loadActiveTabState() {
        waitForStorageManager((storageManager) => {
            if (!storageManager) return;
            
            const state = storageManager.get('tabs', this.identifier);
            if (state && state.activeTab) {
                this.savedActiveTab = state.activeTab;
                
                if (this.tabsData[this.savedActiveTab] && !this.activeTab) {
                    setTimeout(() => this.setActiveTab(this.savedActiveTab), 0);
                }
            }
        });
    }
}
