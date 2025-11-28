class FrameTabs {
    constructor(identifier, tabsData = {}, options = {}) {
        this.identifier = identifier;
        this.tabsData = tabsData;
        this.options = {
            saveActiveTab: true,
            ...options
        };
        this.activeTab = null;
        this.tabContents = {};
        this.tabButtons = {};
        
        this.init();
    }
    
    init() { 
        this.createFrameContainer();
        this.loadActiveTabState();
        this.createTabs();
        this.setupEventListeners();
        this.setupResizeObserver();
    }
    
    createFrameContainer() {
        const existingContainer = Q('#' + this.identifier).get(0);
        if (existingContainer) {
            Q(existingContainer).remove();
        }
        
    this.frameWrapper = Q('<div>', { class: 'frame-tabs-wrapper' }).get(0);
        this.frameWrapper.id = this.identifier;
        
    this.tabsHeader = Q('<div>', { class: 'frame-tabs-header' }).get(0);
        
    this.tabsContent = Q('<div>', { class: 'frame-tabs-content' }).get(0);
        
    this.statusArea = Q('<div>', { class: 'frame-tabs-status' }).get(0);
        this.statusArea.id = `${this.identifier}-status`;
        
        // WebSocket info container (első helyen)
    this.websocketInfoArea = Q('<div>', { class: 'websocket-info' }).get(0);
        Q(this.statusArea).append(this.websocketInfoArea);
        
        // Other status elements container
    this.otherStatusArea = Q('<div>', { class: 'other-status' }).get(0);
        this.otherStatusArea.id = "status_area";
        Q(this.statusArea).append(this.otherStatusArea);
        
        Q(this.frameWrapper).append(this.tabsHeader, this.tabsContent);
        
        Q(document.body).append(this.frameWrapper);
    }
    
    createTabs() {
        this.clearTabs();
        
        const sortedTabs = Object.entries(this.tabsData)
            .sort((a, b) => (a[1].order || 0) - (b[1].order || 0));
        
    const tabButtonsContainer = Q('<div>', { class: 'frame-tabs-buttons' }).get(0);
        
        sortedTabs.forEach(([key, tab]) => {
            const tabButton = Q('<div>', { class: 'frame-tab-button', text: tab.title || key }).get(0);
            tabButton.dataset.tabKey = key;
            tabButton.title = tab.description || tab.title || key;
            
            if (tab.icon) {
                const iconElement = Q('<span>', { class: 'frame-tab-icon' }).get(0);
                Q(iconElement).html(tab.icon);
                tabButton.insertBefore(iconElement, tabButton.firstChild);
            }
            
            this.tabButtons[key] = tabButton;
            Q(tabButtonsContainer).append(tabButton);
            
            const tabContent = Q('<div>', { class: 'frame-tab-content' }).get(0);
            tabContent.dataset.tabKey = key;
            tabContent.id = `${this.identifier}-${key}`;
            
            if (tab.content) {
                if (typeof tab.content === 'string') {
                    Q(tabContent).html(tab.content);
                } else if (tab.content instanceof Element) {
                    Q(tabContent).append(tab.content);
                }
            }
            
            this.tabContents[key] = tabContent;
            Q(this.tabsContent).append(tabContent);
        });
        
        Q(this.tabsHeader).append(tabButtonsContainer, this.statusArea);
        
        if (sortedTabs.length > 0) {
            const defaultTab = this.savedActiveTab || sortedTabs[0][0];
            this.setActiveTab(defaultTab);
        }
    }
    
    setupEventListeners() {
        Q(this.tabsHeader).on("click", (event) => {
            if (Q(event.target).hasClass("frame-tab-button") || 
                event.target.closest(".frame-tab-button")) {
                const button = Q(event.target).hasClass("frame-tab-button") ? 
                    event.target : event.target.closest(".frame-tab-button");
                const tabKey = button.dataset.tabKey;
                this.setActiveTab(tabKey);
            }
        });
        
    Q(window).on("beforeunload", () => {
            if (this.options.saveActiveTab) {
                this.saveActiveTabState();
            }
        });
    }
    
    setupResizeObserver() {
        if (window.ResizeObserver) {
            this.resizeObserver = new ResizeObserver(() => {
                this.updateLayout();
            });
            this.resizeObserver.observe(this.frameWrapper);
        }
        
    Q(window).on("resize", () => {
            this.updateLayout();
        });
    }
    
    updateLayout() {
        const headerHeight = this.tabsHeader.offsetHeight;
        const viewportHeight = window.innerHeight;
        const contentHeight = viewportHeight - headerHeight;
        
        Q(this.tabsContent).css('height', `${contentHeight}px`);
    }
    
    setActiveTab(tabKey) {
        if (!this.tabsData[tabKey]) return;
        
        Object.keys(this.tabButtons).forEach(key => {
            Q(this.tabButtons[key]).removeClass("active");
            Q(this.tabContents[key]).removeClass("active");
        });
        
    Q(this.tabButtons[tabKey]).addClass("active");
    Q(this.tabContents[tabKey]).addClass("active");
        
        this.activeTab = tabKey;
        
        if (this.options.saveActiveTab) {
            this.saveActiveTabState();
        }
        
        this.updateLayout();
        this.dispatchTabChangeEvent(tabKey);
    }
    
    dispatchTabChangeEvent(tabKey) {
        Q(this.frameWrapper).trigger("frametab:change", {
            identifier: this.identifier,
            activeTab: tabKey,
            tabData: this.tabsData[tabKey]
        });
    }
    
    addTab(key, tabData) {
        this.tabsData[key] = tabData;
        this.createTabs();
    }
    
    removeTab(key) {
        if (this.tabsData[key]) {
            delete this.tabsData[key];
            
            if (this.activeTab === key) {
                const remainingTabs = Object.keys(this.tabsData);
                if (remainingTabs.length > 0) {
                    this.setActiveTab(remainingTabs[0]);
                }
            }
            
            this.createTabs();
        }
    }
    
    updateTab(key, tabData) {
        if (this.tabsData[key]) {
            this.tabsData[key] = { ...this.tabsData[key], ...tabData };
            this.createTabs();
        }
    }
    
    getActiveTab() {
        return this.activeTab;
    }
    
    getTabContent(tabKey) {
        return this.tabContents[tabKey];
    }
    
    getWebSocketInfoArea() {
        return this.websocketInfoArea;
    }
    
    getOtherStatusArea() {
        return this.otherStatusArea;
    }
    
    setStatusContent(content) {
        if (typeof content === 'string') {
            Q(this.otherStatusArea).html(content);
        } else if (content instanceof Element) {
            Q(this.otherStatusArea).empty();
            Q(this.otherStatusArea).append(content);
        }
    }
    
    setWebSocketInfo(content) {
        if (typeof content === 'string') {
            Q(this.websocketInfoArea).html(content);
        } else if (content instanceof Element) {
            Q(this.websocketInfoArea).empty();
            Q(this.websocketInfoArea).append(content);
        }
    }
    
    clearTabs() {
        if (this.tabsHeader) {
            Q(this.tabsHeader).empty();
        }
        if (this.tabsContent) {
            Q(this.tabsContent).empty();
        }
        if (this.websocketInfoArea) {
            Q(this.websocketInfoArea).empty();
        }
        if (this.otherStatusArea) {
            Q(this.otherStatusArea).empty();
        }
        this.tabButtons = {};
        this.tabContents = {};
    }
    
    saveActiveTabState() {
        if (!this.activeTab) return;
        
        const state = {
            activeTab: this.activeTab
        };
        
        waitForStorageManager((storageManager) => {
            if (storageManager) {
                storageManager.set('frameTabs', this.identifier, state);
            }
        });
    }
    
    loadActiveTabState() {
        waitForStorageManager((storageManager) => {
            if (!storageManager) return;
            
            const state = storageManager.get('frameTabs', this.identifier);
            if (state && state.activeTab) {
                this.savedActiveTab = state.activeTab;
            }
        });
    }
    
    destroy() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        
        if (this.frameWrapper) {
            Q(this.frameWrapper).remove();
        }
        
        this.tabButtons = {};
        this.tabContents = {};
        this.activeTab = null;
    }
}

function createFrameTabs(identifier, tabsData, options = {}) {
    return new FrameTabs(identifier, tabsData, options);
}
