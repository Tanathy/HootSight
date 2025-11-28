class SplitBox {
    constructor(container, config) {
        this.container = container;
        this.config = config;
        this.panels = [];
        this.resizers = [];
        this.isResizing = false;
        this.currentResizer = null;
        this.startPosition = 0;
        this.startSizes = [];
        this.savedRatios = null;
        
        this.init();
    }

    init() {
    Q(this.container).addClass('splitbox-container');
        
        const direction = this.getDirection();
    Q(this.container).addClass(direction);
        
        // Scrollable property kezelése
        if (this.config.scrollable === false) {
            Q(this.container).addClass('no-scroll');
        } else if (this.config.scrollable === true) {
            Q(this.container).addClass('scrollable');
        }
        
        // Először hozzuk létre a paneleket alapértelmezett arányokkal
        this.createPanels();
        
        // Majd aszinkron módon betöltjük és alkalmazzuk a mentett állapotot
        this.loadSplitboxState();
        
        if (this.config.resizable) {
            this.createResizers();
            this.attachEvents();
        }
    }

    getDirection() {
        return 'horizontal';
    }

    createPanels() {
        const splits = this.config.splits || [];
        const fallbackRatios = this.config.split_ratios || [1.0];
        const fallbackColors = this.config.colors || [];
        
        if (splits.length > 0) {
            splits.forEach((split, index) => {
                const panel = Q('<div>', { class: 'split-panel' }).get(0);
                panel.id = split.id;
                panel.dataset.splitIndex = index;
                panel.style.flex = (split.ratio || 1.0).toString();
                
                if (split.colors) {
                    if (split.colors.startsWith('--')) {
                        panel.style.backgroundColor = `var(${split.colors})`;
                    } else {
                        panel.style.backgroundColor = split.colors;
                    }
                }
                
                const contentDiv = Q('<div>', { class: 'split-panel-content' }).get(0);
                
                if (split.scrollable) {
                    Q(contentDiv).addClass('scrollable');
                }
                
                if (split.title) {
                    const titleElement = Q('<h2>', { class: 'section_title split-panel-title', text: split.title }).get(0);
                    contentDiv.appendChild(titleElement);
                }
                
                if (split.description) {
                    const descElement = Q('<h4>', { class: 'section_description split-panel-description', text: split.description }).get(0);
                    contentDiv.appendChild(descElement);
                }
                
                if (split.tabs && split.tabs.length > 0) {
                    const tabsContainer = Q('<div>').get(0);
                    tabsContainer.id = `${split.id}-tabs-container`;
                    contentDiv.appendChild(tabsContainer);
                }
                
                panel.appendChild(contentDiv);
                this.panels.push(panel);
                this.container.appendChild(panel);
            });
        } else {
            // Most mindig az alapértelmezett arányokat használjuk, a mentett arányokat később alkalmazzuk
            const ratios = fallbackRatios;
            ratios.forEach((ratio, index) => {
                const panel = Q('<div>', { class: 'split-panel' }).get(0);
                
                const panelId = Array.isArray(this.config.id) ? this.config.id[index] : `${this.config.id}-panel-${index}`;
                panel.id = panelId;
                panel.dataset.splitIndex = index;
                panel.style.flex = ratio.toString();
                
                if (fallbackColors[index] && fallbackColors[index] !== null) {
                    if (fallbackColors[index].startsWith('--')) {
                        panel.style.backgroundColor = `var(${fallbackColors[index]})`;
                    } else {
                        panel.style.backgroundColor = fallbackColors[index];
                    }
                }
                
                const contentDiv = Q('<div>', { class: 'split-panel-content' }).get(0);
                panel.appendChild(contentDiv);
                
                this.panels.push(panel);
                this.container.appendChild(panel);
            });
        }
    }

    createResizers() {
        for (let i = 0; i < this.panels.length - 1; i++) {
            const resizer = Q('<div>', { class: 'split-resizer horizontal' }).get(0);
            resizer.dataset.leftIndex = i;
            resizer.dataset.rightIndex = i + 1;
            
            this.resizers.push(resizer);
            
            const leftPanel = this.panels[i];
            leftPanel.insertAdjacentElement('afterend', resizer);
        }
    }

    attachEvents() {
        this.resizers.forEach(resizer => {
            Q(resizer).on('mousedown', this.handleMouseDown.bind(this, resizer));
        });

        Q(document).on('mousemove', this.handleMouseMove.bind(this));
        Q(document).on('mouseup', this.handleMouseUp.bind(this));
    }

    handleMouseDown(resizer, event) {
        event.preventDefault();
        
        this.isResizing = true;
        this.currentResizer = resizer;
    Q(this.container).addClass('resizing');
    Q(resizer).addClass('dragging');
        
        this.startPosition = event.clientX;
        
        this.startSizes = this.panels.map(panel => panel.offsetWidth);
    }

    handleMouseMove(event) {
        if (!this.isResizing || !this.currentResizer) return;
        
        event.preventDefault();
        
        const resizer = this.currentResizer;
        const currentPosition = event.clientX;
        const delta = currentPosition - this.startPosition;
        
        const leftIndex = parseInt(resizer.dataset.leftIndex);
        const rightIndex = parseInt(resizer.dataset.rightIndex);
        
        const leftPanel = this.panels[leftIndex];
        const rightPanel = this.panels[rightIndex];
        
        const leftSize = this.startSizes[leftIndex] + delta;
        const rightSize = this.startSizes[rightIndex] - delta;
        
        const minSize = 10;
        
        if (leftSize >= minSize && rightSize >= minSize) {
            const containerSize = this.container.offsetWidth;
            
            const newSizes = [...this.startSizes];
            newSizes[leftIndex] = leftSize;
            newSizes[rightIndex] = rightSize;
            
            const totalSize = newSizes.reduce((sum, size) => sum + size, 0);
            
            this.panels.forEach((panel, index) => {
                const ratio = newSizes[index] / totalSize;
                panel.style.flex = ratio.toString();
            });
        }
    }

    handleMouseUp(event) {
        if (!this.isResizing) return;
        
        this.isResizing = false;
    Q(this.container).removeClass('resizing');
        
        if (this.currentResizer) {
            Q(this.currentResizer).removeClass('dragging');
            this.currentResizer = null;
        }
        
        this.startPosition = 0;
        this.startSizes = [];
        
        this.saveSplitboxState();
    }

    saveSplitboxState() {
        const ratios = this.getRatios();
        const configId = this.config.id || (this.config.splits && this.config.splits.length > 0 ? 
            this.config.splits.map(s => s.id).join('-') : 'unknown');
        
        const state = {
            split_ratios: ratios,
            id: configId
        };
        
        waitForStorageManager((storageManager) => {
            if (storageManager) {
                storageManager.set('splitboxes', configId, state);
            }
        });
    }

    loadSplitboxState() {
        const configId = this.config.id || (this.config.splits && this.config.splits.length > 0 ? 
            this.config.splits.map(s => s.id).join('-') : 'unknown');
        
        waitForStorageManager((storageManager) => {
            if (!storageManager) return;
            
            const state = storageManager.get('splitboxes', configId);
            if (state && state.split_ratios) {
                this.savedRatios = state.split_ratios;
                
                if (this.panels && this.panels.length > 0) {
                    this.applyRatios(this.savedRatios);
                }
            }
        });
    }

    getSavedRatios() {
        return this.savedRatios;
    }

    applyRatios(ratios) {
        if (!this.panels || this.panels.length === 0 || !ratios) return;
        
        this.panels.forEach((panel, index) => {
            if (ratios[index] !== undefined) {
                panel.style.flex = ratios[index].toString();
            }
        });
    }

    destroy() {
        this.resizers.forEach(resizer => {
            if (resizer.parentNode) {
                resizer.parentNode.removeChild(resizer);
            }
        });
        
        this.panels.forEach(panel => {
            if (panel.parentNode) {
                panel.parentNode.removeChild(panel);
            }
        });
        
        this.panels = [];
        this.resizers = [];
    }

    updateRatios(newRatios) {
        this.config.split_ratios = newRatios;
        this.panels.forEach((panel, index) => {
            panel.style.flex = newRatios[index].toString();
        });
        this.saveSplitboxState();
    }

    getPanelById(panelId) {
        return this.panels.find(panel => panel.id === panelId);
    }

    getRatios() {
        return this.panels.map(panel => parseFloat(panel.style.flex) || 1);
    }
}

function createSplitBox(container, config) {
    return new SplitBox(container, config);
}
