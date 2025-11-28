class FolderBrowser {
    constructor(identifier, paths = [], placeholder = "Browse folders...") {
        this.identifier = identifier;
        this.paths = paths;
        this.placeholder = placeholder;
        this.selectedPath = '';
        this.selectedDisplayPath = '';
        this.isOpen = false;
        this.callback = null;
        this.folderStructure = {};
        this.pathSeparator = this.detectPathSeparator(paths);
        
        this.init();
    }

    init() {
        this.buildFolderStructure();
        this.createElement();
        this.setupEventListeners();
    }

    detectPathSeparator(paths) {
        if (!Array.isArray(paths)) {
            return '/';
        }

        for (const path of paths) {
            if (typeof path === 'string' && path.includes('\\')) {
                return '\\';
            }
        }
        return '/';
    }

    buildFolderStructure() {
        this.folderStructure = {};
        
        this.paths.forEach(path => {
            if (typeof path !== 'string') return;

            const normalizedPath = path.replace(/\\/g, '/');
            const parts = normalizedPath.split('/').filter(part => part.length > 0);
            const rawParts = path.split(/[\\/]/).filter(part => part.length > 0);
            const originalSeparator = path.includes('\\') ? '\\' : '/';

            let current = this.folderStructure;
            
            parts.forEach((part, index) => {
                const normalizedSubPath = parts.slice(0, index + 1).join('/');
                const displaySubPath = rawParts.slice(0, index + 1).join(this.pathSeparator);
                const originalSubPath = rawParts.slice(0, index + 1).join(originalSeparator);

                if (!current[part]) {
                    current[part] = {
                        isFolder: index < parts.length - 1,
                        children: {},
                        fullPath: normalizedSubPath,
                        originalPath: originalSubPath,
                        displayPath: displaySubPath
                    };
                } else {
                    if (!current[part].fullPath) {
                        current[part].fullPath = normalizedSubPath;
                    }
                    if (!current[part].originalPath) {
                        current[part].originalPath = originalSubPath;
                    }
                    if (!current[part].displayPath) {
                        current[part].displayPath = displaySubPath;
                    }
                }

                current = current[part].children;
            });
        });
    }

    createElement() {
    this.element = Q('<div>', { class: 'folder-browser' }).get(0);
        this.element.id = this.identifier;

    this.toggleButton = Q('<div>', { class: 'folder-browser-toggle' }).get(0);
        const toggleIconMarkup = (window.UI_ICONS && window.UI_ICONS.folder && window.UI_ICONS.folder.toggleButton)
            ? window.UI_ICONS.folder.toggleButton
            : '<svg class="folder-icon" width="16" height="16" viewBox="0 -960 960 960" fill="currentColor"><path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640v400q0 33-23.5 56.5T800-160H160Zm0-80h640v-400H447l-80-80H160v480Zm0 0v-480 480Z"/></svg>';
        Q(this.toggleButton).html(`
            ${toggleIconMarkup}
            <span class="folder-browser-text">${this.placeholder}</span>
            <span class="dropdown-arrow">▼</span>
        `);

    this.dropdown = Q('<div>', { class: 'folder-browser-dropdown' }).get(0);
        Q(this.dropdown).hide();

        this.buildDropdownContent();

        Q(this.element).append(this.toggleButton, this.dropdown);
    }

    buildDropdownContent() {
    Q(this.dropdown).empty();

    const allOption = Q('<div>', { class: 'folder-item' }).get(0);
        allOption.dataset.path = '';
        Q(allOption).html(`
            <span class="folder-item-text">All Folders</span>
        `);
    Q(allOption).on('click', () => this.selectPath('', ''));
        Q(this.dropdown).append(allOption);

    const separator = Q('<div>', { class: 'folder-separator' }).get(0);
        Q(this.dropdown).append(separator);

        this.renderFolderLevel(this.folderStructure, this.dropdown, 0);
    }

    renderFolderLevel(structure, container, level) {
        Object.entries(structure).forEach(([name, data]) => {
            const item = Q('<div>', { class: 'folder-item' }).get(0);
            item.dataset.path = data.originalPath;
            item.dataset.normalizedPath = data.fullPath;
            item.dataset.displayPath = data.displayPath;
            Q(item).css('paddingLeft', `${level}px`);

            const hasChildren = Object.keys(data.children).length > 0;
            
            const expandIconMarkup = hasChildren
                ? (window.UI_ICONS && window.UI_ICONS.folder && window.UI_ICONS.folder.expand
                    ? window.UI_ICONS.folder.expand
                    : '<svg class="folder-expand-icon" width="12" height="12" viewBox="0 0 24 24" fill="currentColor"><path d="M8.59 16.58L13.17 12L8.59 7.41L10 6L16 12L10 18L8.59 16.58Z"/></svg>')
                : '<span class="folder-spacer"></span>';

            const itemIconMarkup = (window.UI_ICONS && window.UI_ICONS.folder && window.UI_ICONS.folder.item)
                ? window.UI_ICONS.folder.item
                : '<svg class="folder-item-icon" width="14" height="14" viewBox="0 -960 960 960" fill="currentColor"><path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640v400q0 33-23.5 56.5T800-160H160Zm0-80h640v-400H447l-80-80H160v480Zm0 0v-480 480Z"/></svg>';

            Q(item).html(`
                ${expandIconMarkup}
                ${itemIconMarkup}
                <span class="folder-item-text">${name}</span>
            `);

            Q(item).on('click', (e) => {
                e.preventDefault();
                e.stopPropagation();
                
                if (hasChildren) {
                    this.toggleFolder(item, data.children, level + 1);
                } else {
                    this.selectPath(data.originalPath, data.displayPath);
                }
            });

            Q(container).append(item);

            if (hasChildren) {
                const childContainer = Q('<div>', { class: 'folder-children' }).get(0);
                Q(childContainer).hide();
                this.renderFolderLevel(data.children, childContainer, level + 1);
                Q(container).append(childContainer);
            }
        });
    }

    toggleFolder(folderItem, children, level) {
        const childContainer = folderItem.nextElementSibling;
        const expandIcon = folderItem.querySelector('.folder-expand-icon');
        const folderIcon = folderItem.querySelector('.folder-item-icon');
        
    if (childContainer && Q(childContainer).hasClass('folder-children')) {
            const isExpanded = childContainer.style.display !== 'none';
            
            if (isExpanded) {
                Q(childContainer).hide();
                Q(expandIcon).css('transform', 'rotate(0deg)');
                const closedPath = (window.UI_ICONS && window.UI_ICONS.folder && window.UI_ICONS.folder.itemPathClosed)
                    ? window.UI_ICONS.folder.itemPathClosed
                    : '<path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640v400q0 33-23.5 56.5T800-160H160Zm0-80h640v-400H447l-80-80H160v480Zm0 0v-480 480Z"/>';
                Q(folderIcon).html(closedPath);
            } else {
                Q(childContainer).show();
                Q(expandIcon).css('transform', 'rotate(90deg)');
                const openPath = (window.UI_ICONS && window.UI_ICONS.folder && window.UI_ICONS.folder.itemPathOpen)
                    ? window.UI_ICONS.folder.itemPathOpen
                    : '<path d="M160-160q-33 0-56.5-23.5T80-240v-480q0-33 23.5-56.5T160-800h240l80 80h320q33 0 56.5 23.5T880-640H447l-80-80H160v480l96-320h684L837-217q-8 26-29.5 41.5T760-160H160Zm84-80h516l72-240H316l-72 240Zm0 0 72-240-72 240Zm-84-400v-80 80Z"/>';
                Q(folderIcon).html(openPath);
            }
        }
    }

    selectPath(path, displayPath = null) {
        this.selectedPath = path;
        this.selectedDisplayPath = displayPath ?? path;
        
    const displayText = this.selectedDisplayPath ? this.selectedDisplayPath : this.placeholder;
    Q(this.toggleButton).find('.folder-browser-text').text(displayText);
        
        this.close();
        
        if (this.callback) {
            this.callback(path);
        }
        
    Q(this.element).trigger('pathChanged', { path, displayPath: this.selectedDisplayPath });
    }

    setupEventListeners() {
    Q(this.toggleButton).on('click', (e) => {
            e.preventDefault();
            e.stopPropagation();
            this.toggle();
        });

    Q(document).on('click', (e) => {
            if (!this.element.contains(e.target)) {
                this.close();
            }
        });

    Q(document).on('keydown', (e) => {
            if (e.key === 'Escape') {
                this.close();
            }
        });
    }

    toggle() {
        if (this.isOpen) {
            this.close();
        } else {
            this.open();
        }
    }

    open() {
    this.isOpen = true;
    Q(this.dropdown).show();
    Q(this.element).addClass('folder-browser-open');
    Q(this.toggleButton).find('.dropdown-arrow').css('transform', 'rotate(180deg)');
    }

    close() {
    this.isOpen = false;
    Q(this.dropdown).hide();
    Q(this.element).removeClass('folder-browser-open');
    Q(this.toggleButton).find('.dropdown-arrow').css('transform', 'rotate(0deg)');
    }

    updatePaths(paths) {
        this.paths = paths;
        this.pathSeparator = this.detectPathSeparator(paths);
        this.buildFolderStructure();
        this.buildDropdownContent();
    }

    getSelectedPath() {
        return this.selectedPath;
    }

    getSelectedDisplayPath() {
        return this.selectedDisplayPath;
    }

    setCallback(callback) {
        this.callback = callback;
    }

    getElement() {
        return this.element;
    }

    destroy() {
        if (this.element) {
            Q(this.element).remove();
        }
    }
}
