class modelCard {
    constructor(identifier, config = {}) {
        this.identifier = identifier;
        this.config = config;
        this.modelData = [];
        this.callback = config.callback || null;
        this.apiEndpoint = config.api_endpoint || '/models';
        this.apiLoadEndpoint = config.api_endpoint_load || '/load';
        this.apiUnloadEndpoint = config.api_endpoint_unload || '/unload';
        this.apiMethod = config.api_endpoint_method || 'GET';
        this.componentType = config.component_type || 'unknown'; // e.g., 'models', 'loras', 'lycoris', 'vaes'
        this.searchTerm = '';
        this.selectedType = '__all__';
        this.selectedPath = '';
        this.sortOption = 'a-to-z'; // Default sorting option
        this.favorites = new Set();
        this.modelStats = {};
        
        this.loadedModels = new Set();
        this.modelsInUse = new Set();
        this.modelInUseMaximum = config.model_in_use_maximum || 1;

        // Register this instance in the global registry
        modelCard.registerInstance(this.componentType, this);

        this.modelCardElement = Q('<div>', { class: 'model_card_wrapper', id: identifier }).get(0);

        this.filtersContainer = Q('<div>', { class: 'model_filters' }).get(0);

        this.folderBrowser = new FolderBrowser(`${identifier}_folder_browser`, [], lang('ui.model_card.all_folders'));
        this.folderBrowser.setCallback((path) => {
            this.selectedPath = path;
            this.render();
        });

        this.searchInput = new Input("model_search", "text", "", "", "", lang('ui.model_card.search_placeholder'));
        Q(this.searchInput.getElement()).on("change", (e) => {
            this.searchTerm = e.target.value.toLowerCase();
            this.render();
        });

        // Create refresh button
        this.refreshButton = Q('<button>', { 
            class: 'model_card_refresh_button',
            'data-tooltip': 'ui.model_card.refresh_tooltip'
        }).get(0);
        Q(this.refreshButton).html(window.UI_ICONS.reload);
        Q(this.refreshButton).on('click', () => {
            this.refreshModels();
        });

        this.typeDropdown = new Dropdown("architecture_filter", [lang('ui.model_card.all')], lang('ui.model_card.all'));
        this.typeDropdown.updateOptions({
            options: [lang('ui.model_card.all')],
            valueMapping: {
                [lang('ui.model_card.all')]: '__all__'
            }
        });
        this.typeDropdown.setOnChange(() => {
            this.selectedType = this.typeDropdown.get();
            this.render();
        });
        this.typeDropdown.set('__all__');

        // Add sorting and filter dropdown
        this.sortDropdown = new Dropdown("model_sort_filter", this._getSortOptions(), lang('ui.model_card.sort_a_to_z'));
        this.sortDropdown.updateOptions({
            options: this._getSortOptions(),
            valueMapping: this._getSortMapping()
        });
        this.sortDropdown.setOnChange(() => {
            this.sortOption = this.sortDropdown.get();
            this.render();
        });
        this.sortDropdown.set('a-to-z');

        Q(this.filtersContainer).append(this.folderBrowser.getElement());
        Q(this.filtersContainer).append(this.typeDropdown.getElement());
        Q(this.filtersContainer).append(this.sortDropdown.getElement());
        Q(this.filtersContainer).append(this.searchInput.getElement());
        Q(this.filtersContainer).append(this.refreshButton);

        this.cardsContainer = Q('<div>', { class: 'model_card' }).get(0);

        Q(this.modelCardElement).append(this.filtersContainer);
        Q(this.modelCardElement).append(this.cardsContainer);

        // Load favorites and stats on initialization
        this.loadFavoritesAndStats();
        this.loadModels();

        // NEW: Setup bidirectional sync for autohandle models
        // If this modelcard has autohandle config, listen to prompt changes
        if (this._getAutohandleAssign() && this._getAutohandlePattern()) {
            this._setupPromptSyncListener();
        }
    }

    /**
     * Helper: Setup listener to sync modelcard with prompt changes.
     * When prompt is edited, parse autohandle tags and update loaded/in-use status.
     */
    _setupPromptSyncListener() {
        const assign = this._getAutohandleAssign();
        const pattern = this._getAutohandlePattern();
        
        console.log(`[autohandle] Setting up sync listener for assign="${assign}", pattern="${pattern}"`);
        
        if (!assign) return;

        // Listen for document-level changes (prompt field updates)
        // Use a timer to debounce frequent updates
        let syncTimer = null;
        const debounceSync = () => {
            if (syncTimer) clearTimeout(syncTimer);
            syncTimer = setTimeout(() => {
                console.log(`[autohandle] Debounced sync triggered for "${assign}"`);
                this._syncModelcardFromPrompt();
            }, 500); // Wait 500ms after prompt stops changing
        };

        // Listen to document changes that might affect the prompt
        Q(document).on('change', (e) => {
            // Check if the change involves the prompt field
            if (e.target && e.target.classList && 
                (e.target.classList.contains('promptfield') || 
                 e.target.id?.includes('prompt') ||
                 e.target.getAttribute('data-assign')?.includes('prompt'))) {
                console.log(`[autohandle] Prompt change detected, triggering sync`);
                debounceSync();
            }
        });

        // Also trigger sync when modelcard itself loads/unloads
        // to catch any manual prompt edits
        Q(document).on('model:in-use-changed', () => {
            debounceSync();
        });
    }

    /**
     * Parse autohandle tags from prompt and sync modelcard loaded status.
     * Called when prompt changes to ensure modelcard reflects prompt state.
     */
    _syncModelcardFromPrompt() {
        const assign = this._getAutohandleAssign();
        const pattern = this._getAutohandlePattern();
        
        if (!assign || !pattern) return;

        try {
            // Use frameTab-aware get to read from active frame
            const prompt = ModuleAPI.getOnActiveFrame(assign) || '';
            console.log(`[autohandle] _syncModelcardFromPrompt: assign="${assign}", prompt="${prompt}"`);
            
            if (!prompt) {
                // Empty prompt - clear all loaded models for this type
                const toRemove = Array.from(this.loadedModels);
                toRemove.forEach(modelId => this.removeLoadedModel(modelId));
                return;
            }

            // Extract autohandle tags from prompt
            // Pattern like "<lora:model_name:weight>"
            // Step 1: Escape special regex characters
            let tagPattern = pattern
                .replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
            
            // Step 2: Replace escaped placeholders with regex captures
            tagPattern = tagPattern
                .replace(/\\\{name\\\}/g, '([^:>]+)')     // Capture model name
                .replace(/\\\{weight\\\}/g, '[\\d.]+');   // Match weight
            
            const regex = new RegExp(tagPattern, 'g');
            const tagsInPrompt = new Set();
            let match;
            while ((match = regex.exec(prompt)) !== null) {
                const modelName = match[1]; // First capture group is the model name
                tagsInPrompt.add(modelName);
            }

            // Find matching models in our data
            const modelsToKeepLoaded = new Set();
            this.modelData.forEach(model => {
                if (tagsInPrompt.has(model.name)) {
                    modelsToKeepLoaded.add(model.id);
                }
            });

            // Sync: Remove models not in prompt tags, add models that are
            const currentLoaded = new Set(this.loadedModels);
            
            // Remove models that are no longer in prompt
            currentLoaded.forEach(modelId => {
                if (!modelsToKeepLoaded.has(modelId)) {
                    this.removeLoadedModel(modelId);
                }
            });

            // Add models that are now in prompt
            modelsToKeepLoaded.forEach(modelId => {
                if (!currentLoaded.has(modelId)) {
                    this.addLoadedModel(modelId);
                    // Also mark as in-use since it's in the prompt
                    this.setModelInUse(modelId);
                }
            });

            this.render();
        } catch (e) {
            console.warn(`Failed to sync modelcard from prompt: ${e}`);
        }
    }

    // Map component_type to API category
    _getCategory() {
        const typeMap = {
            'models': 'checkpoint',
            'loras': 'lora',
            'vaes': 'vae',
            'textual_inversion': 'textual_inversion',
            'lycoris': 'lycoris'
        };
        return typeMap[this.componentType] || this.componentType;
    }

    // Load favorites and model stats from backend
    loadFavoritesAndStats() {
        const self = this;
        const category = this._getCategory();
        
        // Load favorites (all categories)
        Q.get('/favorite')
            .then(response => {
                if (response.status && response.favorites) {
                    const categoryFavorites = response.favorites[category] || [];
                    self.favorites = new Set(categoryFavorites);
                    self.render();
                }
            })
            .catch(error => console.error('Failed to load favorites:', error));
        
        // Load stats
        Q.get('/stats')
            .then(response => {
                if (response.status && response.stats) {
                    const categoryStats = response.stats[category] || {};
                    self.modelStats = categoryStats;
                    self.render();
                }
            })
            .catch(error => console.error('Failed to load model stats:', error));
    }

    _getSortOptions() {
        return [
            lang('ui.model_card.sort_a_to_z'),
            lang('ui.model_card.sort_z_to_a'),
            lang('ui.model_card.sort_most_used'),
            lang('ui.model_card.sort_least_used'),
            lang('ui.model_card.sort_loaded'),
            lang('ui.model_card.sort_favorited')
        ];
    }

    _getSortMapping() {
        return {
            [lang('ui.model_card.sort_a_to_z')]: 'a-to-z',
            [lang('ui.model_card.sort_z_to_a')]: 'z-to-a',
            [lang('ui.model_card.sort_most_used')]: 'most-used',
            [lang('ui.model_card.sort_least_used')]: 'least-used',
            [lang('ui.model_card.sort_loaded')]: 'loaded',
            [lang('ui.model_card.sort_favorited')]: 'favorited'
        };
    }

    _sortModels(models) {
        const sorted = [...models];
        const self = this;
        
        switch (this.sortOption) {
            case 'a-to-z':
                sorted.sort((a, b) => {
                    const nameA = (a.name || a.filename || '').toLowerCase();
                    const nameB = (b.name || b.filename || '').toLowerCase();
                    return nameA.localeCompare(nameB);
                });
                break;
            case 'z-to-a':
                sorted.sort((a, b) => {
                    const nameA = (a.name || a.filename || '').toLowerCase();
                    const nameB = (b.name || b.filename || '').toLowerCase();
                    return nameB.localeCompare(nameA);
                });
                break;
            case 'most-used':
                sorted.sort((a, b) => {
                    const countA = self.modelStats[a.id] || 0;
                    const countB = self.modelStats[b.id] || 0;
                    return countB - countA;
                });
                break;
            case 'least-used':
                sorted.sort((a, b) => {
                    const countA = self.modelStats[a.id] || 0;
                    const countB = self.modelStats[b.id] || 0;
                    return countA - countB;
                });
                break;
            case 'loaded':
                sorted.sort((a, b) => {
                    const aLoaded = self.isModelLoaded(a.id) ? 0 : 1;
                    const bLoaded = self.isModelLoaded(b.id) ? 0 : 1;
                    return aLoaded - bLoaded;
                });
                break;
            case 'favorited':
                sorted.sort((a, b) => {
                    const aFav = self.favorites.has(a.id) ? 0 : 1;
                    const bFav = self.favorites.has(b.id) ? 0 : 1;
                    return aFav - bFav;
                });
                break;
        }
        
        return sorted;
    }

    toggleFavorite(model) {
        // Use the model UID as the stable identifier for favorites
        const modelUID = model.id;
        
        // URL encode the model ID for safe transmission
        const encodedUID = encodeURIComponent(modelUID);
        const endpoint = `/favorite/${encodedUID}`;
        
        // POST to toggle favorite (endpoint handles add/remove based on current state)
        const request = {
            url: endpoint,
            method: 'POST',
            contentType: 'application/json'
        };
        
        Q.ajax(request)
            .then(response => {
                if (response.status) {
                    // Update local favorites set based on response
                    if (response.is_favorite) {
                        this.favorites.add(modelUID);
                        if (NOTIFICATION_MANAGER) {
                            NOTIFICATION_MANAGER.show(lang('ui.model_card.favorite_added'));
                        }
                    } else {
                        this.favorites.delete(modelUID);
                        if (NOTIFICATION_MANAGER) {
                            NOTIFICATION_MANAGER.show(lang('ui.model_card.favorite_removed'));
                        }
                    }
                    this.render();
                }
            })
            .catch(error => {
                console.error('Failed to toggle favorite:', error);
                if (NOTIFICATION_MANAGER) {
                    NOTIFICATION_MANAGER.show(lang('ui.model_card.favorite_save_failed'), 'error');
                }
            });
    }


    // Állapot lekérdező metódusok
    getLoadedModelsList() {
        return Array.from(this.loadedModels);
    }

    getCurrentModelInUse() {
        return Array.from(this.modelsInUse);
    }

    get() {
        return Array.from(this.modelsInUse);
    }

    set(value) {
        this.modelsInUse.clear();
        if (Array.isArray(value)) {
            value.forEach(modelId => {
                if (this.isModelLoaded(modelId)) {
                    this.modelsInUse.add(modelId);
                }
            });
        } else if (typeof value === 'string' && this.isModelLoaded(value)) {
            this.modelsInUse.add(value);
        }
        this.render();
    }

    isModelLoaded(modelId) {
        return this.loadedModels.has(modelId);
    }

    isModelInUse(modelId) {
        return this.modelsInUse.has(modelId);
    }

    addLoadedModel(modelId) {
        this.loadedModels.add(modelId);
        this.render();
        
        // Sync with all other instances of the same type
        modelCard.syncLoadedModels(this.componentType, this, this.loadedModels);
    }

    removeLoadedModel(modelId) {
        this.loadedModels.delete(modelId);
        this.modelsInUse.delete(modelId);
        this.render();
        
        // Sync loaded models with all other instances of the same type
        modelCard.syncLoadedModels(this.componentType, this, this.loadedModels);
        
        // Also sync modelsInUse since we removed it from there too
        modelCard.syncModelsInUse(this.componentType, this, this.modelsInUse);
    }

    setModelInUse(modelId) {
        if (!this.isModelLoaded(modelId)) {
            return false;
        }
        
        if (this.modelsInUse.size >= this.modelInUseMaximum) {
            // Evict the oldest selected model
            const oldestId = this.modelsInUse.values().next().value;
            if (oldestId !== undefined) {
                this.modelsInUse.delete(oldestId);
            }
        }
        
        this.modelsInUse.add(modelId);
        this.render();
        
        // Sync with all other instances of the same type
        modelCard.syncModelsInUse(this.componentType, this, this.modelsInUse);
        
        Q(document).trigger('model:in-use-changed', { modelIds: Array.from(this.modelsInUse) });
        
        return true;
    }

    removeModelFromUse(modelId) {
        this.modelsInUse.delete(modelId);
        this.render();
        
        // Sync with all other instances of the same type
        modelCard.syncModelsInUse(this.componentType, this, this.modelsInUse);
        
        Q(document).trigger('model:in-use-changed', { modelIds: Array.from(this.modelsInUse) });
    }

    loadModels() {
        if (typeof Q !== 'undefined' && Q.ajax) {
            const method = this.apiMethod.toUpperCase();
            if (method === 'POST') {
                Q.post(this.apiEndpoint, {})
                    .then(data => this.data(data))
                    .catch(error => {
                        console.error('Failed to load models (POST):', error);
                        this.render();
                    });
            } else {
                Q.get(this.apiEndpoint)
                    .then(data => this.data(data))
                    .catch(error => {
                        console.error('Failed to load models (GET):', error);
                        this.render();
                    });
            }
        } else {
            console.warn('Q.ajax not available, rendering empty model card');
            this.render();
        }
    }

    refreshModels() {
        // Reload models from the API endpoint (same as initial load)
        this.loadModels();
    }

    data(data) {
        this.modelData = Array.isArray(data) ? data : (data.data || data.models || []);
        
        if (data.loaded && Array.isArray(data.loaded)) {
            this.loadedModels = new Set(data.loaded);
        }
        
        if (data.model_in_use) {
            if (Array.isArray(data.model_in_use)) {
                this.modelsInUse = new Set(data.model_in_use);
                this.loadedModels = new Set(data.model_in_use); // Sync loaded with in-use
            } else if (typeof data.model_in_use === 'string') {
                this.modelsInUse = new Set([data.model_in_use]);
                this.loadedModels = new Set([data.model_in_use]); // Sync loaded with in-use
            }
        }
        
        this.updateTypeDropdown();
        this.updateFolderBrowser();
        this.render();
    }

    updateTypeDropdown() {
        const displayAll = lang('ui.model_card.all');
        const options = [displayAll];
        const valueMapping = { [displayAll]: '__all__' };
        const seen = new Set(['__all__']);

        if (this.modelData && this.modelData.length > 0) {
            this.modelData.forEach(model => {
                const rawType = this.getModelArchitecture(model);
                const normalized = this.normalizeArchitecture(rawType);
                if (!normalized || seen.has(normalized)) {
                    return;
                }

                seen.add(normalized);
                const display = rawType && typeof rawType === 'string' ? rawType : normalized;
                options.push(display);
                valueMapping[display] = normalized;
            });
        }

        this.typeDropdown.updateOptions({ options, valueMapping });

        if (!seen.has(this.selectedType)) {
            this.selectedType = '__all__';
        }

        this.typeDropdown.set(this.selectedType);
    }

    getModelArchitecture(model) {
        if (!model) {
            return '';
        }

        let type = model.architecture || model.metadata?.model_architecture;

        if (!type && model.metadata && model.metadata['sd version']) {
            type = model.metadata['sd version'];
        }

        if (!type && Array.isArray(model.compatibility) && model.compatibility.length > 0) {
            type = model.compatibility[0];
        }

        return typeof type === 'string' ? type : '';
    }

    normalizeArchitecture(value) {
        return value ? String(value).trim().toLowerCase() : '';
    }

    updateFolderBrowser() {
        if (!this.modelData || this.modelData.length === 0) {
            return;
        }

        const paths = new Set();
        
        // Add paths from actual models only
        this.modelData.forEach(model => {
            if (model.path) {
                paths.add(model.path);
            }
        });

        // Update the folder browser with paths that actually contain models
        this.folderBrowser.updatePaths(Array.from(paths));
    }

    render() {
        Q(this.cardsContainer).empty();

        if (!this.modelData || this.modelData.length === 0) {
            const noModels = Q('<div>', { class: 'no_models', text: lang('ui.model_card.no_models_available') }).get(0);
            Q(this.cardsContainer).append(noModels);
            return;
        }

        let filteredModels = this.modelData.filter(model => {
            // Use name field if available, otherwise fallback to filename
            const modelName = model.name || model.filename || '';
            const matchesSearch = modelName.toLowerCase().includes(this.searchTerm);
            const architectureRaw = this.getModelArchitecture(model);
            const architectureNormalized = this.normalizeArchitecture(architectureRaw);

            const matchesType = this.selectedType === '__all__' || architectureNormalized === this.selectedType;
            const matchesPath = this.selectedPath === '' || model.path === this.selectedPath;
            
            return matchesSearch && matchesType && matchesPath;
        });

        // Apply sorting
        filteredModels = this._sortModels(filteredModels);

        if (filteredModels.length === 0 && (this.searchTerm || this.selectedPath)) {
            const noResults = Q('<div>', { class: 'no_models', text: lang('ui.model_card.no_models_found') }).get(0);
            Q(this.cardsContainer).append(noResults);
            return;
        }

        filteredModels.forEach(model => {
            const card = Q('<div>', { class: 'card' }).get(0);
            
            const isLoaded = this.isModelLoaded(model.id);
            const isInUse = this.isModelInUse(model.id);
            const isFavorite = this.favorites.has(model.id);
            
            if (isLoaded) {
                Q(card).addClass("card_loaded");
            }
            if (isInUse) {
                Q(card).addClass("card_in_use");
            }
            if (isFavorite) {
                Q(card).addClass("card_favorite");
            }

            const dataContainer = Q('<div>', { class: 'card_data_container' }).get(0);
            const imageDiv = Q('<div>', { class: 'card_image' }).get(0);

            if (model.id) {
                // Build preview URL using model ID (no encoding issues, clean & simple)
                const previewUrl = `/models/preview/${model.id}`;
                Q(imageDiv).css('backgroundImage', `url(${previewUrl})`);
            } else {
                const noPreview = Q('<div>', { class: 'no_preview', text: lang('ui.model_card.no_preview') }).get(0);
                Q(imageDiv).append(noPreview);
            }

            const loadingBarContainer = Q('<div>', { class: 'card_loading_bar_container' }).get(0);
            const loadingBar = Q('<div>', { class: 'card_loading_bar' }).get(0);
            Q(loadingBarContainer).append(loadingBar);

            this.setupLongPressHandler(card, model, loadingBarContainer, loadingBar, isLoaded);

            const purifiedName = (model.name || model.filename || '').replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1$2').replace(/(\w+)\s+v(\d+)\s+(\d+)/, '$1 v$2 ($3)');
            const title = Q('<div>', { class: 'card_title', text: purifiedName }).get(0);

            const tagsDiv = Q('<div>', { class: 'card_tags' }).get(0);
            let architecture = model.architecture || model.metadata?.model_architecture;
            
            // For LoRAs, use sd version directly from metadata
            if (!architecture && model.metadata && model.metadata['sd version']) {
                architecture = model.metadata['sd version'];
            }
            
            const altCore = model.alt_core;
            const variant = model.variant || '';  // get variant from model data
            
            // Combine all tags: architecture, variant, extra tags (furry, anime, etc), altCore, file_format
            const extraTags = model.tags || [];  // Extra tags from discovery: furry, anime, realistic, scifi, fantasy, horror, nsfw
            const tags = [
                architecture,
                variant,  // add variant as a tag if available
                ...extraTags,  // Add extra tags (furry, anime, etc.)
                altCore,
                model.metadata?.file_format
            ].filter(tag => tag);

            // Ensure unique tags while preserving order
            const uniqueTags = [...new Set(tags)];

            uniqueTags.forEach(tag => {
                const tagElement = Q('<span>', { class: 'tag', text: tag }).get(0);
                
                // Set localization key as data-tooltip for all tags
                // tooltip.js will resolve it with lang() when hovering
                // If no localization exists, tooltip.js will show the original tag text
                tagElement.setAttribute('data-tooltip', `ui.tag.${tag}`);
                
                Q(tagsDiv).append(tagElement);
            });

            // Build stats display
            const statsDiv = Q('<div>', { class: 'card_stats' }).get(0);
            const loadCount = this.modelStats[model.id] || 0;
            const statsText = lang('ui.model_card.times_loaded').replace('{count}', loadCount);
            Q(statsDiv).text(statsText);

            const buttonsDiv = Q('<div>', { class: 'card_buttons' }).get(0);
            
            // Create favorite button
            const favoriteButton = Q('<div>', { class: 'favorite_button' + (isFavorite ? ' favorited' : '') }).get(0);
            const favIcon = isFavorite ? window.UI_ICONS.favorite.filled : window.UI_ICONS.favorite.unfilled;
            Q(favoriteButton).html(favIcon);
            Q(favoriteButton).on('click', (e) => {
                e.stopPropagation();
                this.toggleFavorite(model);
            });
            favoriteButton.setAttribute('data-tooltip', isFavorite ? 'ui.model_card.remove_from_favorites' : 'ui.model_card.add_to_favorites');

            const loadButton = Q('<div>', { class: 'load_button', text: isLoaded ? lang('ui.model_card.unload') : lang('ui.model_card.load') }).get(0);
            const infoButton = Q('<div>', { class: 'info_button', text: lang('ui.model_card.info') }).get(0);

            Q(loadButton).on("click", (e) => {
                if (Q(loadingBarContainer).hasClass("active")) {
                    e.stopPropagation();
                    return;
                }
                this.handleLoadUnload(model, loadButton);
            });

            Q(infoButton).on("click", (e) => {
                if (Q(loadingBarContainer).hasClass("active")) {
                    e.stopPropagation();
                    return;
                }
                this.openModelInfoWindow(model);
            });

            Q(buttonsDiv).append(favoriteButton);
            Q(buttonsDiv).append(loadButton);
            Q(buttonsDiv).append(infoButton);

            Q(dataContainer).append(imageDiv);
            Q(dataContainer).append(loadingBarContainer);
            Q(dataContainer).append(tagsDiv);
            Q(dataContainer).append(statsDiv);
            
            const bottomWrapper = Q('<div>', { class: 'card_bottom_wrapper' }).get(0);
            Q(bottomWrapper).append(title);
            Q(bottomWrapper).append(buttonsDiv);
            
            Q(dataContainer).append(bottomWrapper);
            Q(card).append(dataContainer);


            Q(this.cardsContainer).append(card);
        });
    }

    openModelInfoWindow(model) {
        const windowId = `window_${this.identifier}`;
        const purifiedName = model.name.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1$2').replace(/(\w+)\s+v(\d+)\s+(\d+)/, '$1 v$2 ($3)');
        
        const leftContent = this.createLeftContent(model, purifiedName);
        const rightContent = this.createRightContent(model);
        
        const fullContent = `
            <div class="model_info_layout">
                <div class="model_info_left">
                    ${leftContent}
                </div>
                <div class="model_info_right">
                    ${rightContent}
                </div>
            </div>
        `;

        const infoWindow = new FloatingWindow(windowId, {
            title: `${lang('ui.model_card.model_info_title')} - ${purifiedName}`,
            content: fullContent,
            width: 800,
            height: 600,
            minWidth: 600,
            minHeight: 400,
            showButtons: false
        });

        infoWindow.open();
    }

    createLeftContent(model, purifiedName) {
        // Build preview URL using model ID
        let imageUrl = '';
        if (model.id) {
            imageUrl = `/models/preview/${model.id}`;
        }
        
        // Get architecture - simplified version
        let architecture = model.architecture || model.metadata?.model_architecture;
        
        // For LoRAs, use sd version directly from metadata
        if (!architecture && model.metadata && model.metadata['sd version']) {
            architecture = model.metadata['sd version'];
        }
        
        // Use compatibility as fallback for architecture
        if (!architecture && model.compatibility && model.compatibility.length > 0) {
            architecture = model.compatibility[0];
        }
        
        architecture = architecture || lang('ui.model_card.unknown');
        
        const fileSize = model.metadata?.file_size_mb ? `${model.metadata.file_size_mb} MB` : 
                        (model.file_size ? `${(model.file_size / 1024 / 1024).toFixed(2)} MB` : lang('ui.model_card.unknown'));
        const description = model.description || model.metadata?.description || lang('ui.model_card.no_description');
        
        // Check if description contains HTML
        const isHtmlDescription = description && (description.includes('<') || description.includes('&'));

        return `
            <div class="model_info_left_content">
            <h2 class="model_info_title">${purifiedName}</h2>
            ${imageUrl ? `<img class="model_info_image" src="${imageUrl}">` : ''}
            <div class="model_info_section">
                <span class="model_info_section_title">${lang('ui.model_card.architecture')}:</span> ${architecture}
            </div>
            <div class="model_info_section">
                <span class="model_info_section_title">${lang('ui.model_card.file_size')}:</span> ${fileSize}
            </div>
            <div class="model_info_description">
                <span class="model_info_section_title">${lang('ui.model_card.description')}:</span><br>
                ${isHtmlDescription ? 
                    `<div class="model_info_description_html">${description}</div>` : 
                    `<span class="model_info_description_text">${description.replace(/(.{60})/g, '$1<wbr>')}</span>`
                }
            </div>
            </div>
        `;
    }

    createRightContent(model) {
        const metadata = model.metadata || {};
        
        // Get architecture - same logic as in createLeftContent
        let architecture = model.architecture || model.metadata?.model_architecture;
        
        // For LoRAs, use sd version directly from metadata
        if (!architecture && model.metadata && model.metadata['sd version']) {
            architecture = model.metadata['sd version'];
        }
        
        // Use compatibility as fallback for architecture
        if (!architecture && model.compatibility && model.compatibility.length > 0) {
            architecture = model.compatibility[0];
        }
        
        architecture = architecture || lang('ui.model_card.unknown');
        
        let content = `<div class="model_info_right_content"><h3 class="model_info_main_title">${lang('ui.model_card.technical_details')}</h3>`;
        
        const formatValue = (value) => {
            if (value === null || value === undefined) return 'N/A';
            if (typeof value === 'boolean') return value ? 'Yes' : 'No';
            if (typeof value === 'number') return value.toLocaleString();
            if (typeof value === 'string') {
                // Try to parse as JSON if it looks like JSON
                if ((value.startsWith('{') && value.endsWith('}')) || (value.startsWith('[') && value.endsWith(']'))) {
                    try {
                        const parsed = JSON.parse(value);
                        const jsonId = 'json_' + Math.random().toString(36).substr(2, 9);
                        const copyButton = `<div class="json_copy_button_wrapper"><div class="json_copy_button" data-json-id="${jsonId}">${lang('ui.model_card.copy')}</div></div>`;
                        return `<pre class="model_info_json" id="${jsonId}">${JSON.stringify(parsed, null, 2)}</pre>${copyButton}`;
                    } catch (e) {
                        // Not valid JSON, treat as regular string
                    }
                }
                return value.replace(/(.{60})/g, '$1<wbr>');
            }
            if (Array.isArray(value)) {
                return value.map(item => formatValue(item)).join(', ');
            }
            if (typeof value === 'object') {
                const jsonId = 'json_' + Math.random().toString(36).substr(2, 9);
                const copyButton = `<div class="json_copy_button_wrapper"><div class="json_copy_button" data-json-id="${jsonId}">${lang('ui.model_card.copy')}</div></div>`;
                return `<pre class="model_info_json" id="${jsonId}">${JSON.stringify(value, null, 2)}</pre>${copyButton}`;
            }
            return String(value).replace(/(.{60})/g, '$1<wbr>');
        };
        
        const addSection = (title, data) => {
            if (!data || (Array.isArray(data) && data.length === 0) || (typeof data === 'object' && Object.keys(data).length === 0)) return;
            
            content += `<div class="model_info_technical_section">`;
            content += `<h4 class="model_info_technical_title">${title}</h4>`;
            content += `<div class="model_info_technical_content">`;
            
            if (Array.isArray(data)) {
                data.forEach(item => {
                    content += `<div class="model_info_technical_item">${formatValue(item)}</div>`;
                });
            } else if (typeof data === 'object' && data !== null) {
                Object.entries(data).forEach(([key, value]) => {
                    // Try multiple localization strategies for nested keys too
                    let formattedKey = '';
                    
                    // Strategy 1: Try exact key as-is
                    formattedKey = lang(`metadata.${key}`);
                    if (formattedKey !== `metadata.${key}`) {
                        // Found exact match
                    } else {
                        // Strategy 2: Try lowercase with underscores
                        const normalizedKey = key.toLowerCase().replace(/[.\s-]/g, '_');
                        formattedKey = lang(`metadata.${normalizedKey}`);
                        
                        if (formattedKey === `metadata.${normalizedKey}`) {
                            // Strategy 3: Try extracting the last part after dot
                            const lastPart = key.includes('.') ? key.split('.').pop() : key;
                            const normalizedLastPart = lastPart.toLowerCase().replace(/[.\s-]/g, '_');
                            formattedKey = lang(`metadata.${normalizedLastPart}`);
                            
                            if (formattedKey === `metadata.${normalizedLastPart}`) {
                                // Strategy 4: Fall back to formatted key
                                formattedKey = key.replace(/[._-]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                            }
                        }
                    }
                    
                    content += `<div class="model_info_technical_row">`;
                    content += `<span class="model_info_technical_key">${formattedKey}:</span>`;
                    content += `<span class="model_info_technical_value">${formatValue(value)}</span>`;
                    content += `</div>`;
                });
            } else {
                content += `<div class="model_info_technical_simple">${formatValue(data)}</div>`;
            }
            
            content += `</div></div>`;
        };

        const basicInfo = {
            [lang('ui.model_card.model_id')]: model.id,
            [lang('ui.model_card.model_name')]: model.name || model.filename || lang('ui.model_card.unknown'),
            [lang('ui.model_card.file_path')]: model.path,
            [lang('ui.model_card.architecture')]: architecture,
            [lang('ui.model_card.file_size')]: model.file_size ? `${(model.file_size / 1024 / 1024).toFixed(2)} MB` : lang('ui.model_card.unknown')
        };

        addSection(lang('ui.model_card.basic_information'), basicInfo);

        if (Object.keys(metadata).length > 0) {
            content += `<h3 class="model_info_metadata_title">${lang('ui.model_card.metadata')}</h3>`;
            
            Object.entries(metadata).forEach(([key, value]) => {
                // Try multiple localization strategies
                let formattedTitle = '';
                
                // Strategy 1: Try exact key as-is
                formattedTitle = lang(`metadata.${key}`);
                if (formattedTitle !== `metadata.${key}`) {
                    // Found exact match
                } else {
                    // Strategy 2: Try lowercase with underscores
                    const normalizedKey = key.toLowerCase().replace(/[.\s-]/g, '_');
                    formattedTitle = lang(`metadata.${normalizedKey}`);
                    
                    if (formattedTitle === `metadata.${normalizedKey}`) {
                        // Strategy 3: Try extracting the last part after dot
                        const lastPart = key.includes('.') ? key.split('.').pop() : key;
                        const normalizedLastPart = lastPart.toLowerCase().replace(/[.\s-]/g, '_');
                        formattedTitle = lang(`metadata.${normalizedLastPart}`);
                        
                        if (formattedTitle === `metadata.${normalizedLastPart}`) {
                            // Strategy 4: Fall back to formatted key
                            formattedTitle = key.replace(/[._-]/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                        }
                    }
                }
                
                if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
                    addSection(formattedTitle, value);
                } else if (Array.isArray(value) && value.length > 0) {
                    addSection(formattedTitle, value);
                } else if (value !== null && value !== undefined && value !== '') {
                    addSection(formattedTitle, value);
                }
            });
        }

        content += '</div>';
        return content;
    }

    setupLongPressHandler(card, model, loadingBarContainer, loadingBar, isLoaded) {
        let mouseDownTimer = null;
        let progressTimer = null;
        let startTime = 0;
        const LONG_PRESS_DELAY = 100;
        const LONG_PRESS_DURATION = 500;
        let isLongPressActive = false;

        const startLongPress = (e) => {
            if (e.button !== 0) return;
            
            startTime = Date.now();
            isLongPressActive = false;
            
            mouseDownTimer = setTimeout(() => {
                isLongPressActive = true;
                Q(loadingBarContainer).addClass("active");
                Q(loadingBar).css('transition', `width ${LONG_PRESS_DURATION}ms linear`);
                Q(loadingBar).css('width', "0%");
                
                const loadButton = Q(card).find('.load_button').get(0);
                if (loadButton) {
                    Q(loadButton).addClass("loading");
                    Q(loadButton).text(lang('ui.model_card.loading'));
                }
                
                setTimeout(() => {
                    if (isLongPressActive && Q(loadingBarContainer).hasClass("active")) {
                        Q(loadingBar).css('width', "100%");
                    }
                }, 50);

                progressTimer = setTimeout(() => {
                    if (isLongPressActive && Q(loadingBarContainer).hasClass("active")) {
                        this.triggerLoadUnload(model, isLoaded, loadButton);
                        endLongPress();
                    }
                }, LONG_PRESS_DURATION);
            }, LONG_PRESS_DELAY);
        };

        const endLongPress = () => {
            isLongPressActive = false;
            
            if (mouseDownTimer) {
                clearTimeout(mouseDownTimer);
                mouseDownTimer = null;
            }
            if (progressTimer) {
                clearTimeout(progressTimer);
                progressTimer = null;
            }
            
            const loadButton = card.querySelector('.load_button');
            if (loadButton) {
                Q(loadButton).removeClass("loading");
                const currentlyLoaded = this.isModelLoaded(model.id);
                Q(loadButton).text(currentlyLoaded ? lang('ui.model_card.unload') : lang('ui.model_card.load'));
            }
            
            Q(loadingBarContainer).removeClass("active");
            loadingBar.style.transition = "width 0.2s ease";
            loadingBar.style.width = "0%";
        };

    Q(card).on("mousedown", startLongPress);
    Q(card).on("mouseup", endLongPress);
    Q(card).on("mouseleave", endLongPress);
    Q(card).on("touchstart", startLongPress);
    Q(card).on("touchend", endLongPress);
    Q(card).on("touchcancel", endLongPress);
    Q(card).on("contextmenu", (e) => {
            e.preventDefault();
            endLongPress();
        });
        
    Q(card).on("dragstart", (e) => {
            e.preventDefault();
            endLongPress();
        });
    }

    triggerLoadUnload(model, wasLoaded, loadButton = null, retryCount = 0) {
        // retryCount prevents infinite retry loops - after one auto-unload retry, give up
        const isCurrentlyLoaded = this.isModelLoaded(model.id);
        const endpoint = isCurrentlyLoaded ? `${this.apiUnloadEndpoint}/${model.category}/${model.id}` : `${this.apiLoadEndpoint}/${model.category}/${model.id}`;
        const action = isCurrentlyLoaded ? "unload" : "load";
        
        Q.post(endpoint, {}).then(response => {
            if (loadButton) {
                Q(loadButton).removeClass("loading");
            }
            
            if (response.status) {
                if (action === "load") {
                    this.addLoadedModel(model.id);
                    // Automatically set as in use after loading via long press
                    this.setModelInUse(model.id);
                } else {
                    this.removeLoadedModel(model.id);
                    
                    if (this.isModelInUse(model.id)) {
                        this.setModelInUse("");
                    }
                }
                
                if (loadButton) {
                    Q(loadButton).text(action === "load" ? lang('ui.model_card.unload') : lang('ui.model_card.load'));
                }
                
                this.render();
                
                if (typeof refreshAllModelCards === 'function') {
                    refreshAllModelCards();
                }
            } else {
                if (loadButton) {
                    Q(loadButton).text(action);
                }
                
                // Extract detailed error message from response
                let errorMessage = lang(action === "load" ? 'ui.model_card.model_load_failed' : 'ui.model_card.model_unload_failed');
                
                // Check for detailed error information
                if (response.detail) {
                    errorMessage += `: ${response.detail}`;
                } else if (response.error) {
                    errorMessage += `: ${response.error}`;
                } else if (response.message) {
                    errorMessage += `: ${response.message}`;
                }
                
                // Add HTTP status information if available
                if (response._httpStatus && response._httpStatus !== 200) {
                    errorMessage += ` (HTTP ${response._httpStatus})`;
                }
                
                // Check if limit was reached and we haven't already retried
                const isLimitError = (response.detail && response.detail.includes("Limit reached")) ||
                                    (response.message && response.message.includes("Limit reached"));
                
                if (isLimitError && action === "load" && retryCount === 0) {
                    // Auto-unload oldest model of this category and retry
                    const oldestId = this.loadedModels.values().next().value;
                    if (oldestId && oldestId !== model.id) {
                        console.log(`Limit reached for ${model.category}. Auto-unloading oldest model: ${oldestId}`);
                        
                        // Show info notification about auto-unload
                        if (NOTIFICATION_MANAGER) {
                            NOTIFICATION_MANAGER.show(lang('ui.model_card.model_limit_auto_unload', { oldestModel: oldestId, newModel: model.id }), 'info');
                        }
                        
                        // Auto-unload the oldest model
                        const unloadEndpoint = `${this.apiUnloadEndpoint}/${model.category}/${oldestId}`;
                        Q.post(unloadEndpoint, {}).then(unloadResponse => {
                            if (unloadResponse.status) {
                                console.log(`Auto-unloaded ${oldestId}. Retrying load of ${model.id}...`);
                                this.removeLoadedModel(oldestId);
                                
                                // Retry the original load with retryCount=1 to prevent infinite loops
                                this.triggerLoadUnload(model, wasLoaded, loadButton, 1);
                            } else {
                                // Unload failed, show original error
                                if (NOTIFICATION_MANAGER) {
                                    NOTIFICATION_MANAGER.show(errorMessage, 'error');
                                }
                            }
                        }).catch(unloadError => {
                            // Unload failed, show original error
                            if (NOTIFICATION_MANAGER) {
                                NOTIFICATION_MANAGER.show(errorMessage, 'error');
                            }
                        });
                        return; // Don't show error yet
                    }
                }
                
                if (NOTIFICATION_MANAGER) {
                    NOTIFICATION_MANAGER.show(errorMessage, 'error');
                }
            }
        }).catch(error => {
            if (loadButton) {
                Q(loadButton).removeClass("loading");
                Q(loadButton).text(action);
            }
            
            console.error(`Failed to ${action} model:`, error);
            if (NOTIFICATION_MANAGER) {
                NOTIFICATION_MANAGER.show(lang(action === "load" ? 'ui.model_card.model_load_failed' : 'ui.model_card.model_unload_failed') + `: ${error.message}`, 'error');
            }
        });
    }

    /**
     * Helper: Extract weight from autohandle_pattern (e.g., "<lora:model:0.8>" -> 0.8)
     * Defaults to 1.0 if no explicit weight found.
     */
    _extractWeightFromPattern(pattern) {
        if (!pattern || typeof pattern !== 'string') return 1.0;
        const match = pattern.match(/{weight}/);
        if (match) {
            // Pattern uses {weight} placeholder; try to parse from schema or default to 1.0
            return 1.0;
        }
        // Try to extract weight from pattern like "<lora:name:0.8>"
        const weightMatch = pattern.match(/:[\d.]+>?$/);
        if (weightMatch) {
            const weight = parseFloat(weightMatch[0].replace(/:/g, '').replace(/>/g, ''));
            return isNaN(weight) ? 1.0 : weight;
        }
        return 1.0;
    }

    /**
     * Helper: Generate tag to insert into prompt based on autohandle_pattern.
     * e.g., pattern "<lora:{name}:{weight}>" with model "TestLora" -> "<lora:TestLora:1.0>"
     */
    _generateAutohandleTag(modelName, pattern) {
        if (!pattern || typeof pattern !== 'string') {
            return modelName; // Fallback: just model name
        }
        return pattern.replace(/{name}/g, modelName).replace(/{weight}/g, '1.0');
    }

    /**
     * Helper: Get the prompt assign ID from config (e.g., "positive-prompt" for LoRA)
     */
    _getAutohandleAssign() {
        return this.config.autohandle || null;
    }

    /**
     * Helper: Get the autohandle pattern from config (e.g., "<lora:{name}:{weight}>")
     */
    _getAutohandlePattern() {
        return this.config.autohandle_pattern || null;
    }

    /**
     * Helper: Inject tag into prompt field via ModuleAPI
     */
    _injectPromptTag(assign, tag) {
        if (!assign || !tag) {
            console.warn(`[autohandle] Cannot inject tag: assign="${assign}", tag="${tag}"`);
            return;
        }
        try {
            console.log(`[autohandle] 🔍 Attempting to inject tag into assign="${assign}"`);
            console.log(`[autohandle] 🏗️ COMPONENTS_BUILDER available:`, !!window.COMPONENTS_BUILDER);
            
            // Use frameTab-aware get/set to target active frame specifically
            const currentValue = ModuleAPI.getOnActiveFrame(assign) || '';
            console.log(`[autohandle] 📖 Current "${assign}" value on active frame:`, JSON.stringify(currentValue));
            
            const trimmed = currentValue.trim();
            // Append tag to prompt (with space separator if needed)
            const newValue = trimmed ? `${trimmed} ${tag}` : tag;
            console.log(`[autohandle] ✏️ New value to set:`, JSON.stringify(newValue));
            
            // Use frameTab-aware set
            const setResult = ModuleAPI.setOnActiveFrame(assign, newValue);
            console.log(`[autohandle] 📤 ModuleAPI.setOnActiveFrame() returned:`, setResult);
            
            // Verify the value was actually set
            const verifyValue = ModuleAPI.getOnActiveFrame(assign);
            console.log(`[autohandle] ✅ Verification - "${assign}" value after set:`, JSON.stringify(verifyValue));
            
            if (verifyValue === newValue) {
                console.log(`✓ ✓ ✓ SUCCESSFULLY injected tag into "${assign}": "${tag}"`);
            } else {
                console.warn(`❌ WARNING: Value was not set correctly! Expected: "${newValue}", Got: "${verifyValue}"`);
            }
        } catch (e) {
            console.warn(`[autohandle] ❌ Failed to inject tag into "${assign}":`, e);
        }
    }

    /**
     * Helper: Remove tag from prompt field via ModuleAPI
     */
    _removePromptTag(assign, pattern) {
        if (!assign || !pattern) return;
        try {
            console.log(`[autohandle] 🗑️ Attempting to remove tag from assign="${assign}", pattern="${pattern}"`);
            
            // Use frameTab-aware get
            const currentValue = ModuleAPI.getOnActiveFrame(assign) || '';
            console.log(`[autohandle] 📖 Current "${assign}" value:`, JSON.stringify(currentValue));
            
            // Create a regex that matches the pattern-based tag
            // e.g., pattern "<lora:{name}:{weight}>" becomes regex for "<lora:anyname:anyweight>"
            let regexPattern = pattern
                .replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); // Escape special regex chars
            
            // Now replace placeholders after escaping
            regexPattern = regexPattern
                .replace(/\\\{name\\\}/g, '[^:>]+')      // Match any characters except : and >
                .replace(/\\\{weight\\\}/g, '[\\d.]+');  // Match digits and dots
            
            const regex = new RegExp(`\\s*${regexPattern}\\s*`, 'g');
            const newValue = currentValue.replace(regex, ' ').trim();
            console.log(`[autohandle] ✏️ New value after removal:`, JSON.stringify(newValue));
            
            // Use frameTab-aware set
            ModuleAPI.setOnActiveFrame(assign, newValue);
            console.log(`✓ Removed tag from "${assign}"`);
        } catch (e) {
            console.warn(`[autohandle] ❌ Failed to remove tag from "${assign}":`, e);
        }
    }

    handleLoadUnload(model, buttonElement, retryCount = 0) {
        // retryCount prevents infinite retry loops - after one auto-unload retry, give up
        const isLoaded = this.isModelLoaded(model.id);
        const endpoint = isLoaded ? `${this.apiUnloadEndpoint}/${model.category}/${model.id}` : `${this.apiLoadEndpoint}/${model.category}/${model.id}`;
        const action = isLoaded ? "unload" : "load";
        
    Q(buttonElement).text(lang('ui.model_card.loading'));
    Q(buttonElement).addClass("loading");
        
        Q.post(endpoint, {}).then(response => {
            Q(buttonElement).removeClass("loading");
            
            if (response.status) {
                if (action === "load") {
                    this.addLoadedModel(model.id);
                    // Automatically set model as "In Use" when loaded
                    this.setModelInUse(model.id);
                    Q(buttonElement).text(lang('ui.model_card.unload'));

                    // Handle autohandle: inject tag into prompt
                    const assign = this._getAutohandleAssign();
                    const pattern = this._getAutohandlePattern();
                    if (assign && pattern) {
                        const tag = this._generateAutohandleTag(model.name, pattern);
                        console.log(`[autohandle] Loading ${model.name}: assign="${assign}", pattern="${pattern}", tag="${tag}"`);
                        this._injectPromptTag(assign, tag);
                    }
                } else {
                    this.removeLoadedModel(model.id);
                    Q(buttonElement).text(lang('ui.model_card.load'));
                    
                    // Remove from "In Use" if it was selected
                    if (this.isModelInUse(model.id)) {
                        this.removeModelFromUse(model.id);
                    }

                    // Handle autohandle: remove tag from prompt
                    const assign = this._getAutohandleAssign();
                    const pattern = this._getAutohandlePattern();
                    if (assign && pattern) {
                        console.log(`[autohandle] Unloading model: assign="${assign}", pattern="${pattern}"`);
                        this._removePromptTag(assign, pattern);
                    }
                }
                
                this.render();
                
                if (typeof refreshAllModelCards === 'function') {
                    refreshAllModelCards();
                }
            } else {
                Q(buttonElement).text(action);
                
                // Extract detailed error message from response
                let errorMessage = lang(action === "load" ? 'ui.model_card.model_load_failed' : 'ui.model_card.model_unload_failed');
                
                // Check for detailed error information
                if (response.detail) {
                    errorMessage += `: ${response.detail}`;
                } else if (response.error) {
                    errorMessage += `: ${response.error}`;
                } else if (response.message) {
                    errorMessage += `: ${response.message}`;
                }
                
                // Add HTTP status information if available
                if (response._httpStatus && response._httpStatus !== 200) {
                    errorMessage += ` (HTTP ${response._httpStatus})`;
                }
                
                // Check if limit was reached and we haven't already retried
                const isLimitError = (response.detail && response.detail.includes("Limit reached")) ||
                                    (response.message && response.message.includes("Limit reached"));
                
                if (isLimitError && action === "load" && retryCount === 0) {
                    // Auto-unload oldest model of this category and retry
                    const oldestId = this.loadedModels.values().next().value;
                    if (oldestId && oldestId !== model.id) {
                        console.log(`Limit reached for ${model.category}. Auto-unloading oldest model: ${oldestId}`);
                        
                        // Show info notification about auto-unload
                        if (NOTIFICATION_MANAGER) {
                            NOTIFICATION_MANAGER.show(lang('ui.model_card.model_limit_auto_unload', { oldestModel: oldestId, newModel: model.id }), 'info');
                        }
                        
                        // Auto-unload the oldest model
                        const unloadEndpoint = `${this.apiUnloadEndpoint}/${model.category}/${oldestId}`;
                        Q.post(unloadEndpoint, {}).then(unloadResponse => {
                            if (unloadResponse.status) {
                                console.log(`Auto-unloaded ${oldestId}. Retrying load of ${model.id}...`);
                                this.removeLoadedModel(oldestId);
                                
                                // Retry the original load with retryCount=1 to prevent infinite loops
                                this.handleLoadUnload(model, buttonElement, 1);
                            } else {
                                // Unload failed, show original error
                                if (NOTIFICATION_MANAGER) {
                                    NOTIFICATION_MANAGER.show(errorMessage, 'error');
                                }
                            }
                        }).catch(unloadError => {
                            // Unload failed, show original error
                            if (NOTIFICATION_MANAGER) {
                                NOTIFICATION_MANAGER.show(errorMessage, 'error');
                            }
                        });
                        return; // Don't show error yet
                    }
                }
                
                if (NOTIFICATION_MANAGER) {
                    NOTIFICATION_MANAGER.show(errorMessage, 'error');
                }
            }
        }).catch(error => {
            Q(buttonElement).removeClass("loading");
            Q(buttonElement).text(action);
            
            console.error(`Failed to ${action} model:`, error);
            if (NOTIFICATION_MANAGER) {
                NOTIFICATION_MANAGER.show(lang(action === "load" ? 'ui.model_card.model_load_failed' : 'ui.model_card.model_unload_failed') + `: ${error.message}`, 'error');
            }
        });
    }

    getElement() {
        return this.modelCardElement;
    }

    // Static methods for global instance management
    static _instances = {}; // { componentType: [instance1, instance2, ...] }

    static registerInstance(componentType, instance) {
        if (!modelCard._instances[componentType]) {
            modelCard._instances[componentType] = [];
        }
        modelCard._instances[componentType].push(instance);
        console.log(`ModelCard: Registered instance for type "${componentType}", total: ${modelCard._instances[componentType].length}`);
    }

    static unregisterInstance(componentType, instance) {
        if (modelCard._instances[componentType]) {
            const index = modelCard._instances[componentType].indexOf(instance);
            if (index !== -1) {
                modelCard._instances[componentType].splice(index, 1);
                console.log(`ModelCard: Unregistered instance for type "${componentType}", remaining: ${modelCard._instances[componentType].length}`);
            }
        }
    }

    static syncModelsInUse(componentType, sourceInstance, modelsInUseSet) {
        // Sync all instances of the same componentType
        const instances = modelCard._instances[componentType] || [];
        console.log(`ModelCard: Syncing ${instances.length} instances of type "${componentType}"`);
        
        instances.forEach(instance => {
            if (instance !== sourceInstance) {
                // Update modelsInUse without triggering another sync
                instance.modelsInUse = new Set(modelsInUseSet);
                instance.render();
            }
        });
    }

    static syncLoadedModels(componentType, sourceInstance, loadedModelsSet) {
        // Sync loaded models across all instances of the same componentType
        const instances = modelCard._instances[componentType] || [];
        console.log(`ModelCard: Syncing loaded models for ${instances.length} instances of type "${componentType}"`);
        
        instances.forEach(instance => {
            if (instance !== sourceInstance) {
                // Update loadedModels without triggering another sync
                instance.loadedModels = new Set(loadedModelsSet);
                instance.render();
            }
        });
    }

    destroy() {
        // Cleanup when component is destroyed
        modelCard.unregisterInstance(this.componentType, this);
    }
}

// Global function for copying JSON to clipboard
Q(document).on('click', (e) => {
    const btn = e.target.closest('.json_copy_button');
    if (!btn) return;
    const elementId = btn.getAttribute('data-json-id');
    if (!elementId) return;
    try {
        const element = Q('#' + elementId).get(0);
        if (!element) return;
        const text = element.textContent;
        navigator.clipboard.writeText(text).then(() => {
            if (typeof NOTIFICATION_MANAGER !== 'undefined') {
                NOTIFICATION_MANAGER.show(lang('ui.model_card.copied_to_clipboard'));
            }
        }).catch(err => {
            console.error('Failed to copy: ', err);
            if (typeof NOTIFICATION_MANAGER !== 'undefined') {
                NOTIFICATION_MANAGER.show(lang('ui.model_card.copy_failed'));
            }
        });
    } catch (error) {
        console.error('Copy to clipboard failed:', error);
        if (typeof NOTIFICATION_MANAGER !== 'undefined') {
            NOTIFICATION_MANAGER.show(lang('ui.model_card.copy_failed'));
        }
    }
});