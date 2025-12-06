/**
 * HootSight - Preset Card Widget
 * Displays a single training preset with apply button and size variant selector
 */

const PresetCard = {
    /**
     * Create a preset card element
     * @param {Object} preset - Preset data
     * @param {Function} onApply - Callback when apply is clicked (receives preset and selected size key)
     * @param {string} currentDatasetType - Current project's dataset type
     * @returns {HTMLElement}
     */
    create: function(preset, onApply, currentDatasetType) {
        // Check compatibility
        const isCompatible = this._checkCompatibility(preset, currentDatasetType);
        const cardClass = isCompatible ? 'preset-card' : 'preset-card preset-card-incompatible';
        
        const card = Q('<div>', { class: cardClass }).get(0);

        // Store selected size variant
        let selectedSize = 'medium'; // Default

        const header = Q('<div>', { class: 'preset-card-header' }).get(0);
        const name = Q('<h3>', { class: 'preset-card-name', text: lang(preset.name) }).get(0);
        name.setAttribute('data-lang-key', preset.name);
        const task = Q('<span>', { class: 'preset-card-task', text: preset.task }).get(0);
        
        Q(header).append(name, task);

        const description = Q('<p>', { 
            class: 'preset-card-description', 
            text: lang(preset.description) 
        }).get(0);
        description.setAttribute('data-lang-key', preset.description);

        // Incompatibility warning
        let warning = null;
        if (!isCompatible && currentDatasetType && currentDatasetType !== 'unknown') {
            warning = Q('<div>', { class: 'preset-card-warning' }).get(0);
            const warningIcon = Q('<span>', { class: 'preset-card-warning-icon', text: '!' }).get(0);
            const warningText = Q('<span>', { 
                class: 'preset-card-warning-text', 
                text: lang('training_page.presets.incompatible_dataset') 
            }).get(0);
            warningText.setAttribute('data-lang-key', 'training_page.presets.incompatible_dataset');
            Q(warning).append(warningIcon, warningText);
        }

        // Size variant selector
        let sizeSelector = null;
        let sizeDescription = null;
        if (preset.size_variants && preset.size_variants.length > 0) {
            sizeSelector = Q('<div>', { class: 'preset-card-size-selector' }).get(0);
            
            const sizeLabel = Q('<span>', { 
                class: 'preset-card-size-label', 
                text: lang('training_page.presets.dataset_size') 
            }).get(0);
            sizeLabel.setAttribute('data-lang-key', 'training_page.presets.dataset_size');
            
            const sizeButtons = Q('<div>', { class: 'preset-card-size-buttons' }).get(0);
            
            // Size description element (updated on selection)
            sizeDescription = Q('<div>', { class: 'preset-card-size-description' }).get(0);
            
            preset.size_variants.forEach(variant => {
                const rangeText = this._formatRange(variant.range);
                const translatedDesc = lang(variant.description);
                const btn = Q('<button>', { 
                    class: 'preset-card-size-btn' + (variant.key === 'medium' ? ' active' : ''),
                    text: rangeText,
                    'data-size': variant.key,
                    title: translatedDesc
                }).get(0);
                btn.setAttribute('data-lang-title', 'true');
                btn.setAttribute('data-lang-key', variant.description);
                
                Q(btn).on('click', (e) => {
                    // Update active state
                    Q(sizeButtons).find('.preset-card-size-btn').removeClass('active');
                    Q(btn).addClass('active');
                    selectedSize = variant.key;
                    
                    // Update description
                    Q(sizeDescription).text(lang(variant.description));
                    sizeDescription.setAttribute('data-lang-key', variant.description);
                });
                
                Q(sizeButtons).append(btn);
                
                // Set initial description for medium
                if (variant.key === 'medium') {
                    Q(sizeDescription).text(translatedDesc);
                    sizeDescription.setAttribute('data-lang-key', variant.description);
                }
            });
            
            Q(sizeSelector).append(sizeLabel, sizeButtons);
        }

        let models = null;
        if (preset.recommended_models && preset.recommended_models.length > 0) {
            models = Q('<div>', { class: 'preset-card-models' }).get(0);
            const modelsLabel = Q('<span>', { 
                class: 'preset-card-models-label', 
                text: lang('training_page.presets.recommended') 
            }).get(0);
            modelsLabel.setAttribute('data-lang-key', 'training_page.presets.recommended');
            Q(models).append(modelsLabel);

            preset.recommended_models.forEach(model => {
                const modelTag = Q('<span>', { 
                    class: 'preset-card-model-tag', 
                    text: model 
                }).get(0);
                Q(models).append(modelTag);
            });
        }

        const applyButton = Q('<button>', { 
            class: 'preset-card-apply-button btn btn-primary',
            text: lang('training_page.presets.apply_button')
        }).get(0);
        applyButton.setAttribute('data-lang-key', 'training_page.presets.apply_button');

        Q(applyButton).on('click', () => {
            if (onApply) {
                onApply(preset, selectedSize);
            }
        });

        Q(card).append(header, description);
        if (warning) {
            Q(card).append(warning);
        }
        if (sizeSelector) {
            Q(card).append(sizeSelector);
        }
        if (sizeDescription) {
            Q(card).append(sizeDescription);
        }
        if (models) {
            Q(card).append(models);
        }
        Q(card).append(applyButton);

        return card;
    },

    /**
     * Format range for display
     * @param {Array} range - [min, max] or [min, null] for unbounded
     * @returns {string}
     */
    _formatRange: function(range) {
        if (!range || range.length < 2) return '?';
        const min = range[0] || 0;
        const max = range[1];
        
        // Format number with k/M suffix for readability
        const formatNum = (n) => {
            if (n >= 1000000) return (n / 1000000) + 'M';
            if (n >= 1000) return (n / 1000) + 'k';
            return n.toString();
        };
        
        if (max === null) {
            return formatNum(min) + '+';
        }
        return formatNum(min) + '-' + formatNum(max);
    },

    /**
     * Check if preset is compatible with current dataset type
     * @param {Object} preset - Preset data
     * @param {string} currentDatasetType - Current project's dataset type
     * @returns {boolean}
     */
    _checkCompatibility: function(preset, currentDatasetType) {
        // If no dataset type set or unknown, all presets are "compatible"
        if (!currentDatasetType || currentDatasetType === 'unknown') {
            return true;
        }
        
        // If preset has no dataset_types restriction, it's compatible with everything
        if (!preset.dataset_types || preset.dataset_types.length === 0) {
            return true;
        }
        
        // Check if current dataset type is in the preset's allowed types
        return preset.dataset_types.includes(currentDatasetType);
    }
};
