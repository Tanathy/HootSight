/**
 * HootSight - Augmentation Builder Widget
 * Interactive augmentation pipeline builder with toggle switches and parameter controls
 * Uses existing widget classes: NumberInput, Switch, Dropdown, Text
 * 
 * Features:
 *   - Lists all available augmentations from config
 *   - Toggle switches to enable/disable each augmentation
 *   - Shows parameter controls when augmentation is enabled
 *   - Builds augmentation pipeline array for training config
 */

class AugmentationBuilder {
    constructor(id, options = {}) {
        this.id = id;
        this.options = {
            phase: options.phase || 'train',
            label: options.label || '',
            description: options.description || '',
            augmentationDefaults: options.augmentationDefaults || {},
            currentPipeline: options.currentPipeline || []
        };
        
        this._changeCallbacks = [];
        this._augmentationStates = {}; // { augName: { enabled: bool, params: {} } }
        this._paramWidgets = {}; // { augName: { paramKey: widget } }
        
        this._initializeStates();
        this._build();
    }
    
    /**
     * Initialize augmentation states from current pipeline and defaults
     */
    _initializeStates() {
        const defaults = this.options.augmentationDefaults;
        const pipeline = this.options.currentPipeline || [];
        
        // Create a map of currently enabled augmentations from pipeline
        const enabledMap = {};
        pipeline.forEach(item => {
            if (item && item.type) {
                const type = item.type.toLowerCase();
                if (type !== 'to_tensor' && type !== 'normalize') {
                    enabledMap[type] = item.params || {};
                }
            }
        });
        
        // Initialize states for all known augmentations from defaults
        for (const [augName, augDefaults] of Object.entries(defaults)) {
            if (augName === 'compose' || augName === 'to_tensor' || augName === 'normalize') {
                continue;
            }
            
            const isEnabled = augName in enabledMap;
            const params = isEnabled ? { ...augDefaults, ...enabledMap[augName] } : { ...augDefaults };
            
            this._augmentationStates[augName] = {
                enabled: isEnabled,
                params: params
            };
        }
    }
    
    _build() {
        this.element = Q('<div>', { class: 'widget widget-augmentation-builder' }).get(0);
        this.element.id = this.id;
        
        if (this.options.label) {
            const labelEl = Q('<div>', { class: 'augmentation-builder-label', text: this.options.label }).get(0);
            Q(this.element).append(labelEl);
        }
        
        if (this.options.description) {
            const descEl = Q('<div>', { class: 'widget-description', text: this.options.description }).get(0);
            Q(this.element).append(descEl);
        }
        
        this.listContainer = Q('<div>', { class: 'augmentation-list' }).get(0);
        Q(this.element).append(this.listContainer);
        
        this._buildAugmentationItems();
    }
    
    /**
     * Build the list of augmentation items with toggles and params
     */
    _buildAugmentationItems() {
        const sortedAugs = Object.keys(this._augmentationStates).sort();
        
        sortedAugs.forEach(augName => {
            const item = this._buildAugmentationItem(augName);
            Q(this.listContainer).append(item);
        });
    }
    
    /**
     * Build a single augmentation item
     */
    _buildAugmentationItem(augName) {
        const state = this._augmentationStates[augName];
        const defaults = this.options.augmentationDefaults[augName] || {};
        
        const item = Q('<div>', { class: 'augmentation-item' });
        item.get().dataset.augmentation = augName;
        if (state.enabled) {
            item.addClass('enabled');
        }
        
        // Header with toggle switch using Switch widget pattern
        const header = Q('<div>', { class: 'augmentation-item-header' }).get(0);
        
        // Toggle using existing switch structure
        const switchContainer = Q('<div>', { class: 'switch-container augmentation-switch' }).get(0);
        const checkbox = Q('<input>', { type: 'checkbox', class: 'switch-input' }).get(0);
        checkbox.checked = state.enabled;
        const track = Q('<div>', { class: 'switch-track' });
        const thumb = Q('<div>', { class: 'switch-thumb' }).get(0);
        Q(track.get()).append(thumb);
        Q(switchContainer).append(checkbox);
        Q(switchContainer).append(track.get());
        
        // Update visual state
        if (state.enabled) {
            track.addClass('active');
        }
        
        Q(header).append(switchContainer);
        
        // Title
        const title = Q('<span>', { 
            class: 'augmentation-item-title',
            text: this._formatAugmentationName(augName)
        }).get(0);
        Q(header).append(title);
        
        Q(item.get()).append(header);
        
        // Parameters container
        const paramsContainer = Q('<div>', { class: 'augmentation-item-params' }).get(0);
        if (!state.enabled) {
            paramsContainer.style.display = 'none';
        }
        
        // Build parameter widgets
        this._paramWidgets[augName] = {};
        this._buildParamWidgets(augName, defaults, state.params, paramsContainer);
        
        Q(item.get()).append(paramsContainer);
        
        // Toggle event
        Q(checkbox).on('change', () => {
            state.enabled = checkbox.checked;
            item.toggleClass('enabled', state.enabled);
            track.toggleClass('active', state.enabled);
            paramsContainer.style.display = state.enabled ? '' : 'none';
            this._notifyChange();
        });
        
        // Click on track toggles
        track.on('click', (e) => {
            e.preventDefault();
            checkbox.checked = !checkbox.checked;
            checkbox.dispatchEvent(new Event('change'));
        });
        
        // Click on header title also toggles
        Q(title).on('click', () => {
            checkbox.checked = !checkbox.checked;
            checkbox.dispatchEvent(new Event('change'));
        });
        
        return item.get();
    }
    
    /**
     * Build parameter widgets using existing widget classes
     */
    _buildParamWidgets(augName, defaults, currentParams, container) {
        const paramKeys = Object.keys(defaults);
        
        if (paramKeys.length === 0) {
            const noParams = Q('<div>', { class: 'augmentation-no-params', text: lang('training_page.augmentation.no_params') }).get(0);
            noParams.setAttribute('data-lang-key', 'training_page.augmentation.no_params');
            Q(container).append(noParams);
            return;
        }
        
        paramKeys.forEach(paramKey => {
            const defaultValue = defaults[paramKey];
            const currentValue = currentParams[paramKey] !== undefined ? currentParams[paramKey] : defaultValue;
            
            const widget = this._createParamWidget(augName, paramKey, defaultValue, currentValue);
            if (widget) {
                this._paramWidgets[augName][paramKey] = widget;
                Q(container).append(widget.getElement());
            }
        });
    }
    
    /**
     * Create appropriate widget based on value type
     */
    _createParamWidget(augName, paramKey, defaultValue, currentValue) {
        const label = this._formatParamName(paramKey);
        const widgetId = `${this.id}-${augName}-${paramKey}`;
        
        // Handle null values - show as optional number
        if (defaultValue === null) {
            return this._createNullableNumberWidget(widgetId, augName, paramKey, label, currentValue);
        }
        
        // Handle arrays - create multiple number inputs
        if (Array.isArray(defaultValue)) {
            return this._createArrayWidget(widgetId, augName, paramKey, label, defaultValue, currentValue);
        }
        
        // Handle booleans - use Switch
        if (typeof defaultValue === 'boolean') {
            const widget = new Switch(widgetId, {
                label: label,
                default: currentValue
            });
            
            widget.onChange(newValue => {
                this._augmentationStates[augName].params[paramKey] = newValue;
                this._notifyChange();
            });
            
            return widget;
        }
        
        // Handle numbers - use NumberInput
        if (typeof defaultValue === 'number') {
            const isInteger = Number.isInteger(defaultValue);
            const step = this._getStepForParam(paramKey, defaultValue);
            
            const widget = new NumberInput(widgetId, {
                label: label,
                default: currentValue,
                step: step,
                integer: isInteger,
                min: this._getMinForParam(paramKey),
                max: this._getMaxForParam(paramKey)
            });
            
            widget.onChange(newValue => {
                this._augmentationStates[augName].params[paramKey] = newValue;
                this._notifyChange();
            });
            
            return widget;
        }
        
        // Handle strings - check if enum or free text
        if (typeof defaultValue === 'string') {
            const enumOptions = this._getEnumOptions(paramKey);
            
            if (enumOptions) {
                const optionLabels = {};
                enumOptions.forEach(opt => {
                    optionLabels[opt] = opt.charAt(0).toUpperCase() + opt.slice(1);
                });
                
                const widget = new Dropdown(widgetId, {
                    label: label,
                    options: enumOptions,
                    optionLabels: optionLabels,
                    default: currentValue
                });
                
                widget.onChange(newValue => {
                    this._augmentationStates[augName].params[paramKey] = newValue;
                    this._notifyChange();
                });
                
                return widget;
            } else {
                // Free text - use TextInput widget
                const widget = new TextInput(widgetId, {
                    label: label,
                    default: currentValue
                });
                
                widget.onChange(newValue => {
                    this._augmentationStates[augName].params[paramKey] = newValue;
                    this._notifyChange();
                });
                
                return widget;
            }
        }
        
        return null;
    }
    
    /**
     * Create a nullable number widget (shows empty when null)
     */
    _createNullableNumberWidget(widgetId, augName, paramKey, label, currentValue) {
        // Use TextInput widget with null handling
        const displayValue = currentValue !== null ? String(currentValue) : '';
        
        const widget = new TextInput(widgetId, {
            label: label,
            default: displayValue,
            placeholder: 'null'
        });
        
        widget.onChange(newValue => {
            const trimmed = newValue.trim();
            if (trimmed === '' || trimmed.toLowerCase() === 'null') {
                this._augmentationStates[augName].params[paramKey] = null;
            } else {
                const num = parseFloat(trimmed);
                this._augmentationStates[augName].params[paramKey] = isNaN(num) ? trimmed : num;
            }
            this._notifyChange();
        });
        
        return widget;
    }
    
    /**
     * Create array widget (multiple number inputs in a row)
     */
    _createArrayWidget(widgetId, augName, paramKey, label, defaultValue, currentValue) {
        const arr = Array.isArray(currentValue) ? currentValue : defaultValue;
        
        // Create a wrapper with custom getElement
        const wrapper = {
            element: Q('<div>', { class: 'widget widget-array-inline' }).get(0),
            widgets: [],
            getElement: function() { return this.element; },
            getValue: function() {
                return this.widgets.map(w => w.getValue());
            }
        };
        
        // Label
        const labelEl = Q('<label>', { class: 'widget-label', text: label }).get(0);
        Q(wrapper.element).append(labelEl);
        
        // Inputs container
        const inputsRow = Q('<div>', { class: 'array-inputs-row' }).get(0);
        
        arr.forEach((val, idx) => {
            const itemWidget = new NumberInput(`${widgetId}-${idx}`, {
                label: '',
                default: val,
                step: this._getStepForValue(val)
            });
            
            itemWidget.onChange(newValue => {
                if (!Array.isArray(this._augmentationStates[augName].params[paramKey])) {
                    this._augmentationStates[augName].params[paramKey] = [...arr];
                }
                this._augmentationStates[augName].params[paramKey][idx] = newValue;
                this._notifyChange();
            });
            
            wrapper.widgets.push(itemWidget);
            
            // Get just the input part, not the whole widget
            const el = itemWidget.getElement();
            Q(el).addClass('array-item');
            Q(inputsRow).append(el);
        });
        
        Q(wrapper.element).append(inputsRow);
        
        return wrapper;
    }
    
    _getEnumOptions(paramKey) {
        const enums = {
            'interpolation': ['nearest', 'bilinear', 'bicubic'],
            'padding_mode': ['constant', 'edge', 'reflect', 'symmetric']
        };
        return enums[paramKey] || null;
    }
    
    _getMinForParam(paramKey) {
        const mins = {
            'p': 0,
            'brightness': 0,
            'contrast': 0,
            'saturation': 0,
            'hue': 0,
            'size': 1,
            'bits': 1,
            'threshold': 0,
            'sharpness_factor': 0,
            'distortion_scale': 0
        };
        return mins[paramKey] ?? null;
    }
    
    _getMaxForParam(paramKey) {
        const maxs = {
            'p': 1,
            'hue': 0.5,
            'bits': 8,
            'threshold': 255,
            'distortion_scale': 1
        };
        return maxs[paramKey] ?? null;
    }
    
    _getStepForParam(paramKey, defaultValue) {
        if (paramKey === 'p' || paramKey.includes('probability')) {
            return 0.05;
        }
        if (Math.abs(defaultValue) < 1 && defaultValue !== 0) {
            return 0.01;
        }
        if (Number.isInteger(defaultValue)) {
            return 1;
        }
        return 0.1;
    }
    
    _getStepForValue(value) {
        if (Math.abs(value) < 0.1) return 0.01;
        if (Math.abs(value) < 1) return 0.05;
        if (Number.isInteger(value)) return 1;
        return 0.1;
    }
    
    _formatAugmentationName(name) {
        return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }
    
    _formatParamName(name) {
        return name.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
    }
    
    /**
     * Build the pipeline array from current states
     */
    _buildPipeline() {
        const pipeline = [];
        
        for (const [augName, state] of Object.entries(this._augmentationStates)) {
            if (state.enabled) {
                const transform = { type: augName };
                
                const params = { ...state.params };
                // Keep null values - they're valid for optional params
                Object.keys(params).forEach(k => {
                    if (params[k] === undefined) {
                        delete params[k];
                    }
                });
                
                if (Object.keys(params).length > 0) {
                    transform.params = params;
                }
                
                pipeline.push(transform);
            }
        }
        
        // Always add to_tensor at the end
        pipeline.push({ type: 'to_tensor' });
        
        // Always add normalize with ImageNet defaults
        pipeline.push({ 
            type: 'normalize',
            params: {
                mean: [0.485, 0.456, 0.406],
                std: [0.229, 0.224, 0.225]
            }
        });
        
        return pipeline;
    }
    
    _notifyChange() {
        const pipeline = this._buildPipeline();
        this._changeCallbacks.forEach(cb => cb(pipeline));
    }
    
    // Public API
    
    getValue() {
        return this._buildPipeline();
    }
    
    setValue(pipeline) {
        this.options.currentPipeline = pipeline;
        this._initializeStates();
        this._paramWidgets = {};
        Q(this.listContainer).empty();
        this._buildAugmentationItems();
    }
    
    onChange(callback) {
        if (typeof callback === 'function') {
            this._changeCallbacks.push(callback);
        }
    }
    
    getElement() {
        return this.element;
    }
}

// Export
window.AugmentationBuilder = AugmentationBuilder;
