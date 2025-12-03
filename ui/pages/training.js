/**
 * HootSight - Training Page
 * Training configuration with tabbed interface based on schema groups
 */

const TrainingPage = {
    /**
     * Page identifier
     */
    name: 'training',

    /**
     * Tab definitions based on schema groups
     */
    _tabGroups: [
        { id: 'presets', langKey: 'training_page.tabs.presets' },
        { id: 'model', group: 'training.model', langKey: 'training_page.tabs.model' },
        { id: 'hyperparameters', group: 'training.hyperparameters', langKey: 'training_page.tabs.hyperparameters' },
        { id: 'dataloader', group: 'training.dataloader', langKey: 'training_page.tabs.dataloader' },
        { id: 'augmentation', group: 'training.augmentation', langKey: 'training_page.tabs.augmentation' },
        { id: 'optimizer', group: 'training.optimizer', langKey: 'training_page.tabs.optimizer' },
        { id: 'scheduler', group: 'training.scheduler', langKey: 'training_page.tabs.scheduler' },
        { id: 'loss', group: 'training.loss', langKey: 'training_page.tabs.loss' },
        { id: 'checkpoint', group: 'training.checkpoint', langKey: 'training_page.tabs.checkpoint' },
        { id: 'early_stopping', group: 'training.early_stopping', langKey: 'training_page.tabs.early_stopping' },
        { id: 'gradient', group: 'training.gradient', langKey: 'training_page.tabs.gradient' },
        { id: 'runtime', group: 'training.runtime', langKey: 'training_page.tabs.runtime' }
    ],

    /**
     * Widget instances by path
     */
    _widgets: {},

    /**
     * Tabs instance
     */
    _tabs: null,

    /**
     * Container reference
     */
    _container: null,

    /**
     * Build the Training page
     * @param {HTMLElement} container - Container element
     */
    build: async function(container) {
        // Clear container
        container.innerHTML = '';
        this._container = container;
        this._widgets = {};
        this._pendingDependencies = [];

        // Check if a project is selected
        const activeProject = Config.getActiveProject();
        if (!activeProject) {
            this._showNoProjectMessage(container);
            return;
        }

        // Ensure project config is loaded
        await Config.loadProject(activeProject);

        // Initialize dynamic param configs from schema
        this._initDynamicParamConfigs();

        // Create tabs container
        this._tabs = new Tabs('training-tabs');

        // Get schema
        const schema = Config.getSchema();
        const trainingSchema = schema?.properties?.training;

        if (!trainingSchema) {
            const errorDiv = document.createElement('div');
            errorDiv.className = 'error-message';
            errorDiv.textContent = 'Training schema not available';
            container.appendChild(errorDiv);
            return;
        }

        // Build each tab
        this._tabGroups.forEach(tabDef => {
            let tabContent;
            
            // Special handling for specific tabs
            if (tabDef.id === 'presets') {
                tabContent = this._buildPresetsTab();
            } else if (tabDef.id === 'augmentation') {
                tabContent = this._buildAugmentationTab();
            } else {
                tabContent = this._buildTabContent(tabDef.group, trainingSchema, 'training');
            }
            
            // Pass langKey for live translation support
            this._tabs.addTab(tabDef.id, lang(tabDef.langKey), tabContent, { langKey: tabDef.langKey });
        });

        // Bind widget dependencies after all widgets are built
        this._bindDependencies();

        container.appendChild(this._tabs.getElement());
    },

    /**
     * Show message when no project is selected
     * @param {HTMLElement} container
     */
    _showNoProjectMessage: function(container) {
        const wrapper = Q('<div>', { class: 'no-project-message' }).get(0);
        
        const icon = Q('<div>', { class: 'no-project-icon', text: '!' }).get(0);
        const title = Q('<h2>', { text: lang('training_page.no_project.title') }).get(0);
        title.setAttribute('data-lang-key', 'training_page.no_project.title');
        const desc = Q('<p>', { text: lang('training_page.no_project.description') }).get(0);
        desc.setAttribute('data-lang-key', 'training_page.no_project.description');
        
        const btn = Q('<button>', { 
            class: 'btn btn-primary',
            text: lang('training_page.no_project.button')
        }).get(0);
        btn.setAttribute('data-lang-key', 'training_page.no_project.button');
        
        Q(btn).on('click', () => {
            if (typeof Navigation !== 'undefined') {
                Navigation.navigateTo('projects');
            }
        });
        
        wrapper.appendChild(icon);
        wrapper.appendChild(title);
        wrapper.appendChild(desc);
        wrapper.appendChild(btn);
        container.appendChild(wrapper);
    },

    /**
     * Setup header action buttons (called by app.js after page build)
     */
    setupHeaderActions: function() {
        const headerActions = document.getElementById('header-actions');
        if (!headerActions) return;

        // Load System Defaults button
        this._loadDefaultsBtn = Q('<button>', {
            class: 'btn btn-secondary',
            text: lang('training_page.load_defaults_button')
        }).get(0);
        this._loadDefaultsBtn.setAttribute('data-lang-key', 'training_page.load_defaults_button');
        Q(this._loadDefaultsBtn).on('click', () => this._loadSystemDefaults());
        headerActions.appendChild(this._loadDefaultsBtn);

        // Load Project Defaults button
        this._loadProjectBtn = Q('<button>', {
            class: 'btn btn-secondary',
            text: lang('training_page.load_project_button')
        }).get(0);
        Q(this._loadProjectBtn).on('click', () => this._loadProjectDefaults());
        headerActions.appendChild(this._loadProjectBtn);

        // Save button
        this._saveBtn = Q('<button>', {
            class: 'btn btn-primary',
            text: lang('training_page.save_button')
        }).get(0);
        Q(this._saveBtn).on('click', () => this._saveProjectConfig());
        headerActions.appendChild(this._saveBtn);
    },

    /**
     * Build page header with project name (uses global header-info)
    /**
     * Load system defaults (global config without project overrides)
     */
    _loadSystemDefaults: async function() {
        try {
            // Fetch fresh global config
            const response = await fetch('/config');
            if (!response.ok) throw new Error('Failed to fetch global config');
            const data = await response.json();
            
            if (data.config?.training) {
                // Update in-memory config with global training values
                const globalTraining = data.config.training;
                for (const [key, value] of Object.entries(globalTraining)) {
                    Config.set(`training.${key}`, value);
                }
                
                // Rebuild the page to reflect new values
                await this.build(this._container);
                
                // Show feedback
                this._showFeedback(this._loadDefaultsBtn, 'training_page.defaults_loaded');
            }
        } catch (err) {
            console.error('Failed to load system defaults:', err);
            this._showFeedback(this._loadDefaultsBtn, 'training_page.load_error', true);
        }
    },

    /**
     * Load project defaults (project config or global if no project config)
     */
    _loadProjectDefaults: async function() {
        const projectName = Config.getActiveProject();
        if (!projectName) return;
        
        try {
            // Reload project config (this resets to global first, then merges project)
            await Config.loadProject(projectName);
            
            // Rebuild the page to reflect new values
            await this.build(this._container);
            
            // Show feedback
            this._showFeedback(this._loadProjectBtn, 'training_page.project_loaded');
        } catch (err) {
            console.error('Failed to load project defaults:', err);
            this._showFeedback(this._loadProjectBtn, 'training_page.load_error', true);
        }
    },

    /**
     * Show temporary feedback on a button
     */
    _showFeedback: function(btn, langKey, isError = false) {
        const originalText = btn.textContent;
        const originalClass = btn.className;
        
        btn.textContent = lang(langKey);
        
        setTimeout(() => {
            btn.textContent = originalText;
            btn.className = originalClass;
        }, 2000);
    },

    /**
     * Save current training config to project
     */
    _saveProjectConfig: async function() {
        const projectName = Config.getActiveProject();
        if (!projectName) {
            console.error('No active project');
            return;
        }

        // Disable button during save
        this._saveBtn.disabled = true;
        this._saveBtn.textContent = lang('training_page.saving');

        try {
            // Get training section from current config
            const configToSave = {
                training: Config.get('training')
            };
            
            // Save to project
            const result = await API.projects.saveConfig(projectName, configToSave);
            
            if (result.status === 'success') {
                // Show success feedback
                this._saveBtn.textContent = lang('training_page.saved');
                
                setTimeout(() => {
                    this._saveBtn.textContent = lang('training_page.save_button');
                    this._saveBtn.disabled = false;
                }, 2000);
            } else {
                throw new Error(result.message || 'Save failed');
            }
        } catch (err) {
            console.error('Failed to save project config:', err);
            this._saveBtn.textContent = lang('training_page.save_error');
            
            setTimeout(() => {
                this._saveBtn.textContent = lang('training_page.save_button');
                this._saveBtn.disabled = false;
            }, 3000);
        }
    },

    /**
     * Build content for a tab based on group
     * @param {string} groupName - Group name like "training.model"
     * @param {Object} schema - Schema for this section
     * @param {string} basePath - Base path for config keys (default: "training")
     * @returns {HTMLElement}
     */
    _buildTabContent: function(groupName, schema, basePath = 'training') {
        const content = document.createElement('div');
        content.className = 'tab-content-wrapper';

        // Collect fields belonging to this group
        const fields = this._getFieldsForGroup(groupName, schema, basePath);

        if (fields.length === 0) {
            const emptyMsg = document.createElement('p');
            emptyMsg.className = 'text-muted';
            emptyMsg.textContent = lang('training_page.no_settings');
            content.appendChild(emptyMsg);
            return content;
        }

        // Sort by order
        fields.sort((a, b) => (a.ui?.order || 999) - (b.ui?.order || 999));

        // Build widgets for each field
        fields.forEach(field => {
            const widget = this._buildWidget(field);
            if (widget) {
                // Handle both HTMLElement and widget objects with getElement()
                const el = widget.getElement?.() || widget;
                if (el instanceof HTMLElement) {
                    content.appendChild(el);
                }
            }
        });

        return content;
    },

    /**
     * Get fields belonging to a specific group
     * @param {string} groupName - Group name
     * @param {Object} schema - Schema object
     * @param {string} basePath - Base path for config keys
     * @returns {Array}
     */
    _getFieldsForGroup: function(groupName, schema, basePath = 'training') {
        const fields = [];

        if (!schema?.properties) return fields;

        for (const [key, fieldSchema] of Object.entries(schema.properties)) {
            const fieldGroup = fieldSchema.ui?.group;

            // Direct group match
            if (fieldGroup === groupName) {
                fields.push({
                    key: key,
                    path: `${basePath}.${key}`,
                    ...fieldSchema
                });
            }
        }

        return fields;
    },

    /**
     * Build a widget based on field schema
     * @param {Object} field - Field schema with key and path
     * @returns {HTMLElement|null}
     */
    _buildWidget: function(field) {
        const widgetType = field.ui?.widget;
        const configValue = Config.get(field.path, field.default);

        // Generate a human-readable label from the key (e.g., "model_type" -> "Model Type")
        const label = field.key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ');

        // Common options - include lang keys for live translation
        const options = {
            label: label,
            description: field.description ? lang(field.description) : '',
            descriptionLangKey: field.description || null, // Store the lang key for live refresh
            default: field.default,
            disabled: field.ui?.disabled || false,
            visible: field.ui?.visible !== false
        };

        let widget = null;

        switch (widgetType) {
            case 'dropdown':
                widget = this._buildDropdown(field, options, configValue);
                break;

            case 'switch':
                widget = this._buildSwitch(field, options, configValue);
                break;

            case 'slider':
                widget = this._buildSlider(field, options, configValue);
                break;

            case 'number':
                widget = this._buildNumber(field, options, configValue);
                break;

            case 'text':
                widget = this._buildText(field, options, configValue);
                break;

            case 'auto_or_number':
                widget = this._buildAutoOrNumber(field, options, configValue);
                break;

            case 'nested':
                // Skip nested widgets that are managed by dynamic params
                if (field.ui?.visible === false) {
                    return null;
                }
                widget = this._buildNested(field, options);
                break;

            case 'array':
                // Handle fixed-size number arrays (like normalize mean/std)
                if (field.items?.type === 'number' && field.minItems && field.minItems === field.maxItems) {
                    widget = this._buildNumberArray(field, options, configValue);
                }
                // Skip other complex arrays
                break;

            default:
                // Unknown widget type - skip
                break;
        }

        if (widget) {
            this._widgets[field.path] = widget;
        }

        return widget?.getElement?.() || widget;
    },

    /**
     * Dynamic param type configs - maps type fields to their param sources
     * Key can be either field.key (for top-level) or field.path (for nested)
     */
    _dynamicParamConfigs: {
        'optimizer_type': {
            configPath: 'optimizers.defaults',
            paramsPath: 'training.optimizer_params'
        },
        'scheduler_type': {
            configPath: 'schedulers.defaults',
            paramsPath: 'training.scheduler_params'
        },
        'loss_type': {
            configPath: 'losses.defaults',
            paramsPath: 'training.loss_params',
            paramTypes: null // Will be loaded from schema
        },
        'training.weight_init.type': {
            configPath: 'training.weight_init.defaults',
            paramsPath: 'training.weight_init.params',
            paramTypes: null // Will be loaded from schema
        }
    },

    /**
     * Initialize dynamic param configs from schema
     */
    _initDynamicParamConfigs: function() {
        const schema = Config.getSchema();
        
        // Load weight_init param types
        const weightInitParamTypes = schema?.properties?.training?.properties?.weight_init?.properties?.params?.ui?.param_types;
        if (weightInitParamTypes) {
            this._dynamicParamConfigs['training.weight_init.type'].paramTypes = weightInitParamTypes;
        }
        
        // Load loss_params param types
        const lossParamTypes = schema?.properties?.training?.properties?.loss_params?.ui?.param_types;
        if (lossParamTypes) {
            this._dynamicParamConfigs['loss_type'].paramTypes = lossParamTypes;
        }
    },

    /**
     * Build dropdown widget
     */
    _buildDropdown: function(field, options, value) {
        const enumValues = field.enum || [];
        const enumDescriptor = field.enum_descriptor || null;
        const dependsOn = field.ui?.depends_on || null;

        // Determine initial options based on dependency
        let initialOptions = enumValues;
        if (dependsOn && field[dependsOn.source]) {
            // Get source value from config (not from widget, as it may not exist yet)
            const sourceConfigPath = `training.${dependsOn.field}`;
            const sourceValue = Config.get(sourceConfigPath);
            const variants = field[dependsOn.source];
            if (sourceValue && variants[sourceValue]) {
                initialOptions = variants[sourceValue];
            }
        }

        // Build option labels - use raw value, optionally capitalized
        const buildLabels = (opts) => {
            const labels = {};
            opts.forEach(val => {
                const strVal = String(val);
                labels[val] = strVal.includes('_') 
                    ? strVal.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ')
                    : strVal;
            });
            return labels;
        };

        // Ensure current value is valid for filtered options
        let currentValue = value ?? field.default;
        if (!initialOptions.includes(currentValue) && initialOptions.length > 0) {
            currentValue = initialOptions[0];
        }

        // If we have an enumDescriptor, don't show static description (dynamic will handle it)
        const dropdownOptions = {
            ...options,
            options: initialOptions,
            optionLabels: buildLabels(initialOptions),
            enumDescriptor: enumDescriptor,
            default: currentValue
        };
        
        // Remove static description if we have dynamic descriptions
        if (enumDescriptor) {
            dropdownOptions.description = '';
        }

        const dropdown = new Dropdown(field.key, dropdownOptions);

        dropdown.onChange(newValue => {
            Config.set(field.path, newValue);
        });

        // Set up dependency listener if this dropdown depends on another field
        if (dependsOn && field[dependsOn.source]) {
            const sourceWidgetPath = `training.${dependsOn.field}`;
            const variants = field[dependsOn.source];
            
            // Store dependency info for later binding
            this._pendingDependencies = this._pendingDependencies || [];
            this._pendingDependencies.push({
                targetDropdown: dropdown,
                targetPath: field.path,
                sourceWidgetPath: sourceWidgetPath,
                variants: variants,
                buildLabels: buildLabels
            });
        }

        // Check if this is a type selector that needs dynamic params
        // Try both field.key (top-level) and field.path (nested) as keys
        const dynamicConfig = this._dynamicParamConfigs[field.key] || this._dynamicParamConfigs[field.path];
        if (dynamicConfig) {
            // Create wrapper to hold dropdown + dynamic params
            const wrapper = document.createElement('div');
            wrapper.className = 'type-selector-wrapper';
            wrapper.appendChild(dropdown.getElement());

            // Create dynamic params container
            const paramsContainer = document.createElement('div');
            paramsContainer.className = 'dynamic-params-container';
            paramsContainer.id = `${field.key}-params`;
            wrapper.appendChild(paramsContainer);

            // Build initial params
            this._buildDynamicParams(paramsContainer, currentValue, dynamicConfig);

            // Update params when type changes
            dropdown.onChange(newValue => {
                this._buildDynamicParams(paramsContainer, newValue, dynamicConfig);
            });

            // Return wrapper with a mock getElement for compatibility
            return {
                getElement: () => wrapper,
                get: () => dropdown.get(),
                set: (v) => dropdown.set(v),
                onChange: (cb) => dropdown.onChange(cb)
            };
        }

        return dropdown;
    },

    /**
     * Build dynamic parameter widgets based on selected type
     */
    _buildDynamicParams: function(container, selectedType, dynamicConfig) {
        // Clear existing params
        container.innerHTML = '';

        // Get defaults for this type from config
        const defaults = Config.get(dynamicConfig.configPath);
        if (!defaults || !defaults[selectedType]) {
            return;
        }

        const typeDefaults = defaults[selectedType];
        const paramsPath = dynamicConfig.paramsPath;
        const paramTypes = dynamicConfig.paramTypes || {};

        // Get current params from config (if any)
        const currentParams = Config.get(paramsPath) || {};
        const typeParams = currentParams[selectedType] || {};

        // Build widgets for each parameter
        Object.entries(typeDefaults).forEach(([paramKey, defaultValue]) => {
            const currentValue = typeParams[paramKey] ?? defaultValue;
            const paramPath = `${paramsPath}.${selectedType}.${paramKey}`;
            
            const widget = this._buildParamWidget(paramKey, defaultValue, currentValue, paramPath, paramTypes);
            if (widget) {
                container.appendChild(widget);
            }
        });
    },

    /**
     * Build a single parameter widget based on value type
     */
    _buildParamWidget: function(key, defaultValue, currentValue, configPath, paramTypes) {
        const wrapper = document.createElement('div');
        wrapper.className = 'widget widget-param';

        // Label
        const label = document.createElement('label');
        label.className = 'widget-label';
        label.textContent = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

        // Check if this param has a defined type in schema (dropdown)
        if (paramTypes && paramTypes[key] && paramTypes[key].widget === 'dropdown') {
            wrapper.appendChild(label);
            
            const options = paramTypes[key].enum;
            const optionLabels = {};
            options.forEach(opt => {
                optionLabels[opt] = opt.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
            });
            
            const dropdown = new Dropdown(`param-${key}`, {
                options: options,
                optionLabels: optionLabels,
                default: currentValue ?? defaultValue
            });
            
            dropdown.onChange(newValue => {
                Config.set(configPath, newValue);
            });
            
            wrapper.appendChild(dropdown.getElement());
            return wrapper;
        }

        const valueType = typeof defaultValue;
        
        if (valueType === 'boolean') {
            // Switch for boolean
            wrapper.classList.add('widget-param-boolean');
            
            const switchWidget = new Switch(`param-${key}`, {
                label: label.textContent,
                default: currentValue
            });
            
            switchWidget.onChange(newValue => {
                Config.set(configPath, newValue);
            });
            
            // Use the switch element directly
            const switchEl = switchWidget.getElement();
            wrapper.appendChild(switchEl);
        }
        else if (valueType === 'number') {
            wrapper.appendChild(label);
            
            // Number widget - smart step calculation
            const isInteger = Number.isInteger(defaultValue);
            let step = 1;
            let precision = 0;
            
            if (!isInteger) {
                // Calculate step based on the magnitude of the default value
                const absVal = Math.abs(defaultValue);
                if (absVal === 0) {
                    step = 0.01;
                    precision = 2;
                } else if (absVal < 0.0001) {
                    // Very small values like eps: 1e-8
                    step = absVal / 10;
                    precision = 10;
                } else if (absVal < 0.01) {
                    step = 0.0001;
                    precision = 6;
                } else if (absVal < 1) {
                    step = 0.01;
                    precision = 4;
                } else {
                    step = 0.1;
                    precision = 2;
                }
            }
            
            const numWidget = new NumberInput(`param-${key}`, {
                default: currentValue,
                step: step,
                precision: precision,
                integer: isInteger
            });
            
            numWidget.onChange(newValue => {
                Config.set(configPath, newValue);
            });
            
            // Add just the controls, not the full widget with label
            const controls = numWidget.getElement().querySelector('.number-widget-controls');
            if (controls) {
                wrapper.appendChild(controls);
            } else {
                wrapper.appendChild(numWidget.getElement());
            }
        }
        else if (Array.isArray(defaultValue)) {
            wrapper.appendChild(label);
            
            // Array input (comma separated)
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'widget-input';
            input.value = Array.isArray(currentValue) ? currentValue.join(', ') : defaultValue.join(', ');
            input.placeholder = defaultValue.join(', ');
            
            input.addEventListener('change', () => {
                const parts = input.value.split(',').map(s => s.trim());
                const parsed = parts.map(p => {
                    const num = parseFloat(p);
                    return isNaN(num) ? p : num;
                });
                Config.set(configPath, parsed);
            });
            wrapper.appendChild(input);
        }
        else if (defaultValue === null) {
            wrapper.appendChild(label);
            
            // Nullable - text input
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'widget-input';
            input.value = currentValue ?? '';
            input.placeholder = 'null';
            
            input.addEventListener('change', () => {
                const val = input.value.trim() === '' ? null : input.value;
                Config.set(configPath, val);
            });
            wrapper.appendChild(input);
        }
        else {
            wrapper.appendChild(label);
            
            // String input
            const input = document.createElement('input');
            input.type = 'text';
            input.className = 'widget-input';
            input.value = currentValue ?? defaultValue ?? '';
            
            input.addEventListener('change', () => {
                Config.set(configPath, input.value);
            });
            wrapper.appendChild(input);
        }

        return wrapper;
    },

    /**
     * Bind pending dependencies after all widgets are built
     */
    _bindDependencies: function() {
        if (!this._pendingDependencies) return;

        this._pendingDependencies.forEach(dep => {
            const sourceWidget = this._widgets[dep.sourceWidgetPath];
            if (sourceWidget && sourceWidget.onChange) {
                sourceWidget.onChange(newValue => {
                    const newOptions = dep.variants[newValue] || [];
                    dep.targetDropdown.setOptions(newOptions, dep.buildLabels(newOptions));
                    
                    // Update config with the new value (first option of filtered list)
                    const newSelectedValue = dep.targetDropdown.get();
                    if (dep.targetPath) {
                        Config.set(dep.targetPath, newSelectedValue);
                    }
                });
            }
        });

        this._pendingDependencies = [];
    },

    /**
     * Build switch widget
     */
    _buildSwitch: function(field, options, value) {
        const switchWidget = new Switch(field.key, {
            ...options,
            default: value ?? field.default ?? false
        });

        switchWidget.onChange(newValue => {
            Config.set(field.path, newValue);
        });

        return switchWidget;
    },

    /**
     * Build slider widget
     */
    _buildSlider: function(field, options, value) {
        const slider = new Slider(field.key, {
            ...options,
            min: field.minimum ?? 0,
            max: field.maximum ?? 100,
            step: field.ui?.step ?? 1,
            scale: field.ui?.scale || 'linear',
            precision: field.ui?.precision ?? 2,
            default: value ?? field.default
        });

        slider.onChange(newValue => {
            Config.set(field.path, newValue);
        });

        return slider;
    },

    /**
     * Build number input widget
     */
    _buildNumber: function(field, options, value) {
        const isInteger = field.type === 'integer';
        const numWidget = new NumberInput(field.key, {
            ...options,
            min: field.minimum ?? null,
            max: field.maximum ?? null,
            step: field.ui?.step ?? (isInteger ? 1 : 0.01),
            precision: field.ui?.precision ?? (isInteger ? 0 : 4),
            integer: isInteger,
            default: value ?? field.default
        });

        numWidget.onChange(newValue => {
            Config.set(field.path, newValue);
        });

        return numWidget;
    },

    /**
     * Build text input widget
     */
    _buildText: function(field, options, value) {
        const text = new TextInput(field.key, {
            ...options,
            default: value ?? field.default ?? ''
        });

        text.onChange(newValue => {
            Config.set(field.path, newValue);
        });

        return text;
    },

    /**
     * Build auto or number widget
     */
    _buildAutoOrNumber: function(field, options, value) {
        // Parse number constraints from oneOf schema if available
        let min = field.minimum ?? 1;
        let max = field.maximum ?? 9999;
        let isInteger = true;
        let numberDefault = min;
        
        // Try to get constraints from oneOf (second option is usually the number type)
        if (field.oneOf && Array.isArray(field.oneOf)) {
            for (const opt of field.oneOf) {
                if (opt.type === 'integer' || opt.type === 'number') {
                    if (opt.minimum !== undefined) min = opt.minimum;
                    if (opt.maximum !== undefined) max = opt.maximum;
                    if (opt.default !== undefined) numberDefault = opt.default;
                    isInteger = opt.type === 'integer';
                    break;
                }
            }
        }
        
        const widget = new AutoOrNumber(field.key, {
            label: options.label || '',
            description: options.description || '',
            min: min,
            max: max,
            default: value,
            numberDefault: numberDefault,
            integer: isInteger
        });
        
        widget.onChange((newValue) => {
            Config.set(field.path, newValue);
        });
        
        return widget;
    },

    /**
     * Build fixed-size number array widget (e.g., RGB values for normalize mean/std)
     */
    _buildNumberArray: function(field, options, configValue) {
        const container = document.createElement('div');
        container.className = 'widget widget-array-inline';
        container.id = field.key;

        // Label
        if (options.label) {
            const label = document.createElement('label');
            label.className = 'widget-label';
            label.textContent = options.label;
            container.appendChild(label);
        }

        // Inputs row
        const inputsRow = document.createElement('div');
        inputsRow.className = 'array-inputs-row';
        
        const arraySize = field.minItems || 3;
        const currentValue = Array.isArray(configValue) ? configValue : (field.default || []);
        const inputs = [];
        
        const itemSchema = field.items || {};
        const min = itemSchema.minimum ?? 0;
        const max = itemSchema.maximum ?? 1;
        const step = itemSchema.step || 0.001;

        for (let i = 0; i < arraySize; i++) {
            const numberWidget = new NumberInput(`${field.key}-${i}`, {
                label: '',
                default: currentValue[i] ?? 0,
                min: min,
                max: max,
                step: step,
                precision: 3
            });
            
            const widgetEl = numberWidget.getElement();
            widgetEl.classList.add('array-item');
            inputsRow.appendChild(widgetEl);
            inputs.push(numberWidget);
            
            // Update config on change
            numberWidget.onChange(() => {
                const newArray = inputs.map(inp => inp.getValue());
                Config.set(field.path, newArray);
            });
        }

        container.appendChild(inputsRow);

        // Description
        if (options.description) {
            const desc = document.createElement('div');
            desc.className = 'widget-description';
            desc.textContent = options.description;
            container.appendChild(desc);
        }

        return {
            getElement: () => container,
            getValue: () => inputs.map(inp => inp.getValue()),
            setValue: (val) => {
                if (Array.isArray(val)) {
                    inputs.forEach((inp, i) => {
                        if (val[i] !== undefined) inp.setValue(val[i]);
                    });
                }
            }
        };
    },

    /**
     * Build nested object widget (collapsible group)
     */
    _buildNested: function(field, options) {
        const container = document.createElement('div');
        container.className = 'widget widget-nested';
        container.id = field.key;

        // Header (collapsible)
        const header = document.createElement('div');
        header.className = 'nested-header';
        header.textContent = options.label;
        container.appendChild(header);

        // Content
        const content = document.createElement('div');
        content.className = 'nested-content';

        // Build child widgets
        if (field.properties) {
            for (const [childKey, childSchema] of Object.entries(field.properties)) {
                const childField = {
                    key: childKey,
                    path: `${field.path}.${childKey}`,
                    ...childSchema
                };

                const childWidget = this._buildWidget(childField);
                if (childWidget) {
                    const el = childWidget.getElement?.() || childWidget;
                    if (el instanceof HTMLElement) {
                        content.appendChild(el);
                    }
                }
            }
        }

        container.appendChild(content);

        return { getElement: () => container };
    },

    /**
     * Get current values from all widgets
     * @returns {Object}
     */
    getValues: function() {
        const values = {};
        for (const [path, widget] of Object.entries(this._widgets)) {
            if (widget.getValue) {
                values[path] = widget.getValue();
            }
        }
        return values;
    },

    /**
     * Build the Augmentation tab with split view: settings left, preview right
     * @returns {HTMLElement}
     */
    _buildAugmentationTab: function() {
        const content = document.createElement('div');
        content.className = 'tab-content-wrapper augmentation-tab-split';

        // Get augmentation defaults from config
        const augDefaults = Config.get('augmentations.defaults', {});
        
        // Filter out non-configurable augmentations
        const configurableDefaults = {};
        for (const [augName, defaults] of Object.entries(augDefaults)) {
            if (augName !== 'to_tensor' && augName !== 'normalize' && augName !== 'compose') {
                configurableDefaults[augName] = defaults;
            }
        }

        // === LEFT PANEL: Settings ===
        const leftPanel = Q('<div>', { class: 'augmentation-settings-panel' }).get(0);
        
        // Train augmentation builder
        const trainLabel = Q('<h3>', { 
            class: 'augmentation-section-title', 
            text: lang('training_page.augmentation.train_title') 
        }).get(0);
        leftPanel.appendChild(trainLabel);

        const trainPipeline = Config.get('training.augmentation.train', []);
        const trainBuilder = new AugmentationBuilder('augmentation-train', {
            phase: 'train',
            label: '',
            description: lang('training_page.augmentation.train_description'),
            augmentationDefaults: configurableDefaults,
            currentPipeline: trainPipeline
        });
        
        trainBuilder.onChange(pipeline => {
            Config.set('training.augmentation.train', pipeline);
        });
        
        this._widgets['training.augmentation.train'] = trainBuilder;
        leftPanel.appendChild(trainBuilder.getElement());

        // Spacer
        const spacer = Q('<div>', { class: 'augmentation-section-spacer' }).get(0);
        leftPanel.appendChild(spacer);

        // Validation augmentation builder
        const valLabel = Q('<h3>', { 
            class: 'augmentation-section-title', 
            text: lang('training_page.augmentation.val_title') 
        }).get(0);
        leftPanel.appendChild(valLabel);

        const valPipeline = Config.get('training.augmentation.val', []);
        const valBuilder = new AugmentationBuilder('augmentation-val', {
            phase: 'val',
            label: '',
            description: lang('training_page.augmentation.val_description'),
            augmentationDefaults: configurableDefaults,
            currentPipeline: valPipeline
        });
        
        valBuilder.onChange(pipeline => {
            Config.set('training.augmentation.val', pipeline);
        });
        
        this._widgets['training.augmentation.val'] = valBuilder;
        leftPanel.appendChild(valBuilder.getElement());

        content.appendChild(leftPanel);

        // === RIGHT PANEL: Preview ===
        const rightPanel = Q('<div>', { class: 'augmentation-preview-panel' }).get(0);
        
        // Preview box (1:1 aspect ratio)
        const previewBox = Q('<div>', { class: 'augmentation-preview-box' }).get(0);
        const previewImage = Q('<img>', { class: 'augmentation-preview-image', alt: 'Preview' }).get(0);
        previewImage.src = ''; // Empty initially
        const previewPlaceholder = Q('<div>', { class: 'augmentation-preview-placeholder' }).get(0);
        previewPlaceholder.innerHTML = '<span>' + lang('training_page.augmentation.preview_placeholder') + '</span>';
        previewBox.appendChild(previewImage);
        previewBox.appendChild(previewPlaceholder);
        rightPanel.appendChild(previewBox);
        
        // Store references for preview functionality
        this._previewImage = previewImage;
        this._previewPlaceholder = previewPlaceholder;
        this._currentImagePath = null; // Store the current image path for augmentation
        this._originalImageBase64 = null; // Store original for reset
        
        // Buttons using ActionButton widget
        const buttonGroup = Q('<div>', { class: 'augmentation-preview-buttons' }).get(0);
        
        const randomBtn = new ActionButton('aug-random', {
            label: lang('training_page.augmentation.btn_random'),
            className: 'btn btn-secondary',
            onClick: () => this._loadRandomPreviewImage()
        });
        
        const trainBtn = new ActionButton('aug-train', {
            label: lang('training_page.augmentation.btn_train'),
            className: 'btn btn-primary',
            onClick: () => this._applyAugmentationPreview('train')
        });
        
        const valBtn = new ActionButton('aug-val', {
            label: lang('training_page.augmentation.btn_val'),
            className: 'btn btn-secondary',
            onClick: () => this._applyAugmentationPreview('val')
        });
        
        buttonGroup.appendChild(randomBtn.getElement());
        buttonGroup.appendChild(trainBtn.getElement());
        buttonGroup.appendChild(valBtn.getElement());
        rightPanel.appendChild(buttonGroup);

        content.appendChild(rightPanel);

        return content;
    },

    /**
     * Build the Presets tab
     * @returns {HTMLElement}
     */
    _buildPresetsTab: function() {
        const content = document.createElement('div');
        content.className = 'tab-content-wrapper presets-tab';

        const header = document.createElement('div');
        header.className = 'presets-header';
        
        const title = document.createElement('h2');
        title.textContent = lang('training_page.presets.title');
        title.setAttribute('data-lang-key', 'training_page.presets.title');
        
        const description = document.createElement('p');
        description.className = 'presets-description';
        description.textContent = lang('training_page.presets.description');
        description.setAttribute('data-lang-key', 'training_page.presets.description');
        
        header.appendChild(title);
        header.appendChild(description);
        content.appendChild(header);

        // Search and filter controls container
        const controlsContainer = document.createElement('div');
        controlsContainer.className = 'presets-controls';

        // Search input using TextInput widget
        this._searchWidget = new TextInput('presets-search', {
            label: '',
            placeholder: lang('training_page.presets.search_placeholder') || 'Search presets by name, description, task...',
            placeholderLangKey: 'training_page.presets.search_placeholder',
            default: ''
        });
        
        const searchWrapper = document.createElement('div');
        searchWrapper.className = 'presets-search-wrapper';
        searchWrapper.appendChild(this._searchWidget.getElement());
        
        // Search stats
        const searchStats = document.createElement('span');
        searchStats.className = 'presets-search-stats';
        searchWrapper.appendChild(searchStats);
        this._presetsSearchStats = searchStats;
        
        controlsContainer.appendChild(searchWrapper);

        // Only Compatible switch
        this._compatibleOnlySwitch = new Switch('presets-compatible-only', {
            label: lang('training_page.presets.only_compatible') || 'Only Compatible',
            labelLangKey: 'training_page.presets.only_compatible',
            description: '',
            default: false
        });
        
        controlsContainer.appendChild(this._compatibleOnlySwitch.getElement());
        content.appendChild(controlsContainer);

        const presetsContainer = document.createElement('div');
        presetsContainer.className = 'presets-container';
        content.appendChild(presetsContainer);

        // Store references
        this._presetsContainer = presetsContainer;
        this._presetsData = [];
        this._currentDatasetType = 'unknown';

        // Search event handler
        let searchTimeout;
        this._searchWidget.onChange((value) => {
            clearTimeout(searchTimeout);
            searchTimeout = setTimeout(() => {
                this._filterPresets();
            }, 150);
        });

        // Compatible only switch handler
        this._compatibleOnlySwitch.onChange((value) => {
            this._filterPresets();
        });

        // Load presets
        this._loadPresets(presetsContainer);

        return content;
    },

    /**
     * Filter presets based on search query and compatibility
     */
    _filterPresets: function() {
        if (!this._presetsData || this._presetsData.length === 0) {
            return;
        }

        const container = this._presetsContainer;
        const stats = this._presetsSearchStats;
        const query = this._searchWidget ? this._searchWidget.get() : '';
        const onlyCompatible = this._compatibleOnlySwitch ? this._compatibleOnlySwitch.get() : false;

        let filteredPresets = this._presetsData;

        // Filter by compatibility first
        if (onlyCompatible && this._currentDatasetType && this._currentDatasetType !== 'unknown') {
            filteredPresets = filteredPresets.filter(preset => {
                if (!preset.dataset_types || preset.dataset_types.length === 0) {
                    return true; // No restriction = compatible with all
                }
                return preset.dataset_types.includes(this._currentDatasetType);
            });
        }

        // Then apply search filter with 0.97+ threshold
        let matchedIds = new Set();
        if (query && query.trim() !== '') {
            const results = FuzzySearch.search(
                filteredPresets, 
                query, 
                ['name', 'description', 'task'],
                0.97  // High confidence threshold
            );
            matchedIds = new Set(results.map(r => r.item.id));
            
            // Store scores for ordering
            this._searchScores = {};
            results.forEach(r => {
                this._searchScores[r.item.id] = r.score;
            });
        } else {
            // No search query - show all filtered presets
            matchedIds = new Set(filteredPresets.map(p => p.id));
            this._searchScores = {};
        }

        let visibleCount = 0;

        // Update visibility and order of preset cards
        Array.from(container.children).forEach(card => {
            const presetId = card.dataset.presetId;
            const preset = this._presetsData.find(p => p.id === presetId);
            
            // Check compatibility filter
            let passesCompatibility = true;
            if (onlyCompatible && preset && this._currentDatasetType && this._currentDatasetType !== 'unknown') {
                if (preset.dataset_types && preset.dataset_types.length > 0) {
                    passesCompatibility = preset.dataset_types.includes(this._currentDatasetType);
                }
            }
            
            if (matchedIds.has(presetId) && passesCompatibility) {
                card.style.display = '';
                // Set order based on score
                const score = this._searchScores[presetId];
                card.style.order = score ? Math.round((1 - score) * 100) : 50;
                visibleCount++;
            } else {
                card.style.display = 'none';
            }
        });

        // Update stats
        if (query || onlyCompatible) {
            stats.textContent = lang('training_page.presets.search_results')
                ?.replace('{count}', visibleCount)
                ?.replace('{total}', this._presetsData.length)
                || `${visibleCount} / ${this._presetsData.length}`;
        } else {
            stats.textContent = '';
        }
    },

    /**
     * Load and display presets
     * @param {HTMLElement} container
     */
    _loadPresets: async function(container) {
        const projectName = Config.getActiveProject();
        if (!projectName) {
            container.innerHTML = '<p class="error-message">' + lang('training_page.no_project.description') + '</p>';
            return;
        }

        container.innerHTML = '<p class="loading-message">' + lang('training_page.presets.loading') + '</p>';

        try {
            // Load presets and dataset type in parallel
            const [presetsResponse, datasetTypeData] = await Promise.all([
                fetch('/presets'),
                API.datasetEditor.getDatasetType(projectName)
            ]);
            
            const data = await presetsResponse.json();
            const currentDatasetType = datasetTypeData.type || 'unknown';
            
            // Store dataset type for filtering
            this._currentDatasetType = currentDatasetType;

            if (!data.presets || data.presets.length === 0) {
                container.innerHTML = '<p class="info-message">' + lang('training_page.presets.no_presets') + '</p>';
                return;
            }

            container.innerHTML = '';
            
            // Store presets data for search
            this._presetsData = data.presets;

            data.presets.forEach(preset => {
                const card = PresetCard.create(preset, async (selectedPreset, selectedSize) => {
                    await this._applyPreset(selectedPreset.id, selectedSize);
                }, currentDatasetType);
                // Add preset ID to card for search filtering
                card.dataset.presetId = preset.id;
                container.appendChild(card);
            });

        } catch (error) {
            console.error('Failed to load presets:', error);
            container.innerHTML = '<p class="error-message">' + lang('training_page.presets.apply_error') + '</p>';
        }
    },

    /**
     * Apply a preset to the UI (loads values into Config, does not save to DB)
     * User must click Save to persist the changes
     * @param {string} presetId
     * @param {string} sizeVariant - Size variant key (tiny, small, medium, large, huge)
     */
    _applyPreset: async function(presetId, sizeVariant = 'medium') {
        const projectName = Config.getActiveProject();
        if (!projectName) {
            Modal.alert(lang('training_page.no_project.description'));
            return;
        }

        try {
            // Fetch the preset data
            const response = await fetch(`/presets/${presetId}`);
            const result = await response.json();

            if (result.status !== 'success' || !result.preset) {
                Modal.alert(lang('training_page.presets.apply_error'));
                return;
            }

            const preset = result.preset;
            
            // Get config from the selected size variant
            let trainingConfig = null;
            
            // New structure with configs object
            if (preset.configs && preset.configs[sizeVariant]) {
                trainingConfig = preset.configs[sizeVariant].config?.training;
            }
            // Fallback to old structure
            else if (preset.config && preset.config.training) {
                trainingConfig = preset.config.training;
            }
            
            if (!trainingConfig) {
                Modal.alert(lang('training_page.presets.apply_error'));
                return;
            }

            // Apply all training config values to the in-memory Config
            // This uses deep merge to properly set nested objects
            this._applyPresetValues(trainingConfig, 'training');

            // Get schema for rebuilding
            const schema = Config.getSchema();
            const trainingSchema = schema?.properties?.training;

            if (!trainingSchema) {
                Modal.alert(lang('training_page.presets.apply_error'));
                return;
            }

            // Clear widgets cache
            this._widgets = {};
            
            // Clear and rebuild tab contents with new values
            this._tabGroups.forEach(tabDef => {
                let tabContent;
                
                if (tabDef.id === 'presets') {
                    tabContent = this._buildPresetsTab();
                } else if (tabDef.id === 'augmentation') {
                    tabContent = this._buildAugmentationTab();
                } else {
                    tabContent = this._buildTabContent(tabDef.group, trainingSchema, 'training');
                }
                
                this._tabs.setContent(tabDef.id, tabContent);
            });
            
            // Re-bind widget dependencies
            this._bindDependencies();
            
            // Show success message
            Modal.alert(lang('training_page.presets.applied'));

        } catch (error) {
            console.error('Failed to apply preset:', error);
            Modal.alert(lang('training_page.presets.apply_error'));
        }
    },

    /**
     * Recursively apply preset values to Config
     * @param {Object} values - Values to apply
     * @param {string} basePath - Base config path
     */
    _applyPresetValues: function(values, basePath) {
        for (const [key, value] of Object.entries(values)) {
            const fullPath = `${basePath}.${key}`;
            Config.set(fullPath, value);
        }
    },

    /**
     * Load a random image for preview (original, no augmentation)
     */
    _loadRandomPreviewImage: async function() {
        const projectName = Config.getActiveProject();
        if (!projectName) return;

        try {
            // Request a random preview image with no transforms
            const response = await fetch(`/projects/${projectName}/augmentation/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    phase: 'train', 
                    transforms: [] // Empty transforms = just get original image
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success' && data.original_image) {
                this._previewImage.src = 'data:image/jpeg;base64,' + data.original_image;
                this._previewPlaceholder.style.display = 'none';
                this._previewImage.style.display = 'block';
                // Store the image path and original for later use
                this._currentImagePath = data.image_path;
                this._originalImageBase64 = data.original_image;
            } else {
                console.error('Failed to load preview image:', data.message);
            }
        } catch (err) {
            console.error('Error loading preview image:', err);
        }
    },

    /**
     * Apply augmentation to the current preview image
     * @param {string} phase - 'train' or 'val'
     */
    _applyAugmentationPreview: async function(phase) {
        const projectName = Config.getActiveProject();
        if (!projectName) return;

        // If no image loaded yet, load one first
        if (!this._currentImagePath) {
            await this._loadRandomPreviewImage();
            if (!this._currentImagePath) return; // Still no image
        }

        // Get the appropriate transforms
        const transforms = phase === 'train' 
            ? Config.get('training.augmentation.train', [])
            : Config.get('training.augmentation.val', []);

        try {
            const response = await fetch(`/projects/${projectName}/augmentation/preview`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    phase: phase, 
                    transforms: transforms,
                    image_path: this._currentImagePath // Use the same image!
                })
            });
            
            const data = await response.json();
            
            if (data.status === 'success' && data.augmented_image) {
                this._previewImage.src = 'data:image/jpeg;base64,' + data.augmented_image;
                this._previewPlaceholder.style.display = 'none';
                this._previewImage.style.display = 'block';
            } else {
                console.error('Augmentation preview failed:', data.message);
            }
        } catch (err) {
            console.error('Error applying augmentation:', err);
        }
    }
};

// Register page
Pages.register('training', TrainingPage);

