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

        // Page header with save button
        this._buildHeader(container, activeProject);

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
            const tabContent = this._buildTabContent(tabDef.group, trainingSchema);
            this._tabs.addTab(tabDef.id, lang(tabDef.langKey), tabContent);
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
        const desc = Q('<p>', { text: lang('training_page.no_project.description') }).get(0);
        
        const btn = Q('<button>', { 
            class: 'btn btn-primary',
            text: lang('training_page.no_project.button')
        }).get(0);
        
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
     * Build page header with project name
     * @param {HTMLElement} container
     * @param {string} projectName
     */
    _buildHeader: function(container, projectName) {
        const header = Q('<div>', { class: 'training-header' }).get(0);
        
        // Project info
        const projectInfo = Q('<div>', { class: 'training-project-info' }).get(0);
        const projectLabel = Q('<span>', { 
            class: 'project-label',
            text: lang('training_page.project_label')
        }).get(0);
        const projectNameEl = Q('<span>', { 
            class: 'project-name',
            text: projectName
        }).get(0);
        projectInfo.appendChild(projectLabel);
        projectInfo.appendChild(projectNameEl);
        
        header.appendChild(projectInfo);
        container.appendChild(header);
    },

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
        btn.classList.add(isError ? 'btn-error' : 'btn-success');
        
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
            // Get only training section from current config
            const trainingConfig = Config.get('training');
            
            // Save to project
            const result = await API.projects.saveConfig(projectName, { training: trainingConfig });
            
            if (result.status === 'success') {
                // Show success feedback
                this._saveBtn.textContent = lang('training_page.saved');
                this._saveBtn.classList.add('btn-success');
                
                setTimeout(() => {
                    this._saveBtn.textContent = lang('training_page.save_button');
                    this._saveBtn.classList.remove('btn-success');
                    this._saveBtn.disabled = false;
                }, 2000);
            } else {
                throw new Error(result.message || 'Save failed');
            }
        } catch (err) {
            console.error('Failed to save project config:', err);
            this._saveBtn.textContent = lang('training_page.save_error');
            this._saveBtn.classList.add('btn-error');
            
            setTimeout(() => {
                this._saveBtn.textContent = lang('training_page.save_button');
                this._saveBtn.classList.remove('btn-error');
                this._saveBtn.disabled = false;
            }, 3000);
        }
    },

    /**
     * Build content for a tab based on group
     * @param {string} groupName - Group name like "training.model"
     * @param {Object} trainingSchema - Training section schema
     * @returns {HTMLElement}
     */
    _buildTabContent: function(groupName, trainingSchema) {
        const content = document.createElement('div');
        content.className = 'tab-content-wrapper';

        // Collect fields belonging to this group
        const fields = this._getFieldsForGroup(groupName, trainingSchema);

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
                content.appendChild(widget);
            }
        });

        return content;
    },

    /**
     * Get fields belonging to a specific group
     * @param {string} groupName - Group name
     * @param {Object} schema - Schema object
     * @returns {Array}
     */
    _getFieldsForGroup: function(groupName, schema) {
        const fields = [];

        if (!schema?.properties) return fields;

        for (const [key, fieldSchema] of Object.entries(schema.properties)) {
            const fieldGroup = fieldSchema.ui?.group;

            // Direct group match
            if (fieldGroup === groupName) {
                fields.push({
                    key: key,
                    path: `training.${key}`,
                    ...fieldSchema
                });
            }
            // Check nested properties for the group
            else if (fieldSchema.type === 'object' && fieldSchema.properties) {
                // If the nested object itself belongs to the group
                if (fieldGroup === groupName) {
                    fields.push({
                        key: key,
                        path: `training.${key}`,
                        ...fieldSchema
                    });
                }
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

        // Common options
        const options = {
            label: label,
            description: field.description ? lang(field.description) : '',
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
                // Skip complex arrays for now
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
            paramsPath: 'training.loss_params'
        },
        'training.weight_init.type': {
            configPath: 'training.weight_init.defaults',
            paramsPath: 'training.weight_init.params'
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
                labels[val] = val.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
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

        // Get current params from config (if any)
        const currentParams = Config.get(paramsPath) || {};
        const typeParams = currentParams[selectedType] || {};

        // Build widgets for each parameter
        Object.entries(typeDefaults).forEach(([paramKey, defaultValue]) => {
            const currentValue = typeParams[paramKey] ?? defaultValue;
            const paramPath = `${paramsPath}.${selectedType}.${paramKey}`;
            
            const widget = this._buildParamWidget(paramKey, defaultValue, currentValue, paramPath);
            if (widget) {
                container.appendChild(widget);
            }
        });
    },

    /**
     * Build a single parameter widget based on value type
     */
    _buildParamWidget: function(key, defaultValue, currentValue, configPath) {
        const wrapper = document.createElement('div');
        wrapper.className = 'widget widget-param';

        // Label
        const label = document.createElement('label');
        label.className = 'widget-label';
        label.textContent = key.split('_').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');

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
    }
};

// Register page
Pages.register('training', TrainingPage);
