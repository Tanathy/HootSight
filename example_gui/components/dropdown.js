class Dropdown {
    constructor(identifier, options = [], defaultValue = "", title = "", description = "", valueMapping = null) {
        this.identifier = identifier;
        // Process options: if they're objects with {label, value}, extract label and value
        this.options = [];
        this.optionLabels = {}; // Map from value to display label
        
        if (Array.isArray(options)) {
            options.forEach(opt => {
                if (typeof opt === 'object' && opt !== null && 'value' in opt && 'label' in opt) {
                    // Object format: {label: "...", value: "..."}
                    this.options.push(opt.value);
                    this.optionLabels[opt.value] = typeof opt.label === 'string' && opt.label.startsWith('ui.') 
                        ? lang(opt.label) 
                        : opt.label;
                } else {
                    // Simple string or other format
                    this.options.push(opt);
                    this.optionLabels[opt] = typeof opt === 'string' && opt.startsWith('ui.')
                        ? lang(opt)
                        : String(opt);
                }
            });
        }
        
        // Keep an immutable snapshot of the original option labels so we can restore if DOM text nodes get nuked
        this._originalOptionsSnapshot = [...this.options];
        this.valueMapping = valueMapping && typeof valueMapping === 'object' ? valueMapping : null;
        this.onChangeCallback = null;
        this.dropdownWrapper = Q('<div>', { class: 'dropdown_wrapper' }).get(0);
        if (title) {
            const heading = Q('<h3>', { class: 'inputs_title', text: title }).get(0);
            Q(this.dropdownWrapper).append(heading);
        }
        if (description) {
            const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: description }).get(0);
            Q(this.dropdownWrapper).append(descriptionHeading);
        }
        
        const dropdownContent = Q('<div>', { class: 'dropdown_content' }).get(0);
        this.selectedDisplay = Q('<div>', { class: 'dropdown_selected' }).get(0);
        this.arrow = Q('<span>', { class: 'dropdown_arrow', text: '▼' }).get(0);
        this.optionsContainer = Q('<div>', { class: 'dropdown_options' }).get(0);
        
        this.hiddenInput = Q('<input>', { class: 'hidden_input', type: 'hidden', id: identifier }).get(0);
        
        // Determine display text for the default value
        const defaultDisplayValue = defaultValue 
            ? (this.optionLabels[defaultValue] || String(defaultValue))
            : (this.options.length > 0 ? (this.optionLabels[this.options[0]] || String(this.options[0])) : "Select option");
        
    Q(this.selectedDisplay).text(defaultDisplayValue);
        this.hiddenInput.value = defaultValue || (this.options.length > 0 ? this.options[0] : "");
        
        Q(this.selectedDisplay).append(this.arrow);
        this.setupOptions();
        this.setupEventListeners();
        
        Q(dropdownContent).append(this.selectedDisplay, this.optionsContainer);
        Q(this.dropdownWrapper).append(dropdownContent, this.hiddenInput);
        this.isOpen = false;
    }
    
    setupOptions() {
        this.options.forEach(option => {
            const displayLabel = this.optionLabels[option] || String(option);
            const optionElement = Q('<div>', { class: 'dropdown_option', text: displayLabel }).get(0);
            if (option === this.hiddenInput.value) {
                Q(optionElement).addClass("selected");
            }
            Q(optionElement).on("click", () => {
                this.selectOption(option);
            });
            Q(this.optionsContainer).append(optionElement);
        });
    }
    
    selectOption(option) {
        const displayLabel = this.optionLabels[option] || String(option);
    Q(this.selectedDisplay).text(displayLabel);
        Q(this.selectedDisplay).append(this.arrow);
        this.hiddenInput.value = option;
        Q(this.hiddenInput).trigger('change');
        
        Q(this.optionsContainer).find('.dropdown_option').getAll().forEach(opt => {
            Q(opt).removeClass("selected");
            if (Q(opt).text() === option) {
                Q(opt).addClass("selected");
            }
        });
        
        if (this.onChangeCallback) {
            this.onChangeCallback(option);
        }
        
        this.close();
    }
    
    setupEventListeners() {
        Q(this.selectedDisplay).on("click", (e) => {
            e.stopPropagation();
            this.toggle();
        });
        
    Q(document).on("click", () => {
            if (this.isOpen) {
                this.close();
            }
        });
        
        // Handle window resize to recalculate position
    Q(window).on("resize", () => {
            if (this.isOpen) {
                this.calculatePosition();
            }
        });
        
        // Handle scroll events on parent containers
        let scrollableParents = [];
        let element = this.dropdownWrapper.parentElement;
        while (element) {
            if (element.scrollHeight > element.clientHeight || 
                window.getComputedStyle(element).overflow === 'auto' || 
                window.getComputedStyle(element).overflow === 'scroll') {
                scrollableParents.push(element);
            }
            element = element.parentElement;
        }
        
        scrollableParents.forEach(parent => {
            Q(parent).on("scroll", () => {
                if (this.isOpen) {
                    this.calculatePosition();
                }
            });
        });
    }
    
    toggle() {
        if (!this.isOpen) {
            Q('.dropdown_options.show').getAll().forEach(option => {
                Q(option).removeClass('show');
            });
            Q('.dropdown_arrow').getAll().forEach(arrow => {
                Q(arrow).css('transform', 'rotate(0deg)');
            });
            this.open();
        } else {
            this.close();
        }
    }
    
    open() {
        // If for some shitty reason (CSS transitions, external scripts, mutation observers) the option text nodes
        // got cleared, rebuild them from the preserved snapshot before opening.
        const needsRebuild = (() => {
            const domOptions = Q(this.optionsContainer).find('.dropdown_option').getAll();
            if (domOptions.length === 0) return true; // somehow emptied
            // If every option has empty textContent, we also rebuild
            return domOptions.every(el => !el.textContent || el.textContent.trim() === '');
        })();

        if (needsRebuild && this._originalOptionsSnapshot.length > 0) {
            Q(this.optionsContainer).empty();
            // Re-sync this.options with snapshot only if current options empty
            if (!this.options || this.options.length === 0) {
                this.options = [...this._originalOptionsSnapshot];
            }
            this.setupOptions();
        }

        // Calculate optimal position before showing
        this.calculatePosition();

        Q(this.optionsContainer).addClass("show");
        Q(this.arrow).css('transform', 'rotate(180deg)');
        this.isOpen = true;
    }
    
    close() {
    Q(this.optionsContainer).removeClass("show");
    Q(this.arrow).css('transform', 'rotate(0deg)');
        this.isOpen = false;
    }
    
    calculatePosition() {
        // Temporarily show the options container to measure it
    const wasVisible = Q(this.optionsContainer).hasClass('show');
        if (!wasVisible) {
            Q(this.optionsContainer).css('visibility', 'hidden');
            Q(this.optionsContainer).css('display', 'block');
        }
        
        // Reset to default position first
    Q(this.optionsContainer).css('top', '100%');
    Q(this.optionsContainer).css('bottom', 'auto');
        
        // Force layout calculation
        this.optionsContainer.offsetHeight;
        
        const dropdownRect = this.dropdownWrapper.getBoundingClientRect();
        const optionsRect = this.optionsContainer.getBoundingClientRect();
        
        // Calculate available space considering all parent containers
        const availableSpace = this.getAvailableSpace();
        const spaceBelow = availableSpace.bottom - dropdownRect.bottom;
        const spaceAbove = dropdownRect.top - availableSpace.top;
        
        // Hide again if it wasn't visible
        if (!wasVisible) {
            Q(this.optionsContainer).css('visibility', '');
            Q(this.optionsContainer).css('display', '');
        }
        
        // Check if options fit below
        if (spaceBelow >= optionsRect.height) {
            // Enough space below, keep default position
            Q(this.optionsContainer).css('top', '100%');
            Q(this.optionsContainer).css('bottom', 'auto');
        } else if (spaceAbove >= optionsRect.height) {
            // Not enough space below, but enough above
            Q(this.optionsContainer).css('top', 'auto');
            Q(this.optionsContainer).css('bottom', '100%');
        } else {
            // Not enough space either way, position above and add scroll
            Q(this.optionsContainer).css('top', 'auto');
            Q(this.optionsContainer).css('bottom', '100%');
            // Keep max-height for scrolling
        }
    }
    
    getAvailableSpace() {
        let element = this.dropdownWrapper;
        let top = 0;
        let bottom = window.innerHeight;
        let foundConstraint = false;
        
        // Walk up the DOM tree and find all constraining containers
        while (element) {
            const style = window.getComputedStyle(element);
            
            // Check if this element has overflow constraints
            if (style.overflow === 'hidden' || style.overflow === 'auto' || style.overflow === 'scroll' ||
                style.overflowY === 'hidden' || style.overflowY === 'auto' || style.overflowY === 'scroll') {
                
                const rect = element.getBoundingClientRect();
                top = Math.max(top, rect.top);
                bottom = Math.min(bottom, rect.bottom);
                foundConstraint = true;
                // Continue to find more constraints, don't break
            }
            
            element = element.parentElement;
        }
        
        // If no constraining container found, use viewport
        if (!foundConstraint) {
            return { top: 0, bottom: window.innerHeight };
        }
        
        return { top, bottom };
    }
    
    updateOptions(newOptions, newValueMapping = null) {
        // Handle both simple array and object with options/valueMapping
        if (newOptions && typeof newOptions === 'object' && !Array.isArray(newOptions)) {
            // Extract from object format: { options: [...], valueMapping: {...} }
            newValueMapping = newOptions.valueMapping || newValueMapping;
            newOptions = newOptions.options || [];
        }
        
        // Process options: if they're objects with {label, value}, extract label and value
        this.options = [];
        this.optionLabels = {};
        
        if (Array.isArray(newOptions)) {
            newOptions.forEach(opt => {
                if (typeof opt === 'object' && opt !== null && 'value' in opt && 'label' in opt) {
                    // Object format: {label: "...", value: "..."}
                    this.options.push(opt.value);
                    this.optionLabels[opt.value] = typeof opt.label === 'string' && opt.label.startsWith('ui.') 
                        ? lang(opt.label) 
                        : opt.label;
                } else {
                    // Simple string or other format
                    this.options.push(opt);
                    this.optionLabels[opt] = typeof opt === 'string' && opt.startsWith('ui.')
                        ? lang(opt)
                        : String(opt);
                }
            });
        }
        
        this._originalOptionsSnapshot = [...this.options];
        
        // Update valueMapping if provided
        if (newValueMapping && typeof newValueMapping === 'object') {
            this.valueMapping = newValueMapping;
        }
        
        Q(this.optionsContainer).empty();
        this.setupOptions();
        
        // Preserve current value if it's still valid
        const currentValue = this.get();
        if (this.valueMapping) {
            // Check if current mapped value is still available
            const availableValues = Object.values(this.valueMapping);
            if (availableValues.includes(currentValue)) {
                // Keep current selection
                return;
            }
        } else {
            // Simple array case
            if (this.options.includes(this.hiddenInput.value)) {
                // Keep current selection
                return;
            }
        }
        
        // Current value not available, select first option
        if (this.options.length > 0) {
            this.selectOption(this.options[0]);
        }
    }
    
    get() {
        const raw = this.hiddenInput.value;
        if (this.valueMapping && raw in this.valueMapping) {
            return this.valueMapping[raw];
        }
        return raw;
    }
    
    set(value) {
        // Accept either display value or already-mapped internal value
        if (this.options.includes(value)) {
            this.selectOption(value);
            return;
        }
        if (this.valueMapping) {
            // Try to find the display option whose mapped value equals the provided value
            const display = Object.keys(this.valueMapping).find(k => this.valueMapping[k] === value);
            if (display && this.options.includes(display)) {
                this.selectOption(display);
                return;
            }
        }
    }
    
    getElement() {
        return this.dropdownWrapper;
    }

    setOnChange(callback) {
        if (typeof callback === 'function') {
            this.onChangeCallback = callback;
        } else {
            this.onChangeCallback = null;
        }
    }
}
