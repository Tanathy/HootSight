class Input {
    constructor(identifier, type = "text", value = "", title = "", description = "", placeholder = "", min = "", max = "", step = "") {
        this.identifier = identifier;
        this.type = type;
        this.inputWrapper = Q('<div>', { class: 'input_wrapper' }).get(0);
        if (title) {
            const heading = Q('<h3>', { class: 'inputs_title', text: title }).get(0);
            Q(this.inputWrapper).append(heading);
        }
        if (description) {
            const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: description }).get(0);
            Q(this.inputWrapper).append(descriptionHeading);
        }
        
        const inputContent = Q('<div>', { class: 'input_content' }).get(0);
        const attrs = { class: 'input_field', type: type, value: value, id: identifier };
        if (placeholder) attrs.placeholder = placeholder;
        if (min && (type === "number" || type === "range")) attrs.min = min;
        if (max && (type === "number" || type === "range")) attrs.max = max;
        if (step && (type === "number" || type === "range")) attrs.step = step;
        this.inputField = Q('<input>', attrs).get(0);
        
        this.setupEventListeners();
        
        Q(inputContent).append(this.inputField);
        Q(this.inputWrapper).append(inputContent);
    }
    
    setupEventListeners() {
        Q(this.inputField).on("input", () => {
            Q(this.inputField).trigger('change');
        });
    }
    
    get() {
        if (this.type === "number") {
            const value = parseFloat(this.inputField.value);
            return isNaN(value) ? 0 : value;
        }
        return this.inputField.value;
    }
    
    set(value) {
        this.inputField.value = value;
        Q(this.inputField).trigger('change');
    }
    
    getElement() {
        return this.inputWrapper;
    }
}
