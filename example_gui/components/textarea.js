class Textarea {
    constructor(identifier, value = "", title = "", description = "", placeholder = "", rows = 4, cols = 50) {
        this.identifier = identifier;
        this.textareaWrapper = Q('<div>', { class: 'textarea_wrapper' }).get(0);
        if (title) {
            const heading = Q('<h3>', { class: 'inputs_title', text: title }).get(0);
            this.textareaWrapper.appendChild(heading);
        }
        if (description) {
            const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: description }).get(0);
            this.textareaWrapper.appendChild(descriptionHeading);
        }
        
        const textareaContent = Q('<div>', { class: 'textarea_content' }).get(0);
        const attrs = { class: 'textarea_field', value: value, id: identifier };
        if (placeholder) attrs.placeholder = placeholder;
        if (rows) attrs.rows = rows;
        if (cols) attrs.cols = cols;
        this.textareaField = Q('<textarea>', attrs).get(0);
        
        this.setupEventListeners();
        
        textareaContent.appendChild(this.textareaField);
        this.textareaWrapper.appendChild(textareaContent);
    }
    
    setupEventListeners() {
        Q(this.textareaField).on("input", () => {
            Q(this.textareaField).trigger('change');
        });
    }
    
    get() {
        return this.textareaField.value;
    }
    
    set(value) {
        this.textareaField.value = value;
        Q(this.textareaField).trigger('change');
    }
    
    getElement() {
        return this.textareaWrapper;
    }
}
