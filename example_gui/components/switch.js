class Switch {
    constructor(identifier, value = false, title = "", description = "") {
        this.identifier = identifier;
    this.switchWrapper = Q('<div>', { class: 'switch_wrapper' }).get(0);
            if (title) {
                const heading = Q('<h3>', { class: 'inputs_title', text: title }).get(0);
                this.switchWrapper.appendChild(heading);
            }
    const switchContent = Q('<div>', { class: 'switch_content' }).get(0);
    this.hiddenCheckbox = Q('<input>', { class: 'hidden_checkbox' }).get(0);
        this.hiddenCheckbox.type = "checkbox";
        this.hiddenCheckbox.checked = value;
        this.hiddenCheckbox.setAttribute("id", identifier);
    const switchLabel = Q('<label>', { class: 'switch_label' }).get(0);
        switchLabel.setAttribute("for", identifier);
            const switchDescription = Q('<h4>', { class: 'inputs_description', text: description }).get(0);
        switchContent.append(this.hiddenCheckbox, switchLabel, switchDescription);
        this.switchWrapper.appendChild(switchContent);
    }
    
    get() {
        return this.hiddenCheckbox.checked;
    }
    
    set(value) {
        this.hiddenCheckbox.checked = value;
    Q(this.hiddenCheckbox).trigger("change");
    }
    
    getElement() {
        return this.switchWrapper;
    }
}
