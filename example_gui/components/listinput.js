class ListInput {
    constructor(identifier, value = "", title = "", description = "") {
        this.identifier = identifier;
    this.listWrapper = Q('<div>', { class: 'list_wrapper' }).get(0);
    if (title) {
        const heading = Q('<h3>', { class: 'inputs_title', text: title }).get(0);
        this.listWrapper.appendChild(heading);
    }
    if (description) {
        const descriptionHeading = Q('<h4>', { class: 'inputs_description', text: description }).get(0);
        this.listWrapper.appendChild(descriptionHeading);
    }
    this.listContent = Q('<div>', { class: 'list_content' }).get(0);
    this.hiddenInput = Q('<input>', { class: 'hidden_input' }).get(0);
        this.hiddenInput.type = "hidden";
        this.hiddenInput.value = value;
        this.hiddenInput.setAttribute("id", identifier);
        this.listWrapper.append(this.listContent, this.hiddenInput);
        this.setupEventListeners();
        if (value) {
            const tags = value.split(",");
            tags.forEach(tag => {
                if (tag.trim()) this.createTag(tag.trim());
            });
        }
    }
    
    updateHiddenInput() {
    const tags = Array.from(this.listContent.querySelectorAll(".list_tag"));
    const tagValues = tags.map(tag => tag.textContent.slice(0, -1));
    this.hiddenInput.value = tagValues.join(",");
    Q(this.hiddenInput).trigger("change");
    }
    
    createTag(text) {
    if (Array.from(this.listContent.querySelectorAll(".list_tag")).some(tag => tag.textContent.slice(0, -1) === text)) return;
    const tag = Q('<div>', { class: 'list_tag', text }).get(0);
    const closeButton = Q('<div>', { class: 'list_tag_close', text: 'x' }).get(0);
    Q(closeButton).on("click", () => {
            tag.remove();
            this.updateHiddenInput();
        });
        tag.appendChild(closeButton);
        Q(tag).on("click", () => {
            let input = tag.querySelector(".list_input");
            if (!input) {
                input = Q('<input>', { class: 'list_input' }).get(0);
                Q(tag).text("");
                tag.appendChild(input);
            }
            input.value = tag.textContent || text;
            input.focus();
            Q(input).on("blur", () => {
                if (input.value.trim()) {
                    Q(tag).text(input.value.trim());
                    tag.appendChild(closeButton);
                } else {
                    tag.remove();
                }
                this.updateHiddenInput();
            }, { once: true });
            Q(input).on("keydown", (e) => {
                if (e.key === "Enter" || e.key === ",") {
                    e.preventDefault();
                    if (input.value.trim()) {
                        Q(tag).text(input.value.trim());
                        tag.appendChild(closeButton);
                    } else {
                        tag.remove();
                    }
                    this.updateHiddenInput();
                }
            }, { once: true });
        });
    this.listContent.insertBefore(tag, this.listContent.querySelector(".list_input"));
    }
    
    setupEventListeners() {
    Q(this.listContent).on("click", (e) => {
            if (e.target === this.listContent) {
                let input = this.listContent.querySelector(".list_input");
                if (!input) {
                    input = Q('<input>', { class: 'list_input' }).get(0);
                    this.listContent.appendChild(input);
                }
                input.focus();
        Q(input).on("keydown", (e) => {
                    if (e.key === "Enter" || e.key === ",") {
                        e.preventDefault();
                        if (input.value.trim()) this.createTag(input.value.trim());
                        input.value = "";
                        this.updateHiddenInput();
                    }
                });
        Q(document).on("click", (e) => {
                    if (!this.listContent.contains(e.target) && input.parentNode) {
                        input.remove();
                    }
        }, { once: true });
            }
        });
    }
    
    get() {
        return this.hiddenInput.value;
    }
    
    set(value) {
        this.hiddenInput.value = value;
    Q(this.listContent).empty();
        if (value) {
            const tags = value.split(",");
            tags.forEach(tag => {
                if (tag.trim()) this.createTag(tag.trim());
            });
        }
    Q(this.hiddenInput).trigger("change");
    }
    
    getElement() {
        return this.listWrapper;
    }
}
