class Htmlbox {
    constructor(identifier, content = '') {
        this.identifier = identifier;
        this.wrapper = Q('<div>', { class: 'htmlbox_wrapper' }).get(0);
        this.wrapper.id = identifier;
        this.contentEl = Q('<div>', { class: 'htmlbox_content' }).get(0);
        this.wrapper.appendChild(this.contentEl);
        this.set(content || '');
    }

    // Only set is provided; sets inner HTML content
    set(content) {
        // Direct HTML assignment as requested; caller is responsible for content safety
        this.contentEl.innerHTML = content != null ? String(content) : '';
    }

    getElement() {
        return this.wrapper;
    }
}
