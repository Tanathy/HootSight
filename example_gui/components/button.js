class Button {
    constructor(identifier, text, text_lock = "", buttonClass = "", callback = null, noLock = false) {
        this.identifier = identifier;
        this.text = text;
        this.text_lock = text_lock;
        this.callback = callback;
        this.noLock = noLock;
        this.locked = false;
        this.buttonElement = Q('<div>', { class: 'button_wrapper', text: text, id: identifier }).get(0);
        if (buttonClass) {
            Q(this.buttonElement).addClass(buttonClass);
        }
        this.setupEventListeners();
    }
    
    setupEventListeners() {
        Q(this.buttonElement).on("click", () => {
            if (this.locked) return;
            if (this.callback && typeof this.callback === "function") {
                if (!this.noLock) {
                    this.lock();
                }
                this.callback();
            }
        });
    }
    
    lock() {
        this.locked = true;
        Q(this.buttonElement).addClass("locked");
        if (this.text_lock) {
            Q(this.buttonElement).text(this.text_lock);
        }
    }
    
    ready() {
    this.locked = false;
    Q(this.buttonElement).removeClass("locked");
    Q(this.buttonElement).text(this.text);
    }
    
    getElement() {
        return this.buttonElement;
    }
}
