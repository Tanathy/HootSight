class Alert {
    constructor() {
        this.overlay = null;
        this.alertBox = null;
    }
    
    show(message, callback = null) {
        if (this.overlay) {
            this.hide();
        }
        
    this.overlay = Q('<div>', { class: 'alert_overlay' }).get(0);
    this.alertBox = Q('<div>', { class: 'alert_box' }).get(0);
        
    const messageElement = Q('<div>', { class: 'alert_message' }).get(0);
        Q(messageElement).text(message);
        
    const buttonContainer = Q('<div>', { class: 'alert_button_container' }).get(0);
        
        const buttonComponent = new Button("alert_ok_button", "OK", "", "", () => {
            this.hide();
            if (callback) callback();
            buttonComponent.ready();
        });
        
        Q(buttonContainer).append(buttonComponent.getElement());
        
        Q(this.alertBox).append(messageElement, buttonContainer);
        Q(this.overlay).append(this.alertBox);
        
        Q(document.body).append(this.overlay);
        
    Q(this.overlay).on("click", (event) => {
            if (event.target === this.overlay) {
                this.hide();
                if (callback) callback();
            }
        });
    }
    
    hide() {
        if (this.overlay) {
            Q(this.overlay).remove();
        }
        this.overlay = null;
        this.alertBox = null;
    }
}
