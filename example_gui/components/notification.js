class NotificationManager {
    constructor() {
        this.container = null;
        this.notifications = [];
        this.totalWords = 0;
        this.timeoutId = null;
        this.init();
    }
    
    init() {
    this.container = Q('<div>', { class: 'notification_container' }).get(0);
        Q(document.body).append(this.container);
    }
    
    show(message, type = 'info') {
        const notification = this.createNotification(message, type);
        this.notifications.push(notification);
        Q(this.container).append(notification.element);
        
        // For error messages, use a minimum display time and add extra time based on length
        const words = message.split(" ").length;
        let adjustedWords = words;
        
        if (type === 'error' || message.toLowerCase().includes('error') || message.toLowerCase().includes('failed')) {
            // Errors get minimum 5 seconds + extra time for long messages
            adjustedWords = Math.max(words, 5);
            adjustedWords += Math.floor(words / 10); // Extra time for very long error messages
        }
        
        this.totalWords += adjustedWords;
        
        this.updateTimeout();
    }
    
    createNotification(message, type = 'info') {
    const element = Q('<div>', { class: 'notification_popup' }).get(0);
        
        // Add type-specific styling
        if (type === 'error' || message.toLowerCase().includes('error') || message.toLowerCase().includes('failed')) {
            Q(element).addClass('notification_error');
        } else if (type === 'success' || message.toLowerCase().includes('success')) {
            Q(element).addClass('notification_success');
        } else if (type === 'warning' || message.toLowerCase().includes('warning')) {
            Q(element).addClass('notification_warning');
        }
        
    const header = Q('<div>', { class: 'notification_header' }).get(0);
        
    const time = Q('<span>', { class: 'notification_time' }).get(0);
    Q(time).text(this.getCurrentTime());
        
    const closeButton = Q('<span>', { class: 'notification_close' }).get(0);
    Q(closeButton).text("×");
        
        Q(header).append(time, closeButton);
        
    const messageElement = Q('<div>', { class: 'notification_message' }).get(0);
    Q(messageElement).text(message);
        
        // For long messages, add scrolling
        if (message.length > 200) {
            Q(messageElement).css('maxHeight', '100px');
            Q(messageElement).css('overflowY', 'auto');
        }
        
        Q(element).append(header, messageElement);
        
        const words = message.split(" ").length;
        let adjustedWords = words;
        
        if (type === 'error' || message.toLowerCase().includes('error') || message.toLowerCase().includes('failed')) {
            adjustedWords = Math.max(words, 5);
        }
        
        const notification = {
            element: element,
            message: message,
            words: adjustedWords,
            type: type
        };
        
        Q(closeButton).on("click", () => {
            this.removeNotification(notification);
        });
        
        return notification;
    }
    
    removeNotification(notification) {
        const index = this.notifications.indexOf(notification);
        if (index !== -1) {
            this.notifications.splice(index, 1);
            this.totalWords -= notification.words;
            
            Q(notification.element).remove();
            
            if (this.notifications.length === 0) {
                this.clearTimeout();
            } else {
                this.updateTimeout();
            }
        }
    }
    
    updateTimeout() {
        this.clearTimeout();
        
        const duration = this.totalWords * 1000;
        
        this.timeoutId = setTimeout(() => {
            this.clearAll();
        }, duration);
    }
    
    clearTimeout() {
        if (this.timeoutId) {
            clearTimeout(this.timeoutId);
            this.timeoutId = null;
        }
    }
    
    clearAll() {
        this.notifications.forEach(notification => {
            Q(notification.element).remove();
        });
        this.notifications = [];
        this.totalWords = 0;
        this.clearTimeout();
    }
    
    getCurrentTime() {
        const now = new Date();
        return now.getHours().toString().padStart(2, "0") + ":" + 
               now.getMinutes().toString().padStart(2, "0") + ":" + 
               now.getSeconds().toString().padStart(2, "0");
    }
}
