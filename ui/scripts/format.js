/**
 * HootSight - Number Formatting Utilities
 * Scientific notation and smart number formatting
 */

const Format = {
    /**
     * Format a number with smart scientific notation for very small values
     * @param {number} value - The number to format
     * @param {number} [precision=4] - Decimal places for normal display
     * @param {number} [scientificThreshold=0.001] - Below this, use scientific notation
     * @returns {string} Formatted number string
     */
    number: function(value, precision = 4, scientificThreshold = 0.001) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        
        const absValue = Math.abs(value);
        
        // Zero is zero
        if (absValue === 0) {
            return '0';
        }
        
        // Very small numbers -> scientific notation
        if (absValue < scientificThreshold && absValue > 0) {
            // Find the exponent
            const exp = Math.floor(Math.log10(absValue));
            const mantissa = value / Math.pow(10, exp);
            
            // Format mantissa with reasonable precision
            const mantissaStr = mantissa.toFixed(2);
            return `${mantissaStr}e${exp}`;
        }
        
        // Very large numbers -> K/M suffix
        if (absValue >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        }
        if (absValue >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        }
        
        // Normal range
        if (absValue >= 1) {
            return value.toFixed(Math.min(precision, 2));
        }
        
        // Between threshold and 1
        return value.toFixed(precision);
    },

    /**
     * Format loss value (optimized for training loss display)
     * @param {number} value - Loss value
     * @returns {string} Formatted loss string
     */
    loss: function(value) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        
        const absValue = Math.abs(value);
        
        if (absValue === 0) {
            return '0';
        }
        
        // Very small loss -> scientific notation (e.g., 1.85e-4)
        if (absValue < 0.001) {
            const exp = Math.floor(Math.log10(absValue));
            const mantissa = value / Math.pow(10, exp);
            return `${mantissa.toFixed(2)}e${exp}`;
        }
        
        // Small loss -> 4 decimal places
        if (absValue < 0.1) {
            return value.toFixed(4);
        }
        
        // Normal loss -> 3 decimal places
        if (absValue < 10) {
            return value.toFixed(3);
        }
        
        // Large loss -> 2 decimal places
        return value.toFixed(2);
    },

    /**
     * Format learning rate value
     * @param {number} value - Learning rate
     * @returns {string} Formatted LR string
     */
    learningRate: function(value) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        
        const absValue = Math.abs(value);
        
        if (absValue === 0) {
            return '0';
        }
        
        // LR is typically small, use scientific for < 0.01
        if (absValue < 0.0001) {
            const exp = Math.floor(Math.log10(absValue));
            const mantissa = value / Math.pow(10, exp);
            return `${mantissa.toFixed(2)}e${exp}`;
        }
        
        if (absValue < 0.001) {
            return value.toFixed(6);
        }
        
        if (absValue < 0.01) {
            return value.toFixed(5);
        }
        
        return value.toFixed(4);
    },

    /**
     * Format percentage value
     * @param {number} value - Percentage (0-100)
     * @param {number} [precision=2] - Decimal places
     * @returns {string} Formatted percentage string
     */
    percent: function(value, precision = 2) {
        if (value === null || value === undefined || isNaN(value)) {
            return 'N/A';
        }
        return value.toFixed(precision) + '%';
    },

    /**
     * Format value for graph Y-axis labels
     * @param {number} value - The value to format
     * @returns {string} Formatted string suitable for axis labels
     */
    graphAxis: function(value) {
        if (value === null || value === undefined || isNaN(value)) {
            return '';
        }
        
        const absValue = Math.abs(value);
        
        if (absValue === 0) {
            return '0';
        }
        
        // Very small -> compact scientific
        if (absValue < 0.001 && absValue > 0) {
            const exp = Math.floor(Math.log10(absValue));
            const mantissa = value / Math.pow(10, exp);
            return `${mantissa.toFixed(1)}e${exp}`;
        }
        
        // Large numbers
        if (absValue >= 1000000) {
            return (value / 1000000).toFixed(1) + 'M';
        }
        if (absValue >= 1000) {
            return (value / 1000).toFixed(1) + 'K';
        }
        
        // Between 0.001 and 1
        if (absValue < 1) {
            // Show meaningful precision
            if (absValue < 0.01) {
                return value.toFixed(3);
            }
            return value.toFixed(2);
        }
        
        // Normal range 1-1000
        if (absValue < 10) {
            return value.toFixed(2);
        }
        if (absValue < 100) {
            return value.toFixed(1);
        }
        
        return Math.round(value).toString();
    }
};

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = Format;
}
