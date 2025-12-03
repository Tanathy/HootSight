/**
 * HootSight - Fuzzy Search Utility
 * Provides Levenshtein distance-based fuzzy search functionality
 */

const FuzzySearch = {
    /**
     * Calculate Levenshtein distance between two strings
     * @param {string} a - First string
     * @param {string} b - Second string
     * @returns {number} - Edit distance
     */
    levenshtein: function(a, b) {
        if (!a || !b) return (a || b).length;
        
        a = a.toLowerCase();
        b = b.toLowerCase();
        
        if (a === b) return 0;
        if (a.length === 0) return b.length;
        if (b.length === 0) return a.length;

        const matrix = [];

        // Initialize first column
        for (let i = 0; i <= b.length; i++) {
            matrix[i] = [i];
        }

        // Initialize first row
        for (let j = 0; j <= a.length; j++) {
            matrix[0][j] = j;
        }

        // Fill in the rest of the matrix
        for (let i = 1; i <= b.length; i++) {
            for (let j = 1; j <= a.length; j++) {
                if (b.charAt(i - 1) === a.charAt(j - 1)) {
                    matrix[i][j] = matrix[i - 1][j - 1];
                } else {
                    matrix[i][j] = Math.min(
                        matrix[i - 1][j - 1] + 1, // substitution
                        matrix[i][j - 1] + 1,     // insertion
                        matrix[i - 1][j] + 1      // deletion
                    );
                }
            }
        }

        return matrix[b.length][a.length];
    },

    /**
     * Calculate similarity ratio (0-1) based on Levenshtein distance
     * @param {string} a - First string
     * @param {string} b - Second string
     * @returns {number} - Similarity ratio (1 = identical, 0 = completely different)
     */
    similarity: function(a, b) {
        if (!a && !b) return 1;
        if (!a || !b) return 0;
        
        const maxLen = Math.max(a.length, b.length);
        if (maxLen === 0) return 1;
        
        const distance = this.levenshtein(a, b);
        return 1 - (distance / maxLen);
    },

    /**
     * Check if query matches text (includes exact substring match + fuzzy)
     * @param {string} text - Text to search in
     * @param {string} query - Search query
     * @param {number} threshold - Minimum similarity threshold (0-1)
     * @returns {object} - {matches: boolean, score: number}
     */
    matches: function(text, query, threshold = 0.3) {
        if (!query || query.trim() === '') {
            return { matches: true, score: 1 };
        }
        if (!text) {
            return { matches: false, score: 0 };
        }

        text = text.toLowerCase();
        query = query.toLowerCase().trim();

        // Exact substring match - highest priority
        if (text.includes(query)) {
            return { matches: true, score: 1 };
        }

        // Word-by-word matching
        const queryWords = query.split(/\s+/);
        const textWords = text.split(/\s+/);
        
        let totalScore = 0;
        let matchedWords = 0;

        for (const queryWord of queryWords) {
            if (queryWord.length < 2) continue;
            
            let bestWordScore = 0;
            
            // Check each word in text
            for (const textWord of textWords) {
                // Exact word match
                if (textWord.includes(queryWord)) {
                    bestWordScore = 1;
                    break;
                }
                
                // Fuzzy match for longer words
                if (queryWord.length >= 3) {
                    const sim = this.similarity(queryWord, textWord);
                    if (sim > bestWordScore) {
                        bestWordScore = sim;
                    }
                }
            }
            
            totalScore += bestWordScore;
            if (bestWordScore >= threshold) {
                matchedWords++;
            }
        }

        const avgScore = queryWords.length > 0 ? totalScore / queryWords.length : 0;
        const wordMatchRatio = queryWords.length > 0 ? matchedWords / queryWords.length : 0;
        
        // Combined score
        const finalScore = (avgScore * 0.6) + (wordMatchRatio * 0.4);
        
        return {
            matches: finalScore >= threshold,
            score: finalScore
        };
    },

    /**
     * Search through an array of items
     * @param {Array} items - Items to search
     * @param {string} query - Search query
     * @param {Array<string>} fields - Field names to search in
     * @param {number} threshold - Minimum similarity threshold
     * @returns {Array} - Filtered and sorted items with scores
     */
    search: function(items, query, fields, threshold = 0.3) {
        if (!query || query.trim() === '') {
            return items.map(item => ({ item, score: 1 }));
        }

        const results = [];

        for (const item of items) {
            let bestScore = 0;

            for (const field of fields) {
                const value = this._getNestedValue(item, field);
                if (value) {
                    const result = this.matches(String(value), query, threshold);
                    if (result.score > bestScore) {
                        bestScore = result.score;
                    }
                }
            }

            if (bestScore >= threshold) {
                results.push({ item, score: bestScore });
            }
        }

        // Sort by score descending
        results.sort((a, b) => b.score - a.score);

        return results;
    },

    /**
     * Get nested object value by dot notation
     * @param {object} obj - Object to search
     * @param {string} path - Dot-notation path
     * @returns {*} - Value at path
     */
    _getNestedValue: function(obj, path) {
        return path.split('.').reduce((acc, part) => acc && acc[part], obj);
    }
};
