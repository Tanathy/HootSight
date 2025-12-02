;(function () {
    'use strict';

    // ============================================================================
    // CONSTANTS & CONFIGURATION
    // ============================================================================

    const DEFAULT_SIZE_PRESETS = {
        Generic: [128, 256, 512, 768, 1024],
        YOLO: [416, 512, 640, 960],
        ResNet: [224, 256, 384, 448, 512],
    };

    const PAGE_SIZES = [25, 50, 100, 200, 500, 750];
    
    // Timing constants (all in milliseconds for clarity)
    const TIMING = {
        ANNOTATION_SAVE_DELAY: 600,      // Debounce interval for annotation autosave
        BOUNDS_SAVE_DELAY: 400,          // Debounce interval for crop bound changes
        SUGGESTION_HIDE_DELAY: 150,      // Delay before hiding autocomplete panel
        TOAST_DURATION: 2500,            // How long to show toast notifications
        POLLING_INTERVAL: 1000,          // Interval for build/discovery status polling
        API_TIMEOUT_SHORT: 8000,         // Short requests (status checks)
        API_TIMEOUT_LONG: 30000,         // Long-running requests (bulk operations)
    };

    // Validation constraints
    const CONSTRAINTS = {
        MAX_SEARCH_LENGTH: 500,
        MAX_TAG_LENGTH: 100,
        MAX_ANNOTATION_LENGTH: 10000,
        MAX_BULK_TAGS: 50,
        MIN_VALID_SIZE: 32,
        MAX_VALID_SIZE: 4096,
        MAX_ZOOM: 20,
        MIN_ZOOM: 1,
        MAX_FETCH_RETRIES: 3,
        RETRY_BACKOFF_MS: 500,           // Base delay for exponential backoff
    };

    // State default/initialization
    const state = {
        projects: [],
        selectedProject: null,
        selectedFolder: "",
        items: [],
        itemOrder: [],
        page: 1,
        pageSize: 50,
        totalPages: 0,
        totalItems: 0,
        search: "",
        stats: null,
        tagHints: [],
        folderTree: null,
        ctrlPressed: false,
        selectedItems: new Set(),
        selectionAnchor: null,
        deletionInFlight: false,
        buildStatus: null,
        discoveryStatus: null,
        buildSizePresets: DEFAULT_SIZE_PRESETS,
        buildDefaultSize: 512,
        selectedBuildSize: null,
        activeBuildSize: null,
        buildSizeDirty: false,
    };

    // ============================================================================
    // UTILITIES: LOGGING, VALIDATION, HELPERS
    // ============================================================================

    // Centralized logger - normalize interface so warn/info/etc always exist
    const LOG = createLogger();

    function createLogger() {
        const call = (method, fallbackMethod, args) => {
            const hs = window.Hootsight || {};
            const target = hs && hs.log;

            if (target && typeof target[method] === 'function') {
                target[method](...args);
                return;
            }

            if (typeof target === 'function') {
                try {
                    target(...args);
                    return;
                } catch (err) {
                    (window.console && window.console.error ? window.console.error : Function.prototype)(err);
                }
            }

            const c = window.console || {};
            const fn = c[method] || c[fallbackMethod] || c.log;
            if (typeof fn === 'function') {
                fn.apply(c, args);
            }
        };

        return {
            debug: (...args) => call('debug', 'log', args),
            info: (...args) => call('info', 'log', args),
            warn: (...args) => call('warn', 'error', args),
            error: (...args) => call('error', 'log', args),
        };
    }

    function locale(key, fallback = '') {
        const hs = window.Hootsight || {};
        const text = hs.text;

        if (text && typeof text.t === 'function') {
            try {
                const localized = text.t(key);
                if (localized && localized !== key) {
                    return localized;
                }
                if (localized && !fallback) {
                    return localized;
                }
            } catch (err) {
                LOG.warn('Localization lookup failed', err);
            }
        }

        if (typeof window.locale === 'function') {
            try {
                const legacyValue = window.locale(key, fallback);
                if (legacyValue) {
                    return legacyValue;
                }
            } catch (err) {
                LOG.warn('Legacy locale lookup failed', err);
            }
        }

        return fallback || key;
    }

    /**
     * Validate search input: length, type, content constraints.
     * Returns { valid: boolean, sanitized: string, error?: string }
     */
    function validateSearch(value) {
        if (typeof value !== 'string') return { valid: false, sanitized: '', error: 'Search must be a string' };
        const trimmed = value.trim();
        if (trimmed.length > CONSTRAINTS.MAX_SEARCH_LENGTH) {
            return { valid: false, sanitized: '', error: `Search exceeds ${CONSTRAINTS.MAX_SEARCH_LENGTH} characters` };
        }
        // Reject null bytes, control characters
        if (/[\x00-\x08\x0B-\x0C\x0E-\x1F]/.test(trimmed)) {
            return { valid: false, sanitized: '', error: 'Search contains invalid characters' };
        }
        return { valid: true, sanitized: trimmed };
    }

    /**
     * Validate a tag: length, format, content constraints.
     * Returns { valid: boolean, sanitized: string, error?: string }
     */
    function validateTag(value) {
        if (typeof value !== 'string') return { valid: false, sanitized: '', error: 'Tag must be a string' };
        const trimmed = value.trim().toLowerCase();
        if (!trimmed) return { valid: false, sanitized: '', error: 'Tag cannot be empty' };
        if (trimmed.length > CONSTRAINTS.MAX_TAG_LENGTH) {
            return { valid: false, sanitized: '', error: `Tag exceeds ${CONSTRAINTS.MAX_TAG_LENGTH} characters` };
        }
        // Tags: only alphanumeric, underscore, hyphen, space
        if (!/^[a-z0-9_\- ]+$/.test(trimmed)) {
            return { valid: false, sanitized: '', error: 'Tag contains invalid characters' };
        }
        return { valid: true, sanitized: trimmed };
    }

    /**
     * Validate annotation: length, content constraints.
     * Returns { valid: boolean, sanitized: string, error?: string }
     */
    function validateAnnotation(value) {
        if (typeof value !== 'string') return { valid: false, sanitized: '', error: 'Annotation must be a string' };
        if (value.length > CONSTRAINTS.MAX_ANNOTATION_LENGTH) {
            return { valid: false, sanitized: '', error: `Annotation exceeds ${CONSTRAINTS.MAX_ANNOTATION_LENGTH} characters` };
        }
        // Reject null bytes
        if (/\x00/.test(value)) {
            return { valid: false, sanitized: '', error: 'Annotation contains invalid characters' };
        }
        return { valid: true, sanitized: value };
    }

    /**
     * Validate image dimension: within acceptable ranges.
     * Returns { valid: boolean, error?: string }
     */
    function validateImageDimension(size) {
        if (!Number.isFinite(size)) return { valid: false, error: 'Dimension must be a finite number' };
        if (size < CONSTRAINTS.MIN_VALID_SIZE || size > CONSTRAINTS.MAX_VALID_SIZE) {
            return { valid: false, error: `Size must be between ${CONSTRAINTS.MIN_VALID_SIZE} and ${CONSTRAINTS.MAX_VALID_SIZE}` };
        }
        return { valid: true };
    }

    /**
     * Validate project name exists and is selected in state.
     * Returns { valid: boolean, error?: string }
     */
    function validateProjectSelected() {
        if (!state.selectedProject || typeof state.selectedProject !== 'string') {
            return { valid: false, error: 'No project selected' };
        }
        return { valid: true };
    }

    /**
     * Parse comma/newline-separated tag input with validation.
     * Returns array of validated tags, or empty array if invalid.
     */
    function parseTagInput(value) {
        if (typeof value !== 'string') return [];
        return value
            .split(/[,\n]/)
            .map((tag) => {
                const validated = validateTag(tag);
                return validated.valid ? validated.sanitized : null;
            })
            .filter(Boolean);
    }

    /**
     * Small helper - check whether a Q() wrapper contains at least one element
     */
    function exists(qEl) {
        return Boolean(qEl && typeof qEl.get === 'function' && qEl.get(0));
    }

function formatLocale(key, fallback, replacements = {}) {
    let template = locale(key, fallback);
    Object.entries(replacements).forEach(([name, value]) => {
        template = template.split(`{${name}}`).join(String(value));
    });
    return template;
}

function formatLocaleWithNewlines(key, fallback, replacements = {}) {
    // First get the text with replacements
    let template = locale(key, fallback);
    Object.entries(replacements).forEach(([name, value]) => {
        template = template.split(`{${name}}`).join(String(value));
    });
    // Then split by newline and filter empty lines
    return template.split('\n').filter(line => line.trim());
}

function joinWithBullet(parts) {
    const separator = locale("bullet_separator", " \u2022 ");
    return parts.filter(Boolean).join(separator);
}

function joinWithComma(parts) {
    const separator = locale("list_separator", ", ");
    return parts.filter(Boolean).join(separator);
}

function formatCount(singleKey, pluralKey, count, singleFallback, pluralFallback) {
    if (count === 1) {
        return formatLocale(singleKey, singleFallback, { count });
    }
    return formatLocale(pluralKey, pluralFallback, { count });
}

// Use Q wrappers from qte.js for DOM interactions. These wrappers are created
// early (may be empty) and will be replaced with actual elements after the
// DOM is built by DatasetEditorDOM.build(). Consuming code should use Q API
// methods (val(), on(), text(), html(), etc.) and, when needed, call .get(0)
// to access the underlying DOM node.
    const els = {
    projectPicker: Q('#projectPicker'),
    refreshProject: Q('#refreshProject'),
    buildButton: Q('#buildDataset'),
    datasetSize: Q('#datasetSize'),
    searchInput: Q('#searchInput'),
    searchClear: Q('#searchClear'),
    pageSize: Q('#pageSize'),
    prevPage: Q('#prevPage'),
    nextPage: Q('#nextPage'),
    pageIndicator: Q('#pageIndicator'),
    grid: Q('#itemGrid'),
    // Legacy numeric stats elements removed; only recommendations remain
    statsRecommendations: Q('#statRecommendations'),
    statRecommendationsBody: Q('#statRecommendationsBody'),
    folderTree: Q('#folderTree'),
    folderRefresh: Q('#folderRefresh'),
    folderReset: Q('#folderReset'),
    bulkAddInput: Q('#bulkAdd'),
    bulkRemoveInput: Q('#bulkRemove'),
    bulkApply: Q('#bulkApply'),
    bulkScope: Q('#bulkScope'),
    toast: Q('#toast'),
    buildProgress: Q('#buildProgress'),
    buildStatusLabel: Q('#buildStatusLabel'),
    buildEta: Q('#buildEta'),
    buildProgressFill: Q('#buildProgressFill'),
    buildProgressStats: Q('#buildProgressStats'),
    discoveryProgress: Q('#discoveryProgress'),
    discoveryStatusLabel: Q('#discoveryStatusLabel'),
    discoveryEta: Q('#discoveryEta'),
    discoveryProgressFill: Q('#discoveryProgressFill'),
    discoveryProgressStats: Q('#discoveryProgressStats'),
    };

    const thumbnailRegistry = new Map();
    const handlerRegistry = {
        global: [] // { target, event, handler, options }
    };
    
    // Request deduplication cache: prevents duplicate concurrent API calls
    // Key: `${method}:${url}:${bodyHash}`, Value: pending Promise
    const pendingRequests = new Map();
const pendingBoundsSaves = new Map();
const pendingAnnotationSaves = new Map();
const autocompleteRegistry = new Map();
const cssEscape = (value) => (window.CSS && window.CSS.escape ? window.CSS.escape(value) : value.replace(/([^a-zA-Z0-9_-])/g, "\\$1"));
let buildPollHandle = null;
let buildPollProject = null;
let lastBuildState = "idle";
let discoveryPollHandle = null;
let discoveryPollProject = null;
let lastDiscoveryState = "idle";
let datasetEditorInitialized = false;
let datasetEditorActive = false;

/**
 * Initialize the dataset editor: build the DOM, attach UI, and load data.
 * This function is idempotent and safe to call multiple times.
 */
function init() {
    // Build the dataset editor DOM structure from JavaScript
    const editorContainer = document.getElementById("dataset-editor-root");
    if (!editorContainer) {
        LOG.error("Dataset editor container not found");
        return;
    }
    if (!window.DatasetEditorDOM) {
        LOG.error("DOM builder not found");
        return;
    }

    
    try {
        const builtEls = window.DatasetEditorDOM.build(editorContainer);
        // Merge the built elements into our els object and convert to Q wrappers
        if (builtEls) {
            Object.keys(builtEls).forEach((key) => {
                try {
                    els[key] = Q(builtEls[key]);
                } catch (ex) {
                    // Fallback: leave as Q() empty wrapper
                    els[key] = Q();
                }
            });
            // Verify critical elements were added
            if (!exists(els.projectPicker)) {
                LOG.error("Critical elements missing after DOM build");
                return;
            }
        } else {
            LOG.error("DOM builder returned no elements");
            return;
        }
    } catch (err) {
        LOG.error("Failed to build dataset editor DOM:", err);
        return;
    }

    bindUI();
    populatePageSizes();
    renderDatasetSizeOptions();
    // Convert any selects within the editor to custom dropdowns (syncs if already converted)
    if (window.Hootsight && window.Hootsight.components && typeof window.Hootsight.components.createCustomDropdownFromSelect === 'function') {
        try {
            const selects = Q(editorContainer).find('select').getAll();
            selects.forEach((s) => {
                try { window.Hootsight.components.createCustomDropdownFromSelect(s); } catch (e) { }
            });
        } catch (ex) { }
    }
        // Dataset editor initialized, proceed to load projects
    loadProjects().catch(reportError);
}

    // Register a global event so we can cleanup on destroy
    function registerGlobalEvent(target, event, handler, options = {}) {
        target.addEventListener(event, handler, options);
        handlerRegistry.global.push({ target, event, handler, options });
    }

    function removeAllGlobalEvents() {
        handlerRegistry.global.forEach(({ target, event, handler, options }) => {
            target.removeEventListener(event, handler, options);
        });
        handlerRegistry.global = [];
    }

/**
 * Attach event handlers to editor controls and global keyboard/window events.
 */
function bindUI() {
    if (!exists(els.projectPicker)) {
    LOG.error("Critical elements not initialized - projectPicker is missing");
        return;
    }
    els.projectPicker.on('change', () => {
        state.selectedProject = els.projectPicker.val();
        state.page = 1;
        state.selectedFolder = "";
        state.folderTree = null;
        state.buildSizePresets = DEFAULT_SIZE_PRESETS;
        state.buildDefaultSize = 512;
        state.selectedBuildSize = null;
        state.activeBuildSize = null;
        state.buildSizeDirty = false;
        renderDatasetSizeOptions();
        stopBuildPolling();
        state.buildStatus = null;
        lastBuildState = "idle";
        stopDiscoveryPolling();
        state.discoveryStatus = null;
        lastDiscoveryState = "idle";
        refreshData();
    });

    if (exists(els.refreshProject)) els.refreshProject.on('click', () => refreshProject().catch(reportError));
    if (exists(els.buildButton)) els.buildButton.on('click', () => runBuild().catch(reportError));
    if (els.datasetSize) {
        if (exists(els.datasetSize)) els.datasetSize.on('change', () => {
            const value = Number(els.datasetSize.val());
            if (Number.isFinite(value)) {
                state.selectedBuildSize = value;
                state.buildSizeDirty = true;
            }
        });
    }
    if (exists(els.folderRefresh)) {
        if (exists(els.folderRefresh)) els.folderRefresh.on('click', () => refreshFoldersManual().catch(reportError));
    }

    if (exists(els.folderReset)) els.folderReset.on('click', () => {
        if (!state.selectedFolder) return;
        state.selectedFolder = "";
        state.page = 1;
        renderFolderTree();
        fetchItems();
    });

    if (exists(els.bulkApply)) els.bulkApply.on('click', () => applyBulkTags().catch(reportError));

    const debouncedSearch = Q.debounce(() => {
        const rawSearch = els.searchInput.val() || "";
        const validated = validateSearch(rawSearch);
        if (!validated.valid) {
            LOG.warn(`Invalid search input: ${validated.error}`);
            notify(validated.error || "Invalid search");
            els.searchInput.val('');
            state.search = "";
        } else {
            state.search = validated.sanitized;
        }
        state.page = 1;
        fetchItems();
    }, 400);
    if (exists(els.searchInput)) els.searchInput.on('input', debouncedSearch);
    if (exists(els.searchClear)) els.searchClear.on('click', () => {
        if (exists(els.searchInput)) els.searchInput.val('');
        state.search = "";
        state.page = 1;
        fetchItems();
    });

    if (exists(els.pageSize)) els.pageSize.on('change', () => {
        state.pageSize = Number(els.pageSize.val());
        state.page = 1;
        fetchItems();
    });

    if (exists(els.prevPage)) els.prevPage.on('click', () => {
        if (state.page > 1) {
            state.page -= 1;
            fetchItems();
        }
    });

    if (exists(els.nextPage)) els.nextPage.on('click', () => {
        if (state.page < state.totalPages) {
            state.page += 1;
            fetchItems();
        }
    });

    const docKeyDown = (event) => {
        if (!datasetEditorActive) return;
        if (event.key === "Control" && !state.ctrlPressed) {
            state.ctrlPressed = true;
            Q(document.body).addClass("crop-enabled");
        }
        handleGlobalKeyDown(event);
    };
    registerGlobalEvent(document, 'keydown', docKeyDown);

    const docKeyUp = (event) => {
        if (!datasetEditorActive) return;
        if (event.key === "Control") {
            state.ctrlPressed = false;
            Q(document.body).removeClass("crop-enabled");
        }
    };
    registerGlobalEvent(document, 'keyup', docKeyUp);

    const windowBlur = () => {
        if (!datasetEditorActive) return;
    state.ctrlPressed = false;
    Q(document.body).removeClass("crop-enabled");
    };
    registerGlobalEvent(window, 'blur', windowBlur);

    const windowResize = () => {
        if (!datasetEditorActive) return;
        thumbnailRegistry.forEach((entry) => applyCrop(entry));
    };
    registerGlobalEvent(window, 'resize', windowResize);
}

function populatePageSizes() {
    PAGE_SIZES.forEach((size) => {
        const option = Q(`<option>`, { value: size, text: `${size}` });
        if (size === state.pageSize) {
            option.prop('selected', true);
        }
        if (exists(els.pageSize)) els.pageSize.append(option);
    });
    // Convert to custom dropdown if helper is available (will sync if already converted)
    if (window.Hootsight && window.Hootsight.components && typeof window.Hootsight.components.createCustomDropdownFromSelect === 'function') {
        try { window.Hootsight.components.createCustomDropdownFromSelect(els.pageSize.get(0)); } catch (e) { }
    }
}

function renderDatasetSizeOptions() {
    if (!exists(els.datasetSize)) return;
    const presets = state.buildSizePresets || {};
    const entries = Object.entries(presets).filter(([, values]) => Array.isArray(values) && values.length);
    if (exists(els.datasetSize)) els.datasetSize.empty();
    if (!entries.length) {
        const option = Q('<option>', { value: '', text: locale('size_no_presets', 'No presets') });
        els.datasetSize.append(option);
        els.datasetSize.prop('disabled', true);
        return;
    }
    entries.forEach(([label, values]) => {
        const optgroup = Q('<optgroup>', { label });
        values.forEach((size) => {
            const option = Q('<option>', { value: size, text: `${size}px` });
            optgroup.append(option);
        });
        if (exists(els.datasetSize)) els.datasetSize.append(optgroup);
    });
    const target = currentBuildSize();
    if (target && exists(els.datasetSize)) {
        els.datasetSize.val(String(target));
    }
    if (exists(els.datasetSize)) els.datasetSize.prop('disabled', false);
    if (window.Hootsight && window.Hootsight.components && typeof window.Hootsight.components.createCustomDropdownFromSelect === 'function') {
        try { window.Hootsight.components.createCustomDropdownFromSelect(els.datasetSize.get(0)); } catch (e) { }
    }
}

function currentBuildSize() {
    const size = state.selectedBuildSize || state.activeBuildSize || state.buildDefaultSize || 512;
    const check = validateImageDimension(size);
    if (!check.valid) {
        LOG.warn(`Invalid build size ${size}: ${check.error}`);
        return state.buildDefaultSize;
    }
    return size;
}

async function loadProjects() {
    const projects = await apiFetch("/dataset/editor/projects");
    state.projects = projects;
    if (exists(els.projectPicker)) els.projectPicker.empty();
    projects.forEach((project) => {
        const imageCount = Number.isFinite(project.image_count) ? project.image_count : 0;
        const option = Q('<option>', { value: project.name, text: `${project.name} (${imageCount})` });
    if (exists(els.projectPicker)) els.projectPicker.append(option);
    });

    if (!state.selectedProject && projects.length) {
        state.selectedProject = projects[0].name;
    }

    if (state.selectedProject && exists(els.projectPicker)) {
        els.projectPicker.val(state.selectedProject);
        refreshData();
    }
    // Convert to custom dropdown while keeping native select for event wiring
    if (window.Hootsight && window.Hootsight.components && typeof window.Hootsight.components.createCustomDropdownFromSelect === 'function') {
        try { window.Hootsight.components.createCustomDropdownFromSelect(els.projectPicker.get(0)); } catch (e) { }
    }
}

function refreshData() {
    fetchItems();
    fetchStats();
    fetchFolders();
    fetchBuildOptions();
    fetchDiscoveryStatus();
    fetchBuildStatus();
}

async function fetchItems() {
    if (!state.selectedProject) return;
    const params = new URLSearchParams({
        page: state.page,
        page_size: state.pageSize,
    });
    if (state.search) params.set("search", state.search);
    if (state.selectedFolder) params.set("folder", state.selectedFolder);

    const data = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/items?${params}`);
    state.items = data.items;
    state.itemOrder = state.items.map((item) => item.id);
    state.page = data.page;
    state.totalPages = data.total_pages;
    state.totalItems = data.total_items;
    syncSelectionWithItems();
    renderItems();
    updatePager();
    updateBulkScope();
}

function buildRecommendationDetailsSection(rec, sectionType, sectionTitle, items, itemKey) {
    if (!items || items.length === 0) return null;

    const section = Q('<div>', { class: 'recommendation-section' });
    const title = Q('<div>', {
        class: 'recommendation-section-title',
        text: locale(`rec_${rec.type}_${sectionType}`, sectionTitle)
    });
    section.append(title);

    items.forEach(item => {
        const row = Q('<div>', {
            class: 'recommendation-label-row',
            text: itemKey === 'rare_tags'
                ? `${item.tag}: ${item.count} occurrences`
                : formatLocale('rec_label_item', '{label}: {count} images', item)
        });
        section.append(row);
    });

    return section;
}

function buildRecommendationCard(rec) {
    const card = Q('<div>', { class: 'card recommendation-card card--compact' });

    const title = Q('<div>', {
        class: 'recommendation-title',
        text: locale(`rec_${rec.type}_title`, rec.type)
    });
    card.append(title);

    const desc = Q('<div>', {
        class: 'recommendation-desc',
        text: formatLocale(`rec_${rec.type}_desc`, '', rec.data || {})
    });
    card.append(desc);

    if (rec.data && Object.keys(rec.data).length > 0) {
        const details = Q('<div>', { class: 'recommendation-details' });

        const underSection = buildRecommendationDetailsSection(rec, 'underrep', 'Underrepresented:', rec.data.underrepresented, 'underrep');
        if (underSection) details.append(underSection);

        const overSection = buildRecommendationDetailsSection(rec, 'overrep', 'Overrepresented:', rec.data.overrepresented, 'overrep');
        if (overSection) details.append(overSection);

        const rareSection = buildRecommendationDetailsSection(rec, 'rare', 'Rarest tags:', rec.data.rare_tags, 'rare_tags');
        if (rareSection) details.append(rareSection);

        card.append(details);
    }

    const actionLines = formatLocaleWithNewlines(`rec_${rec.type}_action`, '', rec.data || {});
    if (actionLines && actionLines.length > 0) {
        const action = Q('<div>', { class: 'recommendation-action' });
        actionLines.forEach((line, idx) => {
            const p = Q('<div>', { text: line.trim() });
            if (idx > 0) p.get(0).style.marginTop = '6px';
            action.append(p);
        });
        card.append(action);
    }

    return card;
}

function renderRecommendations(element, recommendations) {
    if (!element || !element.get(0)) return;
    element.empty();

    if (!recommendations || recommendations.length === 0) {
        const emptyMsg = Q('<div>', {
            class: 'recommendation-empty',
            text: locale('stats_balanced', 'Dataset is well balanced. No immediate action required.')
        });
        element.append(emptyMsg);
        return;
    }

    recommendations.forEach((rec) => {
        const card = buildRecommendationCard(rec);
        element.append(card);
    });
}

async function fetchStats() {
    if (!state.selectedProject) return;

    const stats = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/stats`);
    state.stats = stats;
    state.tagHints = Object.entries(stats.tag_frequencies || {})
        .sort((a, b) => b[1] - a[1])
        .map(([tag]) => tag);
    // Only render recommendations; numeric stats panel has been removed
    renderRecommendations(els.statRecommendationsBody, stats.recommendations);
}

async function fetchFolders() {
    if (!state.selectedProject) return;
    const tree = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/folders`);
    state.folderTree = tree;
    if (state.selectedFolder && !folderContains(tree, state.selectedFolder)) {
        state.selectedFolder = "";
    }
    renderFolderTree();
}

async function fetchBuildOptions() {
    if (!state.selectedProject || !els.datasetSize) return;
    try {
    if (exists(els.datasetSize)) els.datasetSize.prop('disabled', true);
        const options = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/build/options`);
        state.buildSizePresets = options?.presets || DEFAULT_SIZE_PRESETS;
        state.buildDefaultSize = options?.default_size || state.buildDefaultSize || 512;
        state.activeBuildSize = options?.active_size || state.activeBuildSize || state.buildDefaultSize;
        if (!state.buildSizeDirty) {
            state.selectedBuildSize = state.activeBuildSize;
        }
        renderDatasetSizeOptions();
    } catch (error) {
        LOG.warn("Failed to fetch build options", error);
        if (!state.selectedBuildSize) {
            state.selectedBuildSize = currentBuildSize();
        }
        renderDatasetSizeOptions();
    } finally {
    if (exists(els.datasetSize)) els.datasetSize.prop('disabled', false);
    }
}

async function fetchDiscoveryStatus() {
    if (!state.selectedProject) return;
    try {
        const status = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/refresh/status`);
        handleDiscoveryStatus(status);
        if (status?.status === "running" || status?.status === "pending") {
            startDiscoveryPolling(state.selectedProject);
        } else {
            stopDiscoveryPolling();
        }
    } catch (error) {
        LOG.warn("Failed to fetch discovery status", error);
    }
}

async function fetchBuildStatus() {
    if (!state.selectedProject) return;
    try {
        const status = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/build/status`);
        handleBuildStatus(status);
        if (status.status === "running" || status.status === "pending") {
            startBuildPolling(state.selectedProject);
        } else {
            stopBuildPolling();
        }
    } catch (error) {
        LOG.warn("Failed to fetch build status", error);
    }
}

function folderContains(node, targetPath) {
    if (!node) return false;
    if (node.path === targetPath) return true;
    return (node.children || []).some((child) => folderContains(child, targetPath));
}

function renderFolderTree() {
    if (!exists(els.folderTree)) return;
    els.folderTree.empty();
    if (!state.folderTree) {
        els.folderTree.text(locale("folders_loading", "Loading..."));
        return;
    }
    const list = Q('<ul>', { class: 'folder-list' });
    list.append(renderFolderNode(state.folderTree, 0));
    els.folderTree.append(list);
}

function renderFolderNode(node, depth) {
    const li = Q('<li>');
    const button = Q('<button>', { type: 'button', class: 'folder-row' });
    // Set CSS custom property for depth
    const btnNode = button.get(0);
    if (btnNode) btnNode.style.setProperty('--depth', depth);
    if (node.path) {
        button.attr('data-path', node.path);
    }
    button.html(`<span>${node.name}</span><span class="count">${node.image_count}</span>`);
    if ((state.selectedFolder || "") === node.path) {
        button.addClass('active');
    }
    button.on('click', () => {
        state.selectedFolder = node.path || "";
        state.page = 1;
        renderFolderTree();
        fetchItems();
    });
    li.append(button);
    if (node.children && node.children.length) {
        const childList = Q('<ul>');
        node.children.forEach((child) => childList.append(renderFolderNode(child, depth + 1)));
        li.append(childList);
    }
    return li;
}

function updatePager() {
    const total = state.totalPages || 1;
    if (exists(els.pageIndicator)) els.pageIndicator.text(formatLocale(
        "pager_label",
        "Page {current} / {total} • {items} items",
        { current: state.page, total, items: state.totalItems }
    ));
    if (exists(els.prevPage)) els.prevPage.prop('disabled', state.page <= 1);
    if (exists(els.nextPage)) els.nextPage.prop('disabled', state.page >= total);
}

function updateBulkScope() {
    if (!els.bulkScope) return;
    const selectedCount = state.selectedItems?.size || 0;
    if (selectedCount) {
        const label = selectedCount === 1
            ? locale("scope_selected_one", "1 selected item")
            : formatLocale("scope_selected_many", "{count} selected items", { count: selectedCount });
        if (exists(els.bulkScope)) els.bulkScope.text(label);
        return;
    }
    const count = state.totalItems || 0;
    const base = count === 1
        ? locale("scope_total_one", "1 item")
        : formatLocale("scope_total_many", "{count} items", { count });
    const contexts = [];
    if (state.selectedFolder) {
        contexts.push(formatLocale("scope_folder", "folder {folder}", { folder: state.selectedFolder }));
    }
    if (state.search) {
        contexts.push(formatLocale("scope_search", "search \"{term}\"", { term: state.search }));
    }
    if (exists(els.bulkScope)) {
        els.bulkScope.text(contexts.length
            ? `${base}${locale("scope_separator", " • ")}${contexts.join(locale("scope_separator", " • "))}`
            : base);
    }
}

/**
 * Render the dataset items grid. Uses a DocumentFragment to minimize layout thrashing
 * when inserting many cards into the DOM.
 */
function renderItems() {
    if (!exists(els.grid)) return;
    const container = els.grid.get(0);
    // Unregister previous handlers before clearing DOM to avoid leaks
    thumbnailRegistry.forEach((rec, id) => unregisterThumbnail(id));
    autocompleteRegistry.forEach((entry, textarea) => unregisterAutocomplete(textarea));
    els.grid.empty();
    const fragment = document.createDocumentFragment();
    state.items.forEach((item, index) => {
        const card = Q('<article>', { class: 'card', 'data-id': item.id });
        const thumb = Q('<div>', { class: 'thumbnail' });
        const image = Q('<img>', { src: item.image_url, alt: item.filename });
        thumb.append(image);
        card.append(thumb);

        const meta = Q('<div>', { class: 'meta-line' }).html(`<span>${item.category}</span><span>${item.width}×${item.height}</span>`);
        card.append(meta);

        const tags = Q('<div>', { class: 'tag-list' });
        renderTags(tags, item.tags);
        card.append(tags);

        const annotationWrapper = Q('<div>', { class: 'annotation-editor' });
        const textarea = Q('<textarea>');
        textarea.val(item.annotation || '');
        annotationWrapper.append(textarea);
        card.append(annotationWrapper);

        textarea.on('input', () => {
            const value = textarea.val();
            if (value === (item.annotation || '')) {
                card.removeClass('dirty');
                textarea.removeClass('dirty');
                cancelPendingAnnotationSave(item.id);
                return;
            }
            card.addClass('dirty');
            textarea.addClass('dirty');
            scheduleAnnotationSave(item, value, card.get(0));
        });

    attachTagAutocomplete(textarea.get(0));

    fragment.appendChild(card.get(0));

    const clickHandler = (event) => handleCardClick(event, item, index);
    const entry = { itemId: item.id, container: thumb, image, card, clickHandler };
        registerThumbnail(entry, item);
    });
    // Attach all cards in a single DOM update
    if (container) container.appendChild(fragment);
    renderSelectionState();
}

/**
 * Remove autocomplete handlers and associated UI for a textarea.
 * Cleans up all registered handlers and removes the suggestion panel.
 */
function unregisterAutocomplete(textarea) {
    const entry = autocompleteRegistry.get(textarea);
    if (!entry) return;
    const textareaQ = Q(textarea);
    try {
        if (entry.handlers) {
            if (entry.handlers.input) textareaQ.off('input', entry.handlers.input);
            if (entry.handlers.focus) textareaQ.off('focus', entry.handlers.focus);
            if (entry.handlers.keydown) textareaQ.off('keydown', entry.handlers.keydown);
            if (entry.handlers.blur) textareaQ.off('blur', entry.handlers.blur);
        }
        if (entry.panel && entry.handlersPanel) {
            entry.panel.off('mousedown', entry.handlersPanel.mouseDown);
            entry.panel.off('mouseenter', entry.handlersPanel.mouseEnter);
            entry.panel.off('mouseleave', entry.handlersPanel.mouseLeave);
            entry.panel.remove();
        }
    } catch (err) {}
    autocompleteRegistry.delete(textarea);
}

function handleCardClick(event, item, index) {
    if (event.button !== 0) return;
    if (event.target.closest("textarea")) {
        return;
    }
    const isShift = event.shiftKey;
    const isToggle = event.ctrlKey || event.metaKey;

    if (isShift && state.selectionAnchor !== null) {
        selectRange(state.selectionAnchor, index, isToggle);
    } else if (isToggle) {
        toggleSelection(item.id);
        state.selectionAnchor = index;
    } else {
        state.selectedItems.clear();
        state.selectedItems.add(item.id);
        state.selectionAnchor = index;
    }
    renderSelectionState();
}

function toggleSelection(itemId) {
    if (!state.selectedItems.has(itemId)) {
        state.selectedItems.add(itemId);
    } else {
        state.selectedItems.delete(itemId);
    }
    if (!state.selectedItems.size) {
        state.selectionAnchor = null;
    }
}

function selectRange(anchorIndex, targetIndex, additive = false) {
    if (!Array.isArray(state.itemOrder) || !state.itemOrder.length) return;
    const start = Math.max(0, Math.min(anchorIndex, targetIndex));
    const end = Math.min(state.itemOrder.length - 1, Math.max(anchorIndex, targetIndex));
    if (!additive) {
        state.selectedItems.clear();
    }
    for (let i = start; i <= end; i += 1) {
        const id = state.itemOrder[i];
        if (id) {
            state.selectedItems.add(id);
        }
    }
}

function selectAllVisible() {
    if (!Array.isArray(state.itemOrder) || !state.itemOrder.length) return;
    state.selectedItems = new Set(state.itemOrder);
    state.selectionAnchor = state.itemOrder.length ? 0 : null;
    renderSelectionState();
}

function clearSelection() {
    state.selectedItems.clear();
    state.selectionAnchor = null;
    renderSelectionState();
}

function syncSelectionWithItems() {
    if (!state.selectedItems) {
        state.selectedItems = new Set();
        return;
    }
    const valid = new Set(state.itemOrder);
    state.selectedItems.forEach((itemId) => {
        if (!valid.has(itemId)) {
            state.selectedItems.delete(itemId);
        }
    });
    if (!state.selectedItems.size) {
        state.selectionAnchor = null;
    }
}

function renderSelectionState() {
    state.items.forEach((item) => {
        const card = findCard(item.id);
        if (card) {
            Q(card).toggleClass("selected", state.selectedItems.has(item.id));
        }
    });
    updateBulkScope();
}

function getSelectedIds() {
    return Array.from(state.selectedItems || []);
}

async function deleteSelectedItems() {
    if (!state.selectedProject) return;
    if (!state.selectedItems || !state.selectedItems.size) return;
    if (state.deletionInFlight) return;
    state.deletionInFlight = true;
    const items = getSelectedIds();
    try {
        const result = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/items/delete`, {
            method: "POST",
            body: JSON.stringify({ items }),
        });
        const deleted = Number(result?.deleted ?? 0);
        if (deleted) {
            const message = deleted === 1
                ? locale("delete_success_one", "Deleted 1 item")
                : formatLocale("delete_success_many", "Deleted {count} items", { count: deleted });
            notify(message);
        } else {
            notify(locale("delete_none", "No items deleted"));
        }
        state.selectedItems.clear();
        state.selectionAnchor = null;
        renderSelectionState();
        await fetchItems();
        await fetchStats();
    } finally {
        state.deletionInFlight = false;
    }
}

function isEditableElement(element) {
    if (!element) return false;
    const tag = element.tagName?.toLowerCase();
    return tag === "input" || tag === "textarea" || element.isContentEditable;
}

function handleGlobalKeyDown(event) {
    if (!datasetEditorActive) {
        return;
    }
    if (event.key === "Control") {
        return;
    }
    const active = document.activeElement;
    const isEditing = isEditableElement(active);
    if ((event.ctrlKey || event.metaKey) && event.key.toLowerCase() === "a") {
        if (isEditing) {
            return;
        }
        event.preventDefault();
        selectAllVisible();
    } else if (event.key === "Delete") {
        if (isEditing) {
            return;
        }
        if (state.selectedItems && state.selectedItems.size) {
            event.preventDefault();
            deleteSelectedItems().catch(reportError);
        }
    }
}

/**
 * Register a thumbnail and its handlers.
 * The function stores named handler references (pointerdown, wheel, click)
 * so they can be removed later by unregisterThumbnail.
 */
function registerThumbnail(entry, item) {
    const containerQ = entry.container && entry.container.get ? entry.container : Q(entry.container);
    const imageQ = entry.image && entry.image.get ? entry.image : Q(entry.image);
    const cardQ = entry.card && entry.card.get ? entry.card : Q(entry.card);
    const record = { itemId: entry.itemId, container: containerQ, image: imageQ, card: cardQ, item, handlers: {} };
    thumbnailRegistry.set(entry.itemId, record);

    if (containerQ && containerQ.get(0)) {
        const pointerHandler = (event) => beginPan(event, entry.itemId);
        const wheelHandler = (event) => handleWheel(event, entry.itemId);
        containerQ.on('pointerdown', pointerHandler);
        containerQ.on('wheel', wheelHandler);
        record.handlers.pointerdown = pointerHandler;
        record.handlers.wheel = wheelHandler;
    }
    if (cardQ && cardQ.get(0) && entry.clickHandler) {
        const clickHandler = entry.clickHandler;
        cardQ.on('click', clickHandler);
        record.handlers.click = clickHandler;
    }
    imageLoaded(imageQ, () => applyCrop(thumbnailRegistry.get(entry.itemId)));
}

/**
 * Unregisters thumbnail handlers for the item and removes it from the registry.
 * Safe to call multiple times. Removes attached pointer/wheel/click handlers.
 */
function unregisterThumbnail(itemId) {
    const record = thumbnailRegistry.get(itemId);
    if (!record) return;
    const { container, card, handlers } = record;
    try {
        if (container && handlers?.pointerdown) container.off('pointerdown', handlers.pointerdown);
        if (container && handlers?.wheel) container.off('wheel', handlers.wheel);
        if (card && handlers?.click) card.off('click', handlers.click);
    } catch (err) {
        // ignore
    }
    thumbnailRegistry.delete(itemId);
}

function allowThumbnailCrop(event) {
    return Boolean(event?.ctrlKey || state.ctrlPressed);
}

function imageLoaded(img, callback) {
    const node = img && img.get ? img.get(0) : img;
    if (!node) return;
    if (node.complete) {
        callback();
    } else {
        Q(node).on('load', callback, { once: true });
    }
}

function beginPan(event, itemId, options = {}) {
    const { force = false, containerOverride = null } = options;
    if (!force && !allowThumbnailCrop(event)) return;
    event.preventDefault();
    const entry = containerOverride ? null : thumbnailRegistry.get(itemId);
    // Resolve containerOverride or entry.container into a raw DOM element for sizing
    let containerNode = null;
    if (containerOverride) {
        containerNode = containerOverride.get ? containerOverride.get(0) : containerOverride;
    } else if (entry && entry.container) {
        containerNode = entry.container.get ? entry.container.get(0) : entry.container;
    }
    if (!containerNode) return;
    const pointerId = event.pointerId;
    const start = { x: event.clientX, y: event.clientY };

    const handleMove = (moveEvent) => {
        if (moveEvent.pointerId !== pointerId) return;
        if (!force && !allowThumbnailCrop(moveEvent)) return;
        const dx = moveEvent.clientX - start.x;
        const dy = moveEvent.clientY - start.y;
        start.x = moveEvent.clientX;
        start.y = moveEvent.clientY;
        translateBounds(itemId, dx, dy, containerNode.clientWidth);
    };

    const endPan = (upEvent) => {
        if (upEvent.pointerId !== pointerId) return;
        window.removeEventListener("pointermove", handleMove);
        window.removeEventListener("pointerup", endPan);
        window.removeEventListener("pointercancel", endPan);
    };

    window.addEventListener("pointermove", handleMove);
    window.addEventListener("pointerup", endPan);
    window.addEventListener("pointercancel", endPan);
}

function handleWheel(event, itemId, force = false) {
    if (!force && !allowThumbnailCrop(event)) return;
    event.preventDefault();
    const item = getItem(itemId);
    if (!item) return;
    const factor = event.deltaY < 0 ? 0.92 : 1.08;
    const next = { ...item.bounds, zoom: clampZoom(item.bounds.zoom * factor) };
    updateBounds(itemId, next);
}

function translateBounds(itemId, dx, dy, containerSize) {
    const item = getItem(itemId);
    if (!item || !containerSize) return;
    const scale = item.min_dimension / (containerSize * item.bounds.zoom);
    const deltaX = dx * scale / item.width;
    const deltaY = dy * scale / item.height;
    const next = {
        ...item.bounds,
        center_x: item.bounds.center_x - deltaX,
        center_y: item.bounds.center_y - deltaY,
    };
    updateBounds(itemId, next);
}

function clampZoom(zoom) {
    return Math.min(CONSTRAINTS.MAX_ZOOM, Math.max(CONSTRAINTS.MIN_ZOOM, zoom));
}

function clampBounds(bounds, item) {
    const zoom = clampZoom(bounds.zoom);
    bounds.zoom = zoom;
    const cropSize = item.min_dimension / zoom;
    const halfW = cropSize / 2 / item.width;
    const halfH = cropSize / 2 / item.height;
    bounds.center_x = Math.min(1 - halfW, Math.max(halfW, bounds.center_x));
    bounds.center_y = Math.min(1 - halfH, Math.max(halfH, bounds.center_y));
    return bounds;
}

function updateBounds(itemId, partialBounds) {
    const item = getItem(itemId);
    if (!item) return;
    const next = clampBounds({ ...partialBounds }, item);
    item.bounds = next;
    const entry = thumbnailRegistry.get(itemId);
    if (entry) {
        entry.item = item;
        applyCrop(entry);
    }
    scheduleBoundsSave(itemId, next);
}

function scheduleBoundsSave(itemId, bounds) {
    if (pendingBoundsSaves.has(itemId)) {
        clearTimeout(pendingBoundsSaves.get(itemId));
    }
    const timeout = setTimeout(() => {
        pendingBoundsSaves.delete(itemId);
        persistBounds(itemId, bounds).catch(reportError);
    }, TIMING.BOUNDS_SAVE_DELAY);
    pendingBoundsSaves.set(itemId, timeout);
}

async function persistBounds(itemId, bounds) {
    const payload = JSON.stringify(bounds);
    const encoded = encodeURIComponent(itemId);
    await apiFetch(`/dataset/editor/projects/${state.selectedProject}/items/${encoded}/bounds`, {
        method: "POST",
        body: payload,
    });
}

function applyCrop(entry) {
    if (!entry) return;
    const { item } = entry;
    const containerEl = entry.container && entry.container.get ? entry.container.get(0) : entry.container;
    const imageEl = entry.image && entry.image.get ? entry.image.get(0) : entry.image;
    if (!containerEl || !imageEl) return;
    const box = containerEl.getBoundingClientRect();
    const size = box.width || containerEl.clientWidth;
    if (!size) return;

    const { width, height, min_dimension } = item;
    const { zoom, center_x, center_y } = item.bounds;

    const displayWidth = size * (width * zoom) / min_dimension;
    const displayHeight = size * (height * zoom) / min_dimension;
    const cropSize = min_dimension / zoom;
    const left = center_x * width - cropSize / 2;
    const top = center_y * height - cropSize / 2;
    const offsetX = -left * (displayWidth / width);
    const offsetY = -top * (displayHeight / height);

    imageEl.style.width = `${displayWidth}px`;
    imageEl.style.height = `${displayHeight}px`;
    imageEl.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
}

function cancelPendingAnnotationSave(itemId) {
    if (!pendingAnnotationSaves.has(itemId)) return;
    clearTimeout(pendingAnnotationSaves.get(itemId));
    pendingAnnotationSaves.delete(itemId);
}

function scheduleAnnotationSave(item, content, card) {
    if (content === item.annotation) {
        Q(card).removeClass('dirty');
        cancelPendingAnnotationSave(item.id);
        return;
    }
    cancelPendingAnnotationSave(item.id);
    const timeout = setTimeout(() => {
        pendingAnnotationSaves.delete(item.id);
        persistAnnotation(item, content, card).catch(reportError);
    }, TIMING.ANNOTATION_SAVE_DELAY);
    pendingAnnotationSaves.set(item.id, timeout);
}

async function persistAnnotation(item, content, card) {
    // Validate annotation before persisting
    const validated = validateAnnotation(content);
    if (!validated.valid) {
        LOG.warn(`Invalid annotation: ${validated.error}`);
        reportError(new Error(validated.error));
        return;
    }

    const encoded = encodeURIComponent(item.id);
    const payload = { content: validated.sanitized };
    const updated = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/items/${encoded}/annotation`, {
        method: "POST",
        body: JSON.stringify(payload),
    });
    item.annotation = updated.annotation;
    item.tags = updated.tags;
    const targetCardEl = card && card.get ? card.get(0) : (card || findCard(item.id));
    if (targetCardEl) {
        const $card = Q(targetCardEl);
        $card.removeClass('dirty');
        const textareaEl = $card.find('textarea').get(0);
        if (textareaEl && textareaEl !== document.activeElement) {
            Q(textareaEl).val(updated.annotation);
        }
        if (textareaEl) Q(textareaEl).removeClass('dirty');
        const tagListEl = $card.find('.tag-list').get(0);
        if (tagListEl) renderTags(Q(tagListEl), updated.tags);
    }
}

async function runBuild() {
    if (!state.selectedProject) return;
    try {
        const size = currentBuildSize();
        const status = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/build`, {
            method: "POST",
            body: JSON.stringify({ size }),
        });
        notify(formatLocale("build_queue", "Dataset build queued @ {size}px", { size }));
        handleBuildStatus(status);
        if (status.status === "running" || status.status === "pending") {
            startBuildPolling(state.selectedProject);
        }
    } catch (error) {
        reportError(error);
    }
}

async function refreshProject() {
    if (!state.selectedProject) return;
    if (els.refreshProject && exists(els.refreshProject)) {
        els.refreshProject.prop('disabled', true);
        els.refreshProject.text(locale("refresh_queueing", "Queueing…"));
    }
    try {
        const status = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/refresh`, {
            method: "POST",
        });
        notify(locale("discovery_queued", "Discovery queued"));
        handleDiscoveryStatus(status);
        if (status.status === "running" || status.status === "pending") {
            startDiscoveryPolling(state.selectedProject);
        } else if (status.status === "success") {
            refreshData();
        } else {
            stopDiscoveryPolling();
        }
    } catch (error) {
        reportError(error);
        stopDiscoveryPolling();
        updateRefreshButton(null);
    }
}

async function refreshFoldersManual() {
    if (!state.selectedProject) return;
    if (els.folderRefresh) {
    if (els.folderRefresh && exists(els.folderRefresh)) {
            els.folderRefresh.prop('disabled', true);
            els.folderRefresh.text(locale("refreshing_label", "Refreshing…"));
        }
    }
    try {
        await Promise.all([fetchFolders(), fetchStats()]);
        notify(locale("folders_refreshed", "Folders refreshed"));
    } catch (error) {
        reportError(error);
    } finally {
        if (els.folderRefresh) {
            if (els.folderRefresh && exists(els.folderRefresh)) {
                els.folderRefresh.prop('disabled', false);
                els.folderRefresh.text(locale("refresh_label", "Refresh"));
            }
        }
    }
}

function getItem(itemId) {
    return state.items.find((item) => item.id === itemId);
}

function findCard(itemId) {
    const selector = `.card[data-id="${cssEscape(itemId)}"]`;
    return els.grid.find(selector).get(0);
}

async function apiFetch(url, options = {}) {
    const method = (options.method || 'GET').toUpperCase();
    const headers = { 'Content-Type': 'application/json', ...(options.headers || {}) };
    
    // Try to parse JSON body if provided as a string
    let data = undefined;
    if (options.body) {
        if (typeof options.body === 'string') {
            try { data = JSON.parse(options.body); } catch (e) { data = options.body; }
        } else {
            data = options.body;
        }
    }

    // Create a simple hash for deduplication (only for GET, idempotent operations)
    let cacheKey = null;
    if (method === 'GET' && !options.noDedup) {
        // Simple hash: method:url (GET body is rare)
        cacheKey = `${method}:${url}`;
        if (pendingRequests.has(cacheKey)) {
            LOG.debug(`Deduplicating identical request: ${cacheKey}`);
            return pendingRequests.get(cacheKey);
        }
    }

    // Determine appropriate timeout based on operation type
    const timeout = options.timeout || (method === 'GET' ? TIMING.API_TIMEOUT_SHORT : TIMING.API_TIMEOUT_LONG);
    
    // Create the request promise
    const executeRequest = async () => {
        // Retry logic for network errors (exponential backoff)
        let lastError;
        for (let attempt = 0; attempt < CONSTRAINTS.MAX_FETCH_RETRIES; attempt++) {
            try {
                const result = await Q.ajax({ url, method, headers, data, timeout });
                return result;
            } catch (error) {
                lastError = error;
                
                // Only retry on network/timeout errors, not on 4xx/5xx HTTP errors
                const isNetwork = error.message && /network|timeout|connection/i.test(error.message);
                if (!isNetwork || attempt >= CONSTRAINTS.MAX_FETCH_RETRIES - 1) {
                    break;
                }
                
                // Exponential backoff: 500ms, 1000ms, 1500ms...
                const delayMs = CONSTRAINTS.RETRY_BACKOFF_MS * (attempt + 1);
                LOG.warn(`API request failed (attempt ${attempt + 1}/${CONSTRAINTS.MAX_FETCH_RETRIES}), retrying in ${delayMs}ms...`);
                await new Promise(resolve => setTimeout(resolve, delayMs));
            }
        }
        
        // All retries exhausted
        const message = (lastError && lastError.message) ? lastError.message : formatLocale('request_failed', 'Request failed');
        throw new Error(message);
    };

    const promise = executeRequest();
    
    // Store in cache if deduplicating
    if (cacheKey) {
        pendingRequests.set(cacheKey, promise);
        promise.finally(() => {
            pendingRequests.delete(cacheKey);
        });
    }

    return promise;
}

function notify(message) {
    if (exists(els.toast)) {
        els.toast.text(message);
        els.toast.removeClass("hidden");
    }
    setTimeout(() => {
        if (exists(els.toast)) els.toast.addClass('hidden');
    }, TIMING.TOAST_DURATION);
}

function renderTags(container, tags) {
    const $container = container && container.get ? container : Q(container);
    if (!$container || !$container.get(0)) return;
    $container.empty();
    (tags || []).slice(0, 4).forEach((tag) => {
        const pill = Q('<span>', { class: 'tag-pill', text: tag });
        $container.append(pill);
    });
}

function reportError(error) {
    LOG.error(error);
    notify(error.message || locale("unexpected_error", "Unexpected error"));
}

function parseTagInput(value) {
    return value
        .split(/[,\n]/)
        .map((tag) => tag.trim().toLowerCase())
        .filter(Boolean);
}

function detectAnnotationSeparator(annotation) {
    if (!annotation) {
        return ", ";
    }
    if (annotation.includes("\r\n")) return "\r\n";
    if (annotation.includes("\n")) return "\n";
    if (annotation.includes(", ")) return ", ";
    if (annotation.includes(",")) return ",";
    if (annotation.includes("; ")) return "; ";
    if (annotation.includes(";")) return ";";
    if (annotation.includes("|")) return "|";
    if (annotation.includes(" ")) return " ";
    return ", ";
}

function rewriteAnnotationText(previousAnnotation, tags) {
    if (!Array.isArray(tags) || !tags.length) {
        return "";
    }
    const separator = detectAnnotationSeparator(previousAnnotation);
    return tags.join(separator);
}

function arraysEqual(left, right) {
    if (left.length !== right.length) return false;
    for (let index = 0; index < left.length; index += 1) {
        if (left[index] !== right[index]) {
            return false;
        }
    }
    return true;
}

function updateCardEditor(item) {
    const cardEl = findCard(item.id);
    if (!cardEl) return;
    const $card = Q(cardEl);
    const $tagList = $card.find('.tag-list');
    if ($tagList && $tagList.get(0)) {
        renderTags($tagList, item.tags);
    }
    const $textarea = $card.find('.annotation-editor textarea');
    if ($textarea && $textarea.get(0)) {
        const textareaEl = $textarea.get(0);
        const isActive = document.activeElement === textareaEl;
        const start = isActive ? textareaEl.selectionStart : 0;
        const end = isActive ? textareaEl.selectionEnd : 0;
        $textarea.val(item.annotation || '');
        $textarea.removeClass('dirty');
        $card.removeClass('dirty');
        cancelPendingAnnotationSave(item.id);
        if (isActive) {
            const length = textareaEl.value.length;
            textareaEl.selectionStart = Math.min(start, length);
            textareaEl.selectionEnd = Math.min(end, length);
        }
    }
}

function applyBulkTagsLocally(add, remove, targetIds = null) {
    if (!Array.isArray(state.items) || !state.items.length) {
        return 0;
    }
    const items = state.items;
    const addSet = new Set(add);
    const removeSet = new Set(remove);
    const targetSet = Array.isArray(targetIds) && targetIds.length ? new Set(targetIds) : null;
    let touched = 0;
    items.forEach((item) => {
        if (targetSet && !targetSet.has(item.id)) {
            return;
        }
        const nextSet = new Set(item.tags || []);
        removeSet.forEach((tag) => nextSet.delete(tag));
        addSet.forEach((tag) => nextSet.add(tag));
        const nextTags = Array.from(nextSet).sort();
        if (arraysEqual(item.tags || [], nextTags)) {
            return;
        }
        const previousAnnotation = item.annotation || "";
        item.tags = nextTags;
        item.annotation = rewriteAnnotationText(previousAnnotation, nextTags);
        updateCardEditor(item);
        touched += 1;
    });
    if (touched) {
        refreshActiveModalState();
    }
    return touched;
}

function refreshActiveModalState() { }

async function applyBulkTags() {
    // Validate project selection first
    const projCheck = validateProjectSelected();
    if (!projCheck.valid) {
        notify(projCheck.error);
        return;
    }

    const addInput = (exists(els.bulkAddInput) ? els.bulkAddInput.val() : "").trim();
    const removeInput = (exists(els.bulkRemoveInput) ? els.bulkRemoveInput.val() : "").trim();
    const add = parseTagInput(addInput);
    const remove = parseTagInput(removeInput);
    
    // Validate tag count
    if (add.length + remove.length > CONSTRAINTS.MAX_BULK_TAGS) {
        notify(formatLocale("bulk_too_many_tags", "Too many tags (max {count})", { count: CONSTRAINTS.MAX_BULK_TAGS }));
        return;
    }

    if (!add.length && !remove.length) {
        notify(locale("bulk_need_tags", "Add or remove at least one tag"));
        return;
    }

    const selection = getSelectedIds();
    const hasSelection = selection.length > 0;
    const payload = { add, remove };
    if (hasSelection) {
        payload.items = selection;
    } else {
        if (state.search) payload.search = state.search;
        if (state.selectedFolder) payload.folder = state.selectedFolder;
    }

    if (exists(els.bulkApply)) els.bulkApply.prop('disabled', true);
    const originalLabel = exists(els.bulkApply) ? els.bulkApply.get(0).textContent : '';
    if (exists(els.bulkApply)) els.bulkApply.text(locale("bulk_applying", "Applying…"));
    try {
        const result = await apiFetch(`/dataset/editor/projects/${state.selectedProject}/bulk/tags`, {
            method: "POST",
            body: JSON.stringify(payload),
        });
        notify(formatLocale(
            "bulk_result",
            "Bulk edit applied ({updated} updated, {skipped} skipped)",
            {
                updated: Number(result.updated ?? 0),
                skipped: Number(result.skipped ?? 0),
            }
        ));
        if (exists(els.bulkAddInput)) els.bulkAddInput.val('');
        if (exists(els.bulkRemoveInput)) els.bulkRemoveInput.val('');
        applyBulkTagsLocally(add, remove, hasSelection ? selection : null);
        await fetchItems();
        await fetchStats();
    } finally {
    if (exists(els.bulkApply)) els.bulkApply.prop('disabled', false);
    if (exists(els.bulkApply)) els.bulkApply.get(0).textContent = originalLabel;
    }
}

function handleBuildStatus(status) {
    if (!status) return;
    const previous = lastBuildState;
    state.buildStatus = status;
    if (Number.isFinite(status.target_size)) {
        state.activeBuildSize = status.target_size;
        if (!state.buildSizeDirty) {
            state.selectedBuildSize = status.target_size;
            renderDatasetSizeOptions();
        }
    }
    renderBuildProgress(status);
    const wasRunning = previous === "running" || previous === "pending";
    if (wasRunning && status.status === "success") {
        const processed = status.result?.processed_images ?? status.built_images;
        const deleted = status.result?.deleted_images ?? status.deleted_images;
        const summary = [
            formatLocale("build_summary_updated", "{count} updated", { count: processed }),
        ];
        if (deleted) {
            summary.push(formatLocale("build_summary_removed", "{count} removed", { count: deleted }));
        }
        notify(formatLocale("build_success", "Dataset built ({summary})", { summary: joinWithComma(summary) }));
        fetchStats();
    } else if (wasRunning && status.status === "error") {
        const message = status.last_error
            ? formatLocale("build_failed_with_reason", "Build failed: {reason}", { reason: status.last_error })
            : locale("build_failed", "Build failed");
        notify(message);
    }
    lastBuildState = status.status;
    const busy = status.status === "running" || status.status === "pending";
    if (exists(els.buildButton)) {
        els.buildButton.prop('disabled', busy);
        if (busy) {
            if (status.status === "pending") {
                els.buildButton.text(locale("build_button_preparing", "Preparing…"));
            } else {
                const total = status.total_items || status.completed_items || 1;
                els.buildButton.text(formatLocale(
                    "build_button_running",
                    "Building {completed}/{total}",
                    { completed: status.completed_items || 0, total }
                ));
            }
        } else {
            els.buildButton.text(locale("build_button", "Build Dataset"));
        }
    }
}

function renderBuildProgress(status) {
    // Update global status for footer
    if (!window.Hootsight) window.Hootsight = {};
    window.Hootsight.datasetEditorStatus = window.Hootsight.datasetEditorStatus || {};

    // Only keep status if it's active (pending/running)
    if (status && (status.status === 'pending' || status.status === 'running')) {
        window.Hootsight.datasetEditorStatus.build = status;
    } else {
        // Clear status on success/error/idle
        window.Hootsight.datasetEditorStatus.build = null;
    }

    // Trigger footer update if available
    if (window.Hootsight.actions && typeof window.Hootsight.actions.applyFooterStatus === 'function') {
        window.Hootsight.actions.applyFooterStatus();
    }
}

function describeBuildStats(status) {
    if (!status) return locale("build_stats_none", "No builds yet.");
    if (status.status === "idle" && !status.result) {
        return locale("build_stats_none", "No builds yet.");
    }
    const processed = status.result?.processed_images ?? status.built_images;
    const deleted = status.result?.deleted_images ?? status.deleted_images;
    const skipped = status.result?.skipped_images ?? status.skipped_images;
    const parts = [];
    parts.push(formatLocale("build_stats_updated", "{count} updated", { count: processed }));
    if (deleted) parts.push(formatLocale("build_stats_removed", "{count} removed", { count: deleted }));
    if (skipped) parts.push(formatLocale("build_stats_failed", "{count} failed", { count: skipped }));
    if (status.status === "running" && status.total_items) {
        parts.push(formatLocale("build_stats_ratio", "{completed}/{total}", {
            completed: status.completed_items || 0,
            total: status.total_items,
        }));
    }
    if (status.target_size) {
        parts.push(formatLocale("build_stats_size", "{size}px crops", { size: status.target_size }));
    }
    if (status.result?.updated_at) {
        const stamp = new Date(status.result.updated_at);
        const time = stamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        parts.push(formatLocale("build_stats_last_run", "Last run {time}", { time }));
    }
    return parts.filter(Boolean).join(locale("bullet_separator", " • "))
        || locale("build_stats_waiting", "Waiting for first build.");
}

function formatDuration(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    const total = Math.max(0, Math.round(seconds));
    const minutes = Math.floor(total / 60);
    const secs = total % 60;
    if (minutes && secs) return `${minutes}m ${secs}s`;
    if (minutes) return `${minutes}m`;
    return `${secs}s`;
}

function startBuildPolling(projectName) {
    if (!projectName) return;
    if (buildPollProject === projectName && buildPollHandle) {
        return;
    }
    stopBuildPolling();
    buildPollProject = projectName;
    const tick = async () => {
        if (!buildPollProject || buildPollProject !== state.selectedProject) {
            stopBuildPolling();
            return;
        }
        try {
            const status = await apiFetch(`/dataset/editor/projects/${buildPollProject}/build/status`);
            handleBuildStatus(status);
            if (status.status === "running" || status.status === "pending") {
                buildPollHandle = setTimeout(tick, TIMING.POLLING_INTERVAL);
            } else {
                stopBuildPolling();
            }
        } catch (error) {
        LOG.warn("Build poll failed", error);
            stopBuildPolling();
        }
    };
    tick();
}

function stopBuildPolling() {
    if (buildPollHandle) {
        clearTimeout(buildPollHandle);
        buildPollHandle = null;
    }
    buildPollProject = null;

    // Clear global status and update footer
    if (!window.Hootsight) window.Hootsight = {};
    window.Hootsight.datasetEditorStatus = window.Hootsight.datasetEditorStatus || {};
    window.Hootsight.datasetEditorStatus.build = null;

    if (window.Hootsight.actions && typeof window.Hootsight.actions.applyFooterStatus === 'function') {
        window.Hootsight.actions.applyFooterStatus();
    }
}

function handleDiscoveryStatus(status) {
    if (!status) return;
    const previous = lastDiscoveryState;
    state.discoveryStatus = status;
    renderDiscoveryProgress(status);
    updateRefreshButton(status);
    if ((previous === "running" || previous === "pending") && status.status === "success") {
        const parts = [];
        if (status.added_items) parts.push(formatLocale("discovery_summary_added", "{count} new", { count: status.added_items }));
        if (status.updated_items) parts.push(formatLocale("discovery_summary_updated", "{count} changed", { count: status.updated_items }));
        if (status.removed_items) parts.push(formatLocale("discovery_summary_removed", "{count} removed", { count: status.removed_items }));
        if (parts.length) {
            notify(formatLocale("discovery_summary", "Discovery updated ({summary})", { summary: joinWithComma(parts) }));
        } else {
            notify(locale("discovery_refreshed", "Discovery refreshed"));
        }
        refreshData();
    } else if ((previous === "running" || previous === "pending") && status.status === "error") {
        const message = status.last_error
            ? formatLocale("discovery_failed_with_reason", "Discovery failed: {reason}", { reason: status.last_error })
            : locale("discovery_failed", "Discovery failed");
        notify(message);
    }
    lastDiscoveryState = status.status;
}

function renderDiscoveryProgress(status) {
    // Update global status for footer
    if (!window.Hootsight) window.Hootsight = {};
    window.Hootsight.datasetEditorStatus = window.Hootsight.datasetEditorStatus || {};

    // Only keep status if it's active (pending/running)
    if (status && (status.status === 'pending' || status.status === 'running')) {
        window.Hootsight.datasetEditorStatus.discovery = status;
    } else {
        // Clear status on success/error/idle
        window.Hootsight.datasetEditorStatus.discovery = null;
    }

    // Trigger footer update if available
    if (window.Hootsight.actions && typeof window.Hootsight.actions.applyFooterStatus === 'function') {
        window.Hootsight.actions.applyFooterStatus();
    }
}

function describeDiscoveryStats(status) {
    if (!status) return locale("discovery_stats_none", "No scans yet.");
    if (status.status === "idle" && !status.result) {
        return locale("discovery_stats_none", "No scans yet.");
    }
    const parts = [];
    parts.push(formatLocale("discovery_stats_processed", "{count} processed", { count: status.processed_items || 0 }));
    if (status.added_items) parts.push(formatLocale("discovery_stats_added", "{count} new", { count: status.added_items }));
    if (status.updated_items) parts.push(formatLocale("discovery_stats_updated", "{count} changed", { count: status.updated_items }));
    if (status.removed_items) parts.push(formatLocale("discovery_stats_removed", "{count} removed", { count: status.removed_items }));
    if (status.skipped_items) parts.push(formatLocale("discovery_stats_skipped", "{count} skipped", { count: status.skipped_items }));
    if (status.result?.updated_at) {
        const stamp = new Date(status.result.updated_at);
        const time = stamp.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        parts.push(formatLocale("discovery_stats_last_run", "Last run {time}", { time }));
    }
    return parts.filter(Boolean).join(locale("bullet_separator", " • "))
        || locale("discovery_stats_waiting", "Scanning…");
}

function startDiscoveryPolling(projectName) {
    if (!projectName) return;
    if (discoveryPollProject === projectName && discoveryPollHandle) {
        return;
    }
    stopDiscoveryPolling();
    discoveryPollProject = projectName;
    const tick = async () => {
        if (!discoveryPollProject || discoveryPollProject !== state.selectedProject) {
            stopDiscoveryPolling();
            return;
        }
        try {
            const status = await apiFetch(`/dataset/editor/projects/${discoveryPollProject}/refresh/status`);
            handleDiscoveryStatus(status);
            if (status.status === "running" || status.status === "pending") {
                discoveryPollHandle = setTimeout(tick, TIMING.POLLING_INTERVAL);
            } else {
                stopDiscoveryPolling();
            }
        } catch (error) {
        LOG.warn("Discovery poll failed", error);
            stopDiscoveryPolling();
        }
    };
    tick();
}

function stopDiscoveryPolling() {
    if (discoveryPollHandle) {
        clearTimeout(discoveryPollHandle);
        discoveryPollHandle = null;
    }
    discoveryPollProject = null;

    // Clear global status and update footer
    if (!window.Hootsight) window.Hootsight = {};
    window.Hootsight.datasetEditorStatus = window.Hootsight.datasetEditorStatus || {};
    window.Hootsight.datasetEditorStatus.discovery = null;

    if (window.Hootsight.actions && typeof window.Hootsight.actions.applyFooterStatus === 'function') {
        window.Hootsight.actions.applyFooterStatus();
    }
}

function updateRefreshButton(status) {
    if (!exists(els.refreshProject)) return;
    const busy = Boolean(status && (status.status === "running" || status.status === "pending"));
    els.refreshProject.prop('disabled', busy);
    els.refreshProject.text(busy
        ? (status.status === "pending"
            ? locale("refresh_queueing", "Queueing…")
            : locale("refresh_scanning", "Scanning…"))
        : locale("refresh_label", "Refresh"));
}

/**
 * Attach autocomplete support for tags to a textarea element. Creates a
 * suggestion panel and wires input/focus/keydown/blur events. Handler
 * references are stored so they can be removed with unregisterAutocomplete.
 */
function attachTagAutocomplete(textarea) {
    const textareaEl = textarea && textarea.get ? textarea.get(0) : textarea;
    if (!textareaEl || autocompleteRegistry.has(textareaEl)) return;
    const panel = Q('<div>', { class: 'tag-suggestions hidden', role: 'listbox' });
    panel.attr('aria-hidden', 'true');
    // Insert the panel after the textarea
    Q(textareaEl).after(panel);
    const entry = { panel, suggestions: [], activeIndex: -1, hideTimer: null, handlers: {}, handlersPanel: {} };
    autocompleteRegistry.set(textareaEl, entry);
    const handleInput = () => maybeShowTagSuggestions(textareaEl);
    const handleFocus = () => maybeShowTagSuggestions(textareaEl);
    const handleKeydown = (event) => handleTagSuggestionKeydown(event, textareaEl);
    const handleBlur = () => scheduleSuggestionHide(textareaEl);
    entry.handlers.input = handleInput;
    entry.handlers.focus = handleFocus;
    entry.handlers.keydown = handleKeydown;
    entry.handlers.blur = handleBlur;
    Q(textareaEl).on('input', handleInput);
    Q(textareaEl).on('focus', handleFocus);
    Q(textareaEl).on('keydown', handleKeydown);
    Q(textareaEl).on('blur', handleBlur);

    const panelMouseDown = (event) => {
        event.preventDefault();
        const button = event.target.closest('button[data-tag]');
        if (button) {
            applyTagSuggestion(textareaEl, button.dataset.tag);
        }
    };
    const panelMouseEnter = () => clearSuggestionHide(textareaEl);
    const panelMouseLeave = () => scheduleSuggestionHide(textareaEl);
    entry.handlersPanel.mouseDown = panelMouseDown;
    entry.handlersPanel.mouseEnter = panelMouseEnter;
    entry.handlersPanel.mouseLeave = panelMouseLeave;
    panel.on('mousedown', panelMouseDown);
    panel.on('mouseenter', panelMouseEnter);
    panel.on('mouseleave', panelMouseLeave);
}

function maybeShowTagSuggestions(textarea) {
    const entry = autocompleteRegistry.get(textarea);
    if (!entry || !state.tagHints || !state.tagHints.length) {
        hideTagSuggestions(textarea);
        return;
    }
    clearSuggestionHide(textarea);
    const meta = extractTokenMeta(textarea);
    entry.meta = meta;
    const matches = (state.tagHints || [])
        .filter((tag) => {
            if (!meta || !meta.tokenLower) {
                return true;
            }
            return tag.toLowerCase().startsWith(meta.tokenLower);
        })
        .slice(0, 6);
    if (!matches.length) {
        hideTagSuggestions(textarea);
        return;
    }
    entry.suggestions = matches;
    entry.activeIndex = -1;
    entry.panel.empty();
    matches.forEach((tag, idx) => {
        const btn = Q('<button>', { type: 'button', role: 'option', 'data-tag': tag, text: tag });
        if (idx === entry.activeIndex) btn.addClass('active');
        entry.panel.append(btn);
    });
    entry.panel.removeClass('hidden');
    entry.panel.attr('aria-hidden', 'false');
}

function handleTagSuggestionKeydown(event, textarea) {
    const entry = autocompleteRegistry.get(textarea);
    if (!entry || entry.panel.hasClass("hidden")) {
        return;
    }

    if (event.key === "ArrowDown" || event.key === "ArrowUp") {
        event.preventDefault();
        if (!entry.suggestions.length) return;
        const delta = event.key === "ArrowDown" ? 1 : -1;
        entry.activeIndex = (entry.activeIndex + delta + entry.suggestions.length) % entry.suggestions.length;
        updateSuggestionHighlight(entry);
    } else if (event.key === "Enter" || event.key === "Tab") {
        if (entry.suggestions.length > 0) {
            // If the user hasn't moved selection, fall back to the first suggestion
            const index = entry.activeIndex >= 0 ? entry.activeIndex : 0;
            event.preventDefault();
            const tag = entry.suggestions[index];
            applyTagSuggestion(textarea, tag);
        }
    } else if (event.key === "Escape") {
        hideTagSuggestions(textarea);
    }
}

function updateSuggestionHighlight(entry) {
    const buttons = entry.panel.find('button[data-tag]').getAll();
    buttons.forEach((btn, index) => {
        const isActive = index === entry.activeIndex;
        Q(btn).toggleClass('active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
    });
    // Make sure the active suggestion is visible if the list is scrollable
    if (entry.activeIndex >= 0) {
        const activeBtn = buttons[entry.activeIndex];
        if (activeBtn && typeof activeBtn.scrollIntoView === 'function') {
            activeBtn.scrollIntoView({ block: 'nearest' });
        }
    }
}

function scheduleSuggestionHide(textarea) {
    const entry = autocompleteRegistry.get(textarea);
    if (!entry) return;
    clearSuggestionHide(textarea);
    entry.hideTimer = setTimeout(() => hideTagSuggestions(textarea), TIMING.SUGGESTION_HIDE_DELAY);
}

function clearSuggestionHide(textarea) {
    const entry = autocompleteRegistry.get(textarea);
    if (entry && entry.hideTimer) {
        clearTimeout(entry.hideTimer);
        entry.hideTimer = null;
    }
}

function hideTagSuggestions(textarea) {
    const entry = autocompleteRegistry.get(textarea);
    if (!entry) return;
    entry.panel.addClass("hidden");
    entry.panel.attr('aria-hidden', 'true');
    entry.activeIndex = -1;
}

function extractTokenMeta(textarea) {
    const value = textarea.value;
    const cursor = textarea.selectionStart ?? value.length;
    const lastComma = value.lastIndexOf(",", cursor - 1);
    const lastNewline = value.lastIndexOf("\n", cursor - 1);
    const start = Math.max(lastComma, lastNewline) + 1;
    const end = cursor;
    if (start > end) return null;
    const raw = value.slice(start, end);
    const leading = raw.match(/^\s*/)?.[0] ?? "";
    const token = raw.trim().toLowerCase();
    return { start, end, leading, tokenLower: token };
}

function applyTagSuggestion(textarea, suggestion) {
    const meta = extractTokenMeta(textarea);
    if (!meta) return;
    const value = textarea.value;
    const before = value.slice(0, meta.start);
    const after = value.slice(meta.end);
    const replacement = `${meta.leading || ""}${suggestion}`;
    const nextValue = before + replacement + after;
    textarea.value = nextValue;
    const caret = before.length + replacement.length;
    textarea.setSelectionRange(caret, caret);
    hideTagSuggestions(textarea);
    textarea.dispatchEvent(new Event("input", { bubbles: true }));
}

const hootsightRoot = window.Hootsight || (window.Hootsight = {});
hootsightRoot.datasetEditor = {
    init() {
        if (datasetEditorInitialized) {
            return;
        }
        datasetEditorInitialized = true;
        init();
    },
    setActive(active) {
        datasetEditorActive = Boolean(active);
        if (!datasetEditorActive) {
            state.ctrlPressed = false;
            Q(document.body).removeClass("crop-enabled");
            stopBuildPolling();
            stopDiscoveryPolling();
        } else {
            if (!datasetEditorInitialized) {
                this.init();
            } else {
                refreshData();
            }
        }
    }
    ,
    destroy() {
        // Gracefully stop activity and clear cached state
        datasetEditorActive = false;
        datasetEditorInitialized = false;
        stopBuildPolling();
        stopDiscoveryPolling();
        pendingBoundsSaves.forEach((t) => clearTimeout(t));
        pendingBoundsSaves.clear();
        pendingAnnotationSaves.forEach((t) => clearTimeout(t));
        pendingAnnotationSaves.clear();
        thumbnailRegistry.forEach((rec, id) => unregisterThumbnail(id));
        thumbnailRegistry.clear();
        autocompleteRegistry.forEach((entry, textarea) => unregisterAutocomplete(textarea));
        autocompleteRegistry.clear();
        // Remove global handlers
        removeAllGlobalEvents();
    }
};

})();
