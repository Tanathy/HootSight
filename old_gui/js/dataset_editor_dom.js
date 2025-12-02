/**
 * Dataset Editor DOM Generator
 * Generates the entire dataset editor UI structure using qte.js (jQuery-like API)
 * Keeps HTML clean and maintainable, uses chainable DOM manipulation
 */

(function(window) {
    'use strict';

    const DOMGenerator = {
        /**
         * Build the complete dataset editor structure
         * @param {HTMLElement} container - Root container to populate
         * @returns {Object} References to important elements
         */
        build(container) {
            if (!container) {
                throw new Error('Container element required');
            }

            // Clear and setup container
            Q(container).empty().addClass('app-shell');

            // Build main structure
            const $shell = Q(container);
            $shell.append(this._buildHeader());
            $shell.append(this._buildWorkspace());
            $shell.append(this._buildGrid());
            $shell.append(this._buildToast());

            // Collect all element references for easy access
            return this._collectElementReferences(container);
        },

        /**
         * Build header section
         */
        _buildHeader() {
            return Q('<header>', { class: 'app-header' })
                .append(
                    Q('<div>', { class: 'title-group' })
                        .append(Q('<h1>', { 'data-i18n': 'dataset_editor_ui.page_title', text: 'Dataset Builder' }))
                        .append(Q('<div>', { 
                            class: 'subtitle', 
                            'data-i18n': 'dataset_editor_ui.page_description', 
                            text: 'Darkroom for annotation and 1:1 crops' 
                        }))
                )
                .append(
                    Q('<div>', { class: 'project-select' })
                        .append(Q('<label>', { for: 'projectPicker', 'data-i18n': 'dataset_editor_ui.project_label', text: 'Project' }))
                        .append(Q('<select>', { id: 'projectPicker' }))
                        .append(Q('<button>', { id: 'refreshProject', class: 'btn ghost icon', title: 'Rescan folders', text: '⟳' }))
                )
                .append(
                    Q('<div>', { class: 'header-actions' })
                        .append(
                            Q('<div>', { class: 'build-controls' })
                                .append(
                                    Q('<div>', { class: 'dataset-size-control' })
                                        .append(Q('<label>', { for: 'datasetSize', 'data-i18n': 'dataset_editor_ui.crop_size', text: 'Crop size' }))
                                        .append(Q('<select>', { id: 'datasetSize' }))
                                )
                                .append(Q('<button>', { id: 'buildDataset', class: 'btn primary rounded', 'data-i18n': 'dataset_editor_ui.build_dataset', text: 'Build Dataset' }))
                        )
                )
                .getAll()[0];
        },



        /**
         * Build workspace (sidebar + main content)
         */
        _buildWorkspace() {
            const $workspace = Q('<div>', { class: 'workspace' });

            // Folder panel (sidebar)
            const $folderPanel = Q('<aside>', { class: 'panel folder-panel' })
                .append(
                    Q('<div>', { class: 'folder-panel-header' })
                        .append(
                            Q('<div>')
                                .append(Q('<h2>', { 'data-i18n': 'dataset_editor_ui.folders', text: 'Folders' }))
                                .append(Q('<div>', { 
                                    class: 'folder-hint', 
                                    'data-i18n': 'dataset_editor_ui.folder_hint', 
                                    text: 'Hold CTRL to crop thumbs on the fly.' 
                                }))
                        )
                        .append(
                            Q('<div>', { class: 'folder-panel-controls' })
                                .append(Q('<button>', { 
                                    id: 'folderRefresh', 
                                    class: 'btn ghost small', 
                                    'data-i18n': 'dataset_editor_ui.refresh', 
                                    text: 'Refresh' 
                                }))
                                .append(Q('<button>', { 
                                    id: 'folderReset', 
                                    class: 'btn ghost small', 
                                    'data-i18n': 'dataset_editor_ui.reset', 
                                    text: 'Reset' 
                                }))
                        )
                )
                .append(Q('<div>', { id: 'folderTree', class: 'folder-tree' }));

            // Info column
            const $infoColumn = Q('<div>', { class: 'info-column' });

            // Only keep the recommendations block; remove legacy numeric stats panels
            const $statRecommendations = Q('<div>', { class: 'stat-recommendations', id: 'statRecommendations' })
                .append(
                    Q('<div>', { class: 'panel-header' })
                        .append(
                            Q('<div>')
                                .append(Q('<h3>', { 'data-i18n': 'dataset_editor_ui.stats_recommendations_title', text: 'Recommendations' }))
                                .append(Q('<div>', { class: 'panel-note', 'data-i18n': 'dataset_editor_ui.stats_recommendations_note', text: 'Suggested actions to improve dataset balance and quality' }))
                        )
                )
                .append(Q('<div>', { class: 'stat-recommendations-body', id: 'statRecommendationsBody' }));

            const $statsPanel = Q('<section>', { class: 'panel', id: 'statsPanel' })
                .append($statRecommendations.getAll()[0]);

            // Filters panel
            const $filtersPanel = Q('<section>', { class: 'panel filters-panel' })
                // Add a localized header and description for the filters panel
                .append(
                    Q('<div>', { class: 'panel-header filters-header' })
                        .append(
                            Q('<div>')
                                .append(Q('<h3>', { 'data-i18n': 'dataset_editor_ui.filters_title', text: 'Filters' }))
                                .append(Q('<div>', { class: 'panel-note filters-note', 'data-i18n': 'dataset_editor_ui.filters_note', text: 'Filter images by folder, tags, or filename' }))
                        )
                )
                .append(
                    Q('<div>', { class: 'search-box' })
                        .append(Q('<input>', { 
                            id: 'searchInput', 
                            type: 'search',
                            class: 'search-input',
                            'data-i18n-placeholder': 'dataset_editor_ui.search_placeholder', 
                            placeholder: 'Search annotations or filenames' 
                        }))
                        .append(Q('<button>', { 
                            id: 'searchClear', 
                            class: 'btn ghost', 
                            'data-i18n': 'dataset_editor_ui.search_clear', 
                            text: 'Clear' 
                        }))
                )
                .append(
                    Q('<div>', { class: 'pagination-controls' })
                        .append(
                            Q('<label>')
                                .append(Q('<div>', { 'data-i18n': 'dataset_editor_ui.page_size', text: 'Page size' }))
                                .append(Q('<select>', { id: 'pageSize' }))
                        )
                        .append(
                            Q('<div>', { class: 'pager' })
                                .append(Q('<button>', { 
                                    id: 'prevPage', 
                                    class: 'btn ghost', 
                                    'data-i18n': 'dataset_editor_ui.prev', 
                                    text: 'Prev' 
                                }))
                                .append(Q('<div>', { id: 'pageIndicator' }))
                                .append(Q('<button>', { 
                                    id: 'nextPage', 
                                    class: 'btn ghost', 
                                    'data-i18n': 'dataset_editor_ui.next', 
                                    text: 'Next' 
                                }))
                        )
                );

            // Bulk edit panel
            const $bulkPanel = Q('<section>', { class: 'panel bulk-panel' })
                .append(
                    Q('<div>', { class: 'bulk-header panel-header' })
                        .append(
                            Q('<div>')
                                .append(Q('<h3>', { 'data-i18n': 'dataset_editor_ui.bulk_title', text: 'Bulk tag edit' }))
                                .append(Q('<div>', { 
                                    class: 'bulk-note panel-note', 
                                    'data-i18n': 'dataset_editor_ui.bulk_note', 
                                    text: 'Applies to the current filter (search + folder).' 
                                }))
                        )
                        .append(Q('<div>', { id: 'bulkScope', class: 'bulk-scope', text: '0 items' }))
                )
                .append(
                    Q('<div>', { class: 'bulk-inputs' })
                        .append(
                            Q('<div>', { class: 'bulk-field' })
                                .append(Q('<label>', { 
                                    for: 'bulkAdd', 
                                    'data-i18n': 'dataset_editor_ui.bulk_add', 
                                    text: 'Add tags' 
                                }))
                                .append(Q('<input>', { 
                                    id: 'bulkAdd', 
                                    type: 'text', 
                                    'data-i18n-placeholder': 'dataset_editor_ui.bulk_add_placeholder', 
                                    placeholder: 'Comma-separated tags' 
                                }))
                        )
                        .append(
                            Q('<div>', { class: 'bulk-field' })
                                .append(Q('<label>', { 
                                    for: 'bulkRemove', 
                                    'data-i18n': 'dataset_editor_ui.bulk_remove', 
                                    text: 'Remove tags' 
                                }))
                                .append(Q('<input>', { 
                                    id: 'bulkRemove', 
                                    type: 'text', 
                                    'data-i18n-placeholder': 'dataset_editor_ui.bulk_remove_placeholder', 
                                    placeholder: 'Comma-separated tags' 
                                }))
                        )
                        .append(Q('<button>', { 
                            id: 'bulkApply', 
                            class: 'btn ghost', 
                            'data-i18n': 'dataset_editor_ui.bulk_apply', 
                            text: 'Apply to filtered' 
                        }))
                );

            // Group filters and bulk into a single row for side-by-side layout
            const $filtersBulkRow = Q('<div>', { class: 'filters-bulk-row' })
                .append($filtersPanel.getAll()[0])
                .append($bulkPanel.getAll()[0]);

            // Assemble info column
            $infoColumn.append($statsPanel.getAll()[0])
                       .append($filtersBulkRow.getAll()[0]);

            // Assemble workspace
            $workspace.append($folderPanel.getAll()[0])
                     .append($infoColumn.getAll()[0]);

            return $workspace.getAll()[0];
        },

        /**
         * Build a stat item (label + value)
         */
        _buildStatItem(labelKey, valueId, defaultValue) {
            // Use divs instead of spans to allow block-level children (e.g., progress bars)
            return Q('<div>')
                .append(Q('<div>', { 
                    'data-i18n': `dataset_editor_ui.${labelKey}`, 
                    text: labelKey 
                }))
                .append(Q('<div>', { class: 'stat-value', id: valueId, text: defaultValue }));
        },

        /**
         * Build image grid container
         */
        _buildGrid() {
            return Q('<section>', { id: 'itemGrid', class: 'grid' }).getAll()[0];
        },

        /**
         * Build toast notification
         */
        _buildToast() {
            return Q('<div>', { 
                id: 'toast', 
                class: 'toast hidden', 
                role: 'status', 
                'aria-live': 'assertive' 
            }).getAll()[0];
        },

        /**
         * Collect all important element references
         */
        _collectElementReferences(container) {
            return {
                projectPicker: container.querySelector('#projectPicker'),
                refreshProject: container.querySelector('#refreshProject'),
                buildButton: container.querySelector('#buildDataset'),
                datasetSize: container.querySelector('#datasetSize'),
                searchInput: container.querySelector('#searchInput'),
                searchClear: container.querySelector('#searchClear'),
                pageSize: container.querySelector('#pageSize'),
                prevPage: container.querySelector('#prevPage'),
                nextPage: container.querySelector('#nextPage'),
                pageIndicator: container.querySelector('#pageIndicator'),
                grid: container.querySelector('#itemGrid'),
                statsRecommendations: container.querySelector('#statRecommendations'),
                statRecommendationsBody: container.querySelector('#statRecommendationsBody'),
                folderTree: container.querySelector('#folderTree'),
                folderRefresh: container.querySelector('#folderRefresh'),
                folderReset: container.querySelector('#folderReset'),
                bulkAddInput: container.querySelector('#bulkAdd'),
                bulkRemoveInput: container.querySelector('#bulkRemove'),
                bulkApply: container.querySelector('#bulkApply'),
                bulkScope: container.querySelector('#bulkScope'),
                toast: container.querySelector('#toast'),
                buildProgress: container.querySelector('#buildProgress'),
                buildStatusLabel: container.querySelector('#buildStatusLabel'),
                buildEta: container.querySelector('#buildEta'),
                buildProgressFill: container.querySelector('#buildProgressFill'),
                buildProgressStats: container.querySelector('#buildProgressStats'),
                discoveryProgress: container.querySelector('#discoveryProgress'),
                discoveryStatusLabel: container.querySelector('#discoveryStatusLabel'),
                discoveryEta: container.querySelector('#discoveryEta'),
                discoveryProgressFill: container.querySelector('#discoveryProgressFill'),
                discoveryProgressStats: container.querySelector('#discoveryProgressStats'),
            };
        }
    };

    // Export for use
    window.DatasetEditorDOM = DOMGenerator;

})(window);
