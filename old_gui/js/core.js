(function(){
  const root = window;
  const hs = root.Hootsight || (root.Hootsight = {});

  const state = {
    schema: null,
    config: null,
    pages: {},
    dirty: false,
    trainingStatus: null,
    trainingTelemetry: null,
    pendingAutoProject: null,
    pendingAutoTrainingId: null,
    lastAutoLoadedTrainingId: null,
    baseFooterStatus: 'Ready',
    baseFooterIsError: false,
    projects: [],
    currentProject: null,
    i18n: {},
    languages: [],
    activeLanguage: 'en',
    validators: [],
    fieldIndex: {},
    modelTypeField: null,
    modelNameField: null,
    dynamicParams: {
      optimizer: [],
      scheduler: [],
      loss: []
    },
    augmentation: {
      selections: {
        train: {},
        val: {}
      },
      custom: {
        train: [],
        val: []
      },
      params: {
        train: {},
        val: {}
      },
      preview: {
        train: null,
        val: null
      }
    },
    docs: {
      initialized: false,
      files: [],
      cache: {},
      currentPath: null,
      loading: false,
      error: null,
      pendingAnchor: null
    },
    updates: {
      pending: false,
      applying: false,
      files: [],
      orphaned: [],
      lastCheckedAt: null,
      lastError: null,
      message: null
    }
  };

  const dom = {
    sidebar: Q('#sidebar'),
    pageContainer: Q('#page-container'),
    pageTitle: Q('#page-title'),
    footerStatus: Q('#footer-status'),
    btnSaveLegacy: Q('#btn-save-config'),
    btnSaveSystem: Q('#btn-save-system-config'),
    btnExport: Q('#btn-export-config'),
    validationSummary: Q('#validation-summary'),
    languageSelect: Q('#language-select'),
    statusBadge: Q('#status-training')
  };

  function log(){
    if(window.console && typeof console.debug === 'function'){
      console.debug('[UI]', ...arguments);
    }
  }

  hs.state = state;
  hs.dom = dom;
  hs.log = log;
  /**
   * Create a card element (Q wrapper).
   * Usage: hs.createCard({tag: 'div', classes: 'card--glass config-field-card'})
   */
  hs.createCard = function(opts){
    opts = opts || {};
    const tag = opts.tag || 'div';
    const classes = (opts.classes || '').trim();
    const classAttr = classes ? `card ${classes}` : 'card';
    const el = Q(`<${tag} class="${classAttr}">`);
    return el;
  };
})();
