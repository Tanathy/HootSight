(function(){
  const hs = window.Hootsight || (window.Hootsight = {});
  const state = hs.state;
  const dom = hs.dom;
  const log = hs.log;
  const text = hs.text || {};
  const t = text.t || ((key, fallback)=> fallback ?? key);
  const applyLocalization = text.applyLocalization || (()=>{});
  const comm = hs.communication || {};
  const markdown = hs.markdown || null;
  const STORAGE_KEYS = {
    trainingId: 'hs.activeTrainingId',
    trainingProject: 'hs.activeTrainingProject'
  };
  const PROJECT_NAME_MIN_LENGTH = 3;
  const PROJECT_NAME_MAX_LENGTH = 64;
  const PROJECT_NAME_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*$/;

  function formatStatValue(value, fallback = ''){
    if(value === undefined || value === null){
      return fallback;
    }
    if(typeof value === 'number'){
      return Number.isFinite(value) ? value.toLocaleString() : fallback;
    }
    if(typeof value === 'string'){
      const trimmed = value.trim();
      return trimmed.length ? trimmed : fallback;
    }
    if(typeof value === 'boolean'){
      return value ? 'true' : 'false';
    }
    if(value && typeof value.toLocaleString === 'function'){
      const localized = value.toLocaleString();
      if(typeof localized === 'string' && localized.trim().length){
        return localized;
      }
    }
    return String(value);
  }

  function appendStatRow(container, label, value, { valueClass } = {}){
    if(!container || !container.append) return;
    const row = Q('<div>');
    Q('<span>').text(`${label}:`).appendTo(row);
    const valueNode = Q('<span>').text(` ${value}`);
    if(valueClass){
      valueNode.addClass(valueClass);
    }
    row.append(valueNode);
    container.append(row);
  }

  function resolveStatusDisplay(status){
    if(status === undefined || status === null){
      return { text: null, className: null };
    }
    const raw = String(status).trim();
    if(!raw){
      return { text: null, className: null };
    }
    const key = raw.toLowerCase().replace(/\s+/g, '_');
    const classSuffix = raw.toLowerCase().replace(/[^a-z0-9]+/g, '-');
    return {
      text: t(`projects_ui.card.status.${key}`, raw),
      className: classSuffix ? `status-${classSuffix}` : null
    };
  }

  function readStoredTrainingContext(){
    try {
      if(typeof window === 'undefined' || !window.localStorage) return null;
      const trainingId = window.localStorage.getItem(STORAGE_KEYS.trainingId);
      if(!trainingId) return null;
      const project = window.localStorage.getItem(STORAGE_KEYS.trainingProject) || null;
      return { trainingId, project };
    } catch (err){
      log('Failed to read training context:', err);
      return null;
    }
  }

  function persistTrainingContext(trainingId, project){
    try {
      if(typeof window === 'undefined' || !window.localStorage) return;
      if(trainingId){
        window.localStorage.setItem(STORAGE_KEYS.trainingId, trainingId);
        if(project){
          window.localStorage.setItem(STORAGE_KEYS.trainingProject, project);
        } else {
          window.localStorage.removeItem(STORAGE_KEYS.trainingProject);
        }
      } else {
        window.localStorage.removeItem(STORAGE_KEYS.trainingId);
        window.localStorage.removeItem(STORAGE_KEYS.trainingProject);
      }
    } catch (err){
      log('Failed to persist training context:', err);
    }
  }

  function clearStoredTrainingContext(){
    persistTrainingContext(null, null);
  }

  const AUGMENTATION_PRESETS = {
    train: [
      {
        key: 'random_resized_crop',
        labelKey: 'augmentation_ui.random_resized_crop',
        descriptionKey: 'augmentation_ui.random_resized_crop_description',
        default: true,
        params: () => ({
          size: getInputSize(),
          scale: [0.8, 1.0],
          ratio: [0.9, 1.1]
        }),
        fields: [
          {
            path: 'size',
            type: 'number',
            min: 32,
            max: 4096,
            step: 1,
            labelKey: 'augmentation_ui.random_resized_crop.size_label',
            descriptionKey: 'augmentation_ui.random_resized_crop.size_description'
          },
          {
            path: 'scale[0]',
            type: 'number',
            min: 0.05,
            max: 1,
            step: 0.01,
            precision: 2,
            labelKey: 'augmentation_ui.random_resized_crop.scale_min_label',
            descriptionKey: 'augmentation_ui.random_resized_crop.scale_min_description'
          },
          {
            path: 'scale[1]',
            type: 'number',
            min: 0.1,
            max: 1.5,
            step: 0.01,
            precision: 2,
            labelKey: 'augmentation_ui.random_resized_crop.scale_max_label',
            descriptionKey: 'augmentation_ui.random_resized_crop.scale_max_description'
          },
          {
            path: 'ratio[0]',
            type: 'number',
            min: 0.3,
            max: 1,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.random_resized_crop.ratio_min_label',
            descriptionKey: 'augmentation_ui.random_resized_crop.ratio_min_description'
          },
          {
            path: 'ratio[1]',
            type: 'number',
            min: 1,
            max: 3,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.random_resized_crop.ratio_max_label',
            descriptionKey: 'augmentation_ui.random_resized_crop.ratio_max_description'
          }
        ]
      },
      {
        key: 'random_horizontal_flip',
        labelKey: 'augmentation_ui.random_horizontal_flip',
        descriptionKey: 'augmentation_ui.random_horizontal_flip_description',
        default: true,
        params: () => ({ p: 0.5 }),
        fields: [
          {
            path: 'p',
            type: 'probability',
            step: 0.05,
            labelKey: 'augmentation_ui.random_horizontal_flip.p_label',
            descriptionKey: 'augmentation_ui.random_horizontal_flip.p_description'
          }
        ]
      },
      {
        key: 'random_vertical_flip',
        labelKey: 'augmentation_ui.random_vertical_flip',
        descriptionKey: 'augmentation_ui.random_vertical_flip_description',
        default: false,
        params: () => ({ p: 0.15 }),
        fields: [
          {
            path: 'p',
            type: 'probability',
            step: 0.05,
            labelKey: 'augmentation_ui.random_vertical_flip.p_label',
            descriptionKey: 'augmentation_ui.random_vertical_flip.p_description'
          }
        ]
      },
      {
        key: 'random_rotation',
        labelKey: 'augmentation_ui.random_rotation',
        descriptionKey: 'augmentation_ui.random_rotation_description',
        default: false,
        params: () => ({ degrees: [-15, 15] }),
        fields: [
          {
            path: 'degrees[0]',
            type: 'number',
            min: -180,
            max: 180,
            step: 1,
            labelKey: 'augmentation_ui.random_rotation.min_label',
            descriptionKey: 'augmentation_ui.random_rotation.min_description'
          },
          {
            path: 'degrees[1]',
            type: 'number',
            min: -180,
            max: 180,
            step: 1,
            labelKey: 'augmentation_ui.random_rotation.max_label',
            descriptionKey: 'augmentation_ui.random_rotation.max_description'
          }
        ]
      },
      {
        key: 'color_jitter',
        labelKey: 'augmentation_ui.color_jitter',
        descriptionKey: 'augmentation_ui.color_jitter_description',
        default: false,
        params: () => ({
          brightness: 0.2,
          contrast: 0.2,
          saturation: 0.2,
          hue: 0.02
        }),
        fields: [
          {
            path: 'brightness',
            type: 'number',
            min: 0,
            max: 2,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.color_jitter.brightness_label',
            descriptionKey: 'augmentation_ui.color_jitter.brightness_description'
          },
          {
            path: 'contrast',
            type: 'number',
            min: 0,
            max: 2,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.color_jitter.contrast_label',
            descriptionKey: 'augmentation_ui.color_jitter.contrast_description'
          },
          {
            path: 'saturation',
            type: 'number',
            min: 0,
            max: 2,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.color_jitter.saturation_label',
            descriptionKey: 'augmentation_ui.color_jitter.saturation_description'
          },
          {
            path: 'hue',
            type: 'number',
            min: 0,
            max: 0.5,
            step: 0.01,
            precision: 2,
            labelKey: 'augmentation_ui.color_jitter.hue_label',
            descriptionKey: 'augmentation_ui.color_jitter.hue_description'
          }
        ]
      },
      {
        key: 'random_grayscale',
        labelKey: 'augmentation_ui.random_grayscale',
        descriptionKey: 'augmentation_ui.random_grayscale_description',
        default: false,
        params: () => ({ p: 0.1 }),
        fields: [
          {
            path: 'p',
            type: 'probability',
            step: 0.05,
            labelKey: 'augmentation_ui.random_grayscale.p_label',
            descriptionKey: 'augmentation_ui.random_grayscale.p_description'
          }
        ]
      },
      {
        key: 'random_erasing',
        labelKey: 'augmentation_ui.random_erasing',
        descriptionKey: 'augmentation_ui.random_erasing_description',
        default: false,
        params: () => ({
          p: 0.25,
          scale: [0.02, 0.2],
          ratio: [0.3, 3.3],
          value: 0,
          inplace: false
        }),
        fields: [
          {
            path: 'p',
            type: 'probability',
            step: 0.05,
            labelKey: 'augmentation_ui.random_erasing.p_label',
            descriptionKey: 'augmentation_ui.random_erasing.p_description'
          },
          {
            path: 'scale[0]',
            type: 'number',
            min: 0.01,
            max: 0.8,
            step: 0.01,
            precision: 2,
            labelKey: 'augmentation_ui.random_erasing.scale_min_label',
            descriptionKey: 'augmentation_ui.random_erasing.scale_min_description'
          },
          {
            path: 'scale[1]',
            type: 'number',
            min: 0.02,
            max: 0.9,
            step: 0.01,
            precision: 2,
            labelKey: 'augmentation_ui.random_erasing.scale_max_label',
            descriptionKey: 'augmentation_ui.random_erasing.scale_max_description'
          },
          {
            path: 'ratio[0]',
            type: 'number',
            min: 0.1,
            max: 1,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.random_erasing.ratio_min_label',
            descriptionKey: 'augmentation_ui.random_erasing.ratio_min_description'
          },
          {
            path: 'ratio[1]',
            type: 'number',
            min: 1,
            max: 10,
            step: 0.1,
            precision: 2,
            labelKey: 'augmentation_ui.random_erasing.ratio_max_label',
            descriptionKey: 'augmentation_ui.random_erasing.ratio_max_description'
          },
          {
            path: 'value',
            type: 'number',
            min: 0,
            max: 1,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.random_erasing.value_label',
            descriptionKey: 'augmentation_ui.random_erasing.value_description'
          },
          {
            path: 'inplace',
            type: 'boolean',
            labelKey: 'augmentation_ui.random_erasing.inplace_label',
            descriptionKey: 'augmentation_ui.random_erasing.inplace_description'
          }
        ]
      },
      {
        key: 'random_perspective',
        labelKey: 'augmentation_ui.random_perspective',
        descriptionKey: 'augmentation_ui.random_perspective_description',
        default: false,
        params: () => ({
          distortion_scale: 0.3,
          p: 0.5
        }),
        fields: [
          {
            path: 'distortion_scale',
            type: 'number',
            min: 0,
            max: 1,
            step: 0.05,
            precision: 2,
            labelKey: 'augmentation_ui.random_perspective.distortion_scale_label',
            descriptionKey: 'augmentation_ui.random_perspective.distortion_scale_description'
          },
          {
            path: 'p',
            type: 'probability',
            step: 0.05,
            labelKey: 'augmentation_ui.random_perspective.p_label',
            descriptionKey: 'augmentation_ui.random_perspective.p_description'
          }
        ]
      }
    ],
    val: [
      {
        key: 'center_crop',
        labelKey: 'augmentation_ui.center_crop',
        descriptionKey: 'augmentation_ui.center_crop_description',
        default: true,
        params: () => ({ size: getInputSize() }),
        fields: [
          {
            path: 'size',
            type: 'number',
            min: 32,
            max: 4096,
            step: 1,
            labelKey: 'augmentation_ui.center_crop.size_label',
            descriptionKey: 'augmentation_ui.center_crop.size_description'
          }
        ]
      },
      {
        key: 'random_horizontal_flip',
        labelKey: 'augmentation_ui.random_horizontal_flip',
        descriptionKey: 'augmentation_ui.random_horizontal_flip_description',
        default: false,
        params: () => ({ p: 0.5 }),
        fields: [
          {
            path: 'p',
            type: 'probability',
            step: 0.05,
            labelKey: 'augmentation_ui.random_horizontal_flip.p_label',
            descriptionKey: 'augmentation_ui.random_horizontal_flip.p_description'
          }
        ]
      },
      {
        key: 'random_rotation',
        labelKey: 'augmentation_ui.random_rotation',
        descriptionKey: 'augmentation_ui.random_rotation_description',
        default: false,
        params: () => ({ degrees: [-5, 5] }),
        fields: [
          {
            path: 'degrees[0]',
            type: 'number',
            min: -180,
            max: 180,
            step: 1,
            labelKey: 'augmentation_ui.random_rotation.min_label',
            descriptionKey: 'augmentation_ui.random_rotation.min_description'
          },
          {
            path: 'degrees[1]',
            type: 'number',
            min: -180,
            max: 180,
            step: 1,
            labelKey: 'augmentation_ui.random_rotation.max_label',
            descriptionKey: 'augmentation_ui.random_rotation.max_description'
          }
        ]
      }
    ]
  };

  const AUGMENTATION_PHASES = ['train', 'val'];

  function getInputSize(){
    const sizeValue = state.config?.training?.input_size;
    const numeric = Number(sizeValue);
    if(Number.isFinite(numeric) && numeric > 0){
      return Math.round(numeric);
    }
    return 224;
  }

  function deepClone(value){
    if(value === null || value === undefined){
      return value;
    }
    try {
      return JSON.parse(JSON.stringify(value));
    } catch {
      return value;
    }
  }

  function ensureAugmentationContainers(){
    if(!state.config){
      state.config = {};
    }
    if(!state.config.training){
      state.config.training = {};
    }
    if(!state.config.training.augmentation){
      state.config.training.augmentation = { train: [], val: [] };
    }
    if(!state.augmentation.params){
      state.augmentation.params = { train: {}, val: {} };
    }
    if(!state.augmentation.preview){
      state.augmentation.preview = { train: null, val: null };
    }
    AUGMENTATION_PHASES.forEach(phase => {
      if(!Array.isArray(state.config.training.augmentation[phase])){
        state.config.training.augmentation[phase] = [];
      }
      if(!state.augmentation.selections[phase]){
        state.augmentation.selections[phase] = {};
      }
      if(!state.augmentation.custom[phase]){
        state.augmentation.custom[phase] = [];
      }
      if(!state.augmentation.params[phase]){
        state.augmentation.params[phase] = {};
      }
      if(!state.augmentation.preview[phase]){
        state.augmentation.preview[phase] = { status: 'idle' };
      }
    });
  }

  function findPreset(phase, key){
    return (AUGMENTATION_PRESETS[phase] || []).find(option => option.key === key);
  }

  function computeDefaultParams(preset){
    if(!preset){
      return {};
    }
    const source = typeof preset.params === 'function' ? preset.params() : preset.params;
    if(!source){
      return {};
    }
    return deepClone(source);
  }

  function tokenizeParamPath(path){
    if(Array.isArray(path)){
      return path;
    }
    if(typeof path !== 'string' || !path.length){
      return [];
    }
    return path.replace(/\]/g, '').split(/[.\[]/).filter(Boolean);
  }

  function readParamPath(target, path){
    const tokens = tokenizeParamPath(path);
    if(!tokens.length){
      return undefined;
    }
    let cursor = target;
    for(const token of tokens){
      if(cursor === undefined || cursor === null){
        return undefined;
      }
      cursor = cursor[token];
    }
    return cursor;
  }

  function writeParamPath(target, path, value){
    const tokens = tokenizeParamPath(path);
    if(!tokens.length){
      return;
    }
    let cursor = target;
    for(let i = 0; i < tokens.length - 1; i += 1){
      const token = tokens[i];
      let next = cursor[token];
      if(next === undefined || next === null){
        const peek = tokens[i + 1];
        next = (/^\d+$/.test(peek)) ? [] : {};
        cursor[token] = next;
      }
      cursor = next;
    }
    cursor[tokens[tokens.length - 1]] = value;
  }

  function ensurePresetParams(phase, key){
    ensureAugmentationContainers();
    const store = state.augmentation.params[phase];
    if(!store[key]){
      const preset = findPreset(phase, key);
      store[key] = computeDefaultParams(preset);
    }
    return store[key];
  }

  function extractParamsFromTransform(transform, preset){
    if(transform && typeof transform.params === 'object' && !Array.isArray(transform.params)){
      return deepClone(transform.params);
    }
    const fallback = {};
    if(transform && typeof transform === 'object'){
      Object.keys(transform).forEach(key => {
        if(key === 'type') return;
        fallback[key] = deepClone(transform[key]);
      });
    }
    if(Object.keys(fallback).length){
      return fallback;
    }
    return computeDefaultParams(preset);
  }

  function normalizeFieldValue(field, raw, current){
    if(field.type === 'boolean'){
      const value = !!raw;
      return { value, changed: value !== !!current };
    }

    let numeric = Number(raw);
    if(!Number.isFinite(numeric)){
      return { value: current, changed: false };
    }

    if(field.type === 'probability'){
      numeric = Math.min(1, Math.max(0, numeric));
    }
    if(field.min !== undefined && numeric < field.min){
      numeric = field.min;
    }
    if(field.max !== undefined && numeric > field.max){
      numeric = field.max;
    }
    if(field.precision !== undefined && Number.isFinite(field.precision) && field.precision >= 0){
      const factor = Math.pow(10, field.precision);
      numeric = Math.round(numeric * factor) / factor;
    }

    return { value: numeric, changed: numeric !== current };
  }

  function getAugmentationFieldValue(phase, key, field){
    ensureAugmentationContainers();
    const store = state.augmentation.params[phase];
    let params = store[key];
    if(!params){
      params = ensurePresetParams(phase, key);
    }
    let value = readParamPath(params, field.path);
    if(value === undefined){
      const preset = findPreset(phase, key);
      const defaults = computeDefaultParams(preset);
      value = readParamPath(defaults, field.path);
    }
    return value;
  }

  function setAugmentationFieldValue(phase, key, field, rawValue){
    ensureAugmentationContainers();
    const preset = findPreset(phase, key);
    if(!preset){
      return undefined;
    }
    const params = ensurePresetParams(phase, key);
    const current = readParamPath(params, field.path);
    const { value, changed } = normalizeFieldValue(field, rawValue, current);
    if(!changed){
      return current;
    }
    writeParamPath(params, field.path, value);
    applyAugmentationConfig(phase, { silent: false });
    return value;
  }

  function getAugmentationOptions(phase){
    return (AUGMENTATION_PRESETS[phase] || []).map(option => ({ ...option }));
  }

  function getAugmentationSelections(phase){
    ensureAugmentationContainers();
    return state.augmentation.selections[phase];
  }

  function getCustomAugmentationsInternal(phase){
    ensureAugmentationContainers();
    return state.augmentation.custom[phase];
  }

  function listCustomAugmentations(phase){
    return getCustomAugmentationsInternal(phase).map(item => deepClone(item));
  }

  function syncAugmentationsFromConfig(){
    ensureAugmentationContainers();
    const augmentationConfig = state.config.training.augmentation || {};
    AUGMENTATION_PHASES.forEach(phase => {
      const selections = {};
      const custom = [];
      const paramsStore = {};
      const previousStore = state.augmentation.params[phase] || {};
      const transforms = Array.isArray(augmentationConfig[phase]) ? augmentationConfig[phase] : [];
      transforms.forEach(transform => {
        if(!transform || typeof transform !== 'object') return;
        const type = (transform.type || '').toLowerCase();
        if(!type) return;
        if(type === 'to_tensor' || type === 'normalize') return;
        const presetMatch = findPreset(phase, type);
        if(presetMatch){
          selections[type] = true;
          paramsStore[type] = extractParamsFromTransform(transform, presetMatch);
        } else {
          custom.push(deepClone(transform));
        }
      });
      (AUGMENTATION_PRESETS[phase] || []).forEach(option => {
        if(!(option.key in selections)){
          selections[option.key] = !!option.default;
        }
        if(!paramsStore[option.key]){
          if(previousStore[option.key]){
            paramsStore[option.key] = deepClone(previousStore[option.key]);
          } else {
            paramsStore[option.key] = computeDefaultParams(option);
          }
        }
      });
      state.augmentation.selections[phase] = selections;
      state.augmentation.custom[phase] = custom;
      state.augmentation.params[phase] = paramsStore;
      applyAugmentationConfig(phase, { silent: true });
    });
  }

  function applyAugmentationConfig(phase, { silent } = {}){
    ensureAugmentationContainers();
    const selections = getAugmentationSelections(phase);
    const presets = AUGMENTATION_PRESETS[phase] || [];
    const orderedTransforms = [];
    const deferredTransforms = [];
    const paramStore = state.augmentation.params[phase] || {};

    presets.forEach(option => {
      if(selections[option.key]){
        const stored = paramStore[option.key];
        const paramsSource = stored && typeof stored === 'object' ? stored : computeDefaultParams(option);
        const params = paramsSource && Object.keys(paramsSource).length ? deepClone(paramsSource) : null;
        const transform = { type: option.key };
        if(params && Object.keys(params).length){
          transform.params = params;
        }
        if(option.key === 'random_resized_crop'){
          deferredTransforms.push(transform);
        } else {
          orderedTransforms.push(transform);
        }
      }
    });

    getCustomAugmentationsInternal(phase).forEach(item => {
      const customTransform = deepClone(item);
      if((customTransform.type || '').toLowerCase() === 'random_resized_crop'){
        deferredTransforms.push(customTransform);
      } else {
        orderedTransforms.push(customTransform);
      }
    });

    const transforms = orderedTransforms.concat(deferredTransforms);
    transforms.push({ type: 'to_tensor' });

    const normalizeCfg = state.config?.training?.normalize;
    if(normalizeCfg && Array.isArray(normalizeCfg.mean) && Array.isArray(normalizeCfg.std)){
      transforms.push({
        type: 'normalize',
        params: {
          mean: deepClone(normalizeCfg.mean),
          std: deepClone(normalizeCfg.std)
        }
      });
    }

    state.config.training.augmentation[phase] = transforms;

    if(!silent){
      updateAugmentationPreviewState(phase, {
        status: 'idle',
        message: null,
        original_image: null,
        augmented_image: null,
        image_path: ''
      });
      markDirty();
    }
  }

  function updateAugmentationPreviewState(phase, nextState){
    ensureAugmentationContainers();
    const current = state.augmentation.preview[phase] || {};
    state.augmentation.preview[phase] = { ...current, ...nextState };
    renderAugmentationPreview(phase);
  }

  function renderAugmentationPreview(phase){
    const container = document.getElementById(`augmentation-preview-${phase}`);
    if(!container) return;
    const wrap = Q(container);
    const preview = (state.augmentation.preview && state.augmentation.preview[phase]) || { status: 'idle' };
    const status = preview.status || 'idle';

    wrap.html('');

    if(status === 'loading'){
      wrap.append(Q('<div class="augmentation-preview-message loading">').text(t('augmentation_ui.preview_loading', 'Generating preview...')));
      return;
    }

    if(status === 'error'){
      const message = preview.message || t('augmentation_ui.preview_generic_error', 'Failed to generate preview.');
      wrap.append(Q('<div class="augmentation-preview-message error">').text(message));
      return;
    }

    if(status !== 'ready' || !preview.augmented_image){
      const idleMessage = state.currentProject
        ? t('augmentation_ui.preview_idle', 'Click Check Preview to see the augmented image.')
        : t('augmentation_ui.preview_no_project', 'Load a project to preview augmentations.');
      wrap.append(Q('<div class="augmentation-preview-message idle">').text(idleMessage));
      return;
    }

    const grid = Q('<div class="augmentation-preview-grid">');
    const originalColumn = Q('<div class="augmentation-preview-column">');
    Q('<div class="augmentation-preview-label">')
      .text(t('augmentation_ui.preview_original_label', 'Original'))
      .appendTo(originalColumn);
    originalColumn.append(
      Q('<img class="augmentation-preview-image">').attr('src', `data:image/png;base64,${preview.original_image}`)
    );

    const augmentedColumn = Q('<div class="augmentation-preview-column">');
    Q('<div class="augmentation-preview-label">')
      .text(t('augmentation_ui.preview_augmented_label', 'Augmented'))
      .appendTo(augmentedColumn);
    augmentedColumn.append(
      Q('<img class="augmentation-preview-image">').attr('src', `data:image/png;base64,${preview.augmented_image}`)
    );

    grid.append(originalColumn, augmentedColumn);
    wrap.append(grid);

    if(preview.image_path){
      Q('<div class="augmentation-preview-meta">')
        .text(`${t('augmentation_ui.preview_image_path_label', 'Image path')}: ${preview.image_path}`)
        .appendTo(wrap);
    }
  }

  async function requestAugmentationPreview(phase){
    ensureAugmentationContainers();

    if(!state.currentProject){
      updateAugmentationPreviewState(phase, {
        status: 'error',
        message: t('augmentation_ui.preview_no_project', 'Load a project to preview augmentations.')
      });
      return;
    }

    const transformList = state.config?.training?.augmentation?.[phase];
    if(!Array.isArray(transformList) || !transformList.length){
      updateAugmentationPreviewState(phase, {
        status: 'error',
        message: t('augmentation_ui.preview_empty_pipeline', 'Configure at least one transform to preview.')
      });
      return;
    }

    updateAugmentationPreviewState(phase, { status: 'loading', message: null });

    try {
      const response = await comm.previewAugmentation(state.currentProject.name, {
        phase,
        transforms: transformList
      });

      if(!response || response.status !== 'success'){
        const message = response?.message || t('augmentation_ui.preview_generic_error', 'Failed to generate preview.');
        updateAugmentationPreviewState(phase, { status: 'error', message });
        if(response?.message){
          log('Augmentation preview error:', response.message);
        }
        return;
      }

      updateAugmentationPreviewState(phase, {
        status: 'ready',
        message: null,
        original_image: response.original_image,
        augmented_image: response.augmented_image,
        image_path: response.image_path || ''
      });
      setStatus(t('status.augmentation_preview_ready', 'Augmentation preview generated'));
    } catch (err){
      const fallback = t('augmentation_ui.preview_generic_error', 'Failed to generate preview.');
      updateAugmentationPreviewState(phase, {
        status: 'error',
        message: err?.message || fallback
      });
      log(err);
    }
  }

  function isAugmentationEnabled(phase, key){
    const selections = getAugmentationSelections(phase);
    return !!selections[key];
  }

  function setAugmentationEnabled(phase, key, enabled){
    const selections = getAugmentationSelections(phase);
    const nextValue = !!enabled;
    if(selections[key] === nextValue){
      return;
    }
    selections[key] = nextValue;
    if(nextValue){
      ensurePresetParams(phase, key);
    }
    applyAugmentationConfig(phase, { silent: false });
  }

  function getCustomAugmentations(phase){
    return listCustomAugmentations(phase);
  }

  function applyFooterStatus(){
    if(!dom.footerStatus || !dom.footerStatus.text) return;
    const statusData = state.trainingStatus;
    if(statusData && statusData.training_id){
      const project = statusData.project || statusData.training_id;
      const epoch = formatProgress(statusData.current_epoch, statusData.total_epochs);
      const step = formatProgress(statusData.current_step, statusData.total_steps);
      const template = t('status_graph.footer_training', 'Training {project} — Epoch {epoch} • Step {step}');
      dom.footerStatus.text(
        template
          .replace('{project}', project || '—')
          .replace('{epoch}', epoch)
          .replace('{step}', step)
      );
      dom.footerStatus.removeClass('error-text');
      return;
    }

    const fallback = state.baseFooterStatus || t('footer.ready', 'Ready');
    dom.footerStatus.text(fallback);
    if(state.baseFooterIsError){
      dom.footerStatus.addClass('error-text');
    } else {
      dom.footerStatus.removeClass('error-text');
    }
  }

  function setStatus(message, isError = false){
    state.baseFooterStatus = message;
    state.baseFooterIsError = !!isError;
    applyFooterStatus();
  }

  function updateTrainingBadge(){
    const badge = dom.statusBadge;
    if(!badge || !badge.addClass) return;
    const statusData = state.trainingStatus;
    if(statusData && statusData.training_id){
      const template = t('status_graph.badge_training', 'Training: {project}');
      const label = template.replace('{project}', statusData.project || statusData.training_id || '—');
      badge.removeClass('hidden');
      badge.text(label);
      if(statusData.status){
        badge.attr('data-status', statusData.status);
      } else {
        badge.removeAttr('data-status');
      }
    } else {
      badge.addClass('hidden');
      badge.text('');
      badge.removeAttr('data-status');
    }
  }

  function showValidationSummary(errors){
    const box = dom.validationSummary;
    if(!box || !box.addClass) return;
    if(!errors.length){
      box.removeClass('active').text('');
      return;
    }
    const parts = errors.map(e=>`<span>${e.path}: ${e.message}</span>`).join(' | ');
    box.addClass('active').html(parts);
  }

  function markDirty(){
    state.dirty = true;
    if(dom.btnSaveLegacy && dom.btnSaveLegacy.elements && dom.btnSaveLegacy.elements.length){
      dom.btnSaveLegacy.prop('disabled', false);
    }
  }

  function navInit(){
    Q('.nav-item').on('click', function(){
      const page = this.getAttribute('data-page');
      hs.components.showPage(page);
      Q('.nav-item').removeClass('active');
      Q(this).addClass('active');
    });
  }

  async function switchLanguage(langCode){
    try {
      setStatus(t('status.switching_language', 'Switching language...'));
      const payload = await comm.switchLanguage(langCode);
      state.i18n = payload.localization || {};
      state.activeLanguage = langCode;
      applyLocalization();
      setStatus(t('status.language_switched', 'Language switched successfully'));
      setTimeout(()=> window.location.reload(), 200);
    } catch (err) {
      setStatus(t('status.language_switch_failed', 'Language switch failed'), true);
      log(err);
    }
  }

  function normalizeLanguageList(raw){
    const normalized = new Map();
    const list = Array.isArray(raw) ? raw : [];

    list.forEach(item => {
      if(!item) return;
      if(typeof item === 'string'){
        const code = item.toLowerCase();
        if(!code) return;
        if(!normalized.has(code)){
          normalized.set(code, { code, name: code.toUpperCase() });
        }
        return;
      }
      const code = (item.code || item.lang || item.id || '').toString().toLowerCase();
      if(!code) return;
      const candidateName =
        (typeof item.name === 'string' && item.name.trim())
          ? item.name.trim()
          : (typeof item.language_name === 'string' && item.language_name.trim()
              ? item.language_name.trim()
              : code.toUpperCase());
      const existing = normalized.get(code);
      if(!existing || existing.name === existing.code.toUpperCase()){
        normalized.set(code, { code, name: candidateName });
      }
    });

    if(!normalized.size){
      normalized.set('en', { code: 'en', name: 'English' });
    }

    return Array.from(normalized.values()).sort((a, b) => {
      const nameCompare = a.name.localeCompare(b.name, undefined, { sensitivity: 'base' });
      return nameCompare || a.code.localeCompare(b.code);
    });
  }

  async function initLanguageSelector(){
    const container = dom.languageSelect;
    if(!container || !container.get()) return;
    const containerEl = container.get();
    const languageTitle = t('ui.language_select_title', 'Select language');
    if(languageTitle){
      container.attr('title', languageTitle);
      container.attr('aria-label', languageTitle);
      if(containerEl){
        containerEl.setAttribute('title', languageTitle);
        containerEl.setAttribute('aria-label', languageTitle);
      }
    }

    try {
      const data = state.languages.length ? { languages: state.languages, active: state.activeLanguage } : await comm.fetchLocalization();
      const languageEntries = normalizeLanguageList(data.languages);
      let current = (data.active || 'en').toLowerCase();

      if(!languageEntries.some(entry => entry.code === current)){
        current = languageEntries[0].code;
      }

      // Clear existing content
      container.html('');

      // Create custom dropdown
      const dropdownContainer = Q('<div class="custom-dropdown">');
      const dropdownButton = Q('<div class="custom-dropdown-button">');
      const dropdownValue = Q('<span class="custom-dropdown-value">');
      const dropdownArrow = Q('<span class="custom-dropdown-arrow">▼</span>');

      dropdownButton.append(dropdownValue, dropdownArrow);

      const dropdownList = Q('<div class="custom-dropdown-list">');

      // Find current language name
      const currentEntry = languageEntries.find(entry => entry.code === current);
      const currentName = currentEntry ? currentEntry.name : current.toUpperCase();
      dropdownValue.text(currentName);

      languageEntries.forEach(entry => {
        const optionEl = Q('<div class="custom-dropdown-option">').text(entry.name).attr('data-value', entry.code);
        if(entry.code === current) optionEl.addClass('selected');
        optionEl.on('click', (e) => {
          e.stopPropagation();
          dropdownValue.text(entry.name);
          dropdownContainer.removeClass('open');
          const newLang = entry.code;
          if(newLang && newLang !== current){
            current = newLang;
            switchLanguage(newLang);
          }
          // Update selected state
          dropdownList.find('.custom-dropdown-option').removeClass('selected');
          optionEl.addClass('selected');
        });
        dropdownList.append(optionEl);
      });

      dropdownButton.on('click', (e) => {
        e.stopPropagation();
        const isOpen = dropdownContainer.hasClass('open');
        // Close all other dropdowns first
        Q('.custom-dropdown').removeClass('open');
        if(!isOpen){
          dropdownContainer.addClass('open');
        }
      });

      dropdownContainer.append(dropdownButton, dropdownList);
      container.append(dropdownContainer);

    } catch (err) {
      log('Failed to load languages:', err);
    }
  }

  async function refreshProjects(){
    try {
      const data = await comm.fetchProjects();
      state.projects = data.projects || [];
      const cont = Q('#projects-cards');
      if(!cont.elements.length) return;
      cont.html('');

      if(!state.projects.length){
        const empty = Q('<div class="projects-empty">');
        Q('<strong>').text(t('projects_ui.empty_title', 'No projects yet')).appendTo(empty);
        Q('<p>').text(t('projects_ui.empty_message', 'Use Create New Project to scaffold dataset, data source, and model folders.')).appendTo(empty);
        const actionBtn = Q('<button type="button" class="primary projects-empty-action">')
          .text(t('projects_ui.toolbar_create', 'Create New Project'))
          .on('click', () => showProjectCreateDialog());
        empty.append(actionBtn);
        cont.append(empty);
        return;
      }

      const activeTrainingId = state.trainingStatus?.training_id || null;
      const activeTrainingProject = activeTrainingId ? state.trainingStatus?.project : null;
      const activeTrainingStatus = state.trainingStatus?.status || '';
      const notAvailableText = t('ui.not_available', 'N/A');
      const unknownValueText = t('ui.unknown', 'Unknown');
      const projectCardLabels = {
        images: t('projects_ui.card.images', 'Images'),
        labels: t('projects_ui.card.labels', 'Labels'),
        balanceScore: t('projects_ui.card.balance_score', 'Balance Score'),
        balanceStatus: t('projects_ui.card.balance_status', 'Balance'),
        datasetType: t('projects_ui.card.dataset_type', 'Dataset Type')
      };

      state.projects.forEach(project => {
        const isActiveTraining = !!(activeTrainingId && activeTrainingProject === project.name);
        const card = Q('<div class="card">');
        if(isActiveTraining){
          card.addClass('training-active');
        }
        Q('<h3>').text(project.name).appendTo(card);

        const info = Q('<div class="project-info">');
        const imageCountValue = project.image_count ?? project.images;
        appendStatRow(info, projectCardLabels.images, formatStatValue(imageCountValue, notAvailableText));

        const labelCountValue = Array.isArray(project.labels) ? project.labels.length : project.labels;
        appendStatRow(info, projectCardLabels.labels, formatStatValue(labelCountValue, notAvailableText));

        if(project.balance_score !== undefined && project.balance_score !== null){
          appendStatRow(info, projectCardLabels.balanceScore, formatStatValue(project.balance_score, notAvailableText));
        }

        if(project.balance_status){
          const balanceDisplay = resolveStatusDisplay(project.balance_status);
          const balanceValue = balanceDisplay.text || formatStatValue(project.balance_status, notAvailableText);
          appendStatRow(info, projectCardLabels.balanceStatus, balanceValue, {
            valueClass: balanceDisplay.className || undefined
          });
        }

        appendStatRow(info, projectCardLabels.datasetType, formatStatValue(project.dataset_type, unknownValueText));
        if(isActiveTraining){
          const trainingLabel = t('ui.training_in_progress', 'Training in progress');
          const statusLabel = activeTrainingStatus ? ` (${activeTrainingStatus})` : '';
          Q('<div class="training-state">').text(trainingLabel + statusLabel).appendTo(info);
        }
        card.append(info);

        if(state.currentProject && state.currentProject.name === project.name){
          const indicator = Q('<div class="current-project-badge">').text(t('ui.current_project', 'CURRENT PROJECT'));
          card.append(indicator);
        }

        const actionsWrap = Q('<div class="actions">');
        const loadBtn = Q('<button type="button" class="secondary">')
          .text(t('ui.load', 'Load'))
          .on('click', () => loadProject(project.name));
        const trainBtn = Q('<button type="button">')
          .text(t('ui.start_training', 'Start Training'))
          .on('click', () => startTraining(project.name));
        const stopBtn = Q('<button type="button" class="danger">')
          .text(t('ui.stop_training', 'Stop Training'))
          .on('click', () => stopTraining(activeTrainingId));

        if(!state.currentProject || state.currentProject.name !== project.name){
          trainBtn.prop('disabled', true);
          trainBtn.attr('title', t('ui.load_project_first', 'Please load a project first from the Projects tab.'));
        }

        if(isActiveTraining){
          trainBtn.prop('disabled', true);
          stopBtn.prop('disabled', false);
          stopBtn.removeAttr('title');
        } else {
          stopBtn.prop('disabled', true);
          stopBtn.attr('title', t('ui.stop_training_disabled', 'No active training for this project.'));
        }

        actionsWrap.append(loadBtn, trainBtn, stopBtn);
        card.append(actionsWrap);
        cont.append(card);
      });

      if(state.pendingAutoProject && state.pendingAutoTrainingId && state.lastAutoLoadedTrainingId !== state.pendingAutoTrainingId){
        const pendingProject = state.pendingAutoProject;
        const pendingTrainingId = state.pendingAutoTrainingId;
        const exists = state.projects.some(project => project.name === pendingProject);
        if(exists){
          state.pendingAutoProject = null;
          state.pendingAutoTrainingId = null;
          state.lastAutoLoadedTrainingId = pendingTrainingId;
          loadProject(pendingProject).catch(err => log('Auto-load project failed:', err));
        }
      }
    } catch (err){
      setStatus(t('status.project_load_failed', 'Project load failed'));
      log(err);
    }
  }

  function closeProjectCreateDialog(){
    const overlay = document.getElementById('project-create-overlay');
    if(!overlay) return;
    const escHandler = overlay._escHandler;
    if(typeof escHandler === 'function'){
      document.removeEventListener('keydown', escHandler);
    }
    overlay.remove();
  }

  function showProjectCreateDialog(){
    const existing = document.getElementById('project-create-overlay');
    if(existing){
      const inputEl = existing.querySelector('input[name="project-name"]');
      if(inputEl){
        inputEl.focus();
        inputEl.select();
      }
      return;
    }

    const overlay = Q('<div class="modal-overlay" id="project-create-overlay">');
    const dialog = Q('<div class="modal">');

    Q('<h3>').text(t('projects_ui.create_title', 'Create new project')).appendTo(dialog);
    const description = t('projects_ui.create_description', 'Name your project to scaffold dataset, data source, model, and heatmap folders.');
    if(description){
      Q('<p class="modal-description muted">').text(description).appendTo(dialog);
    }

    const form = Q('<form class="modal-form" id="project-create-form">');
    const field = Q('<div class="modal-field">');
    Q('<label for="project-create-name">').text(t('projects_ui.create_name_label', 'Project name')).appendTo(field);
    const input = Q('<input type="text" id="project-create-name" name="project-name" autocomplete="off" spellcheck="false">')
      .attr('placeholder', t('projects_ui.create_name_placeholder', 'e.g. wildlife_classification'));
    field.append(input);
    Q('<small class="modal-supporting">').text(t('projects_ui.create_name_hint', 'Use letters, numbers, hyphens, and underscores only.')).appendTo(field);
    form.append(field);

    const errorBox = Q('<div class="modal-error" role="alert" aria-live="polite">');
    form.append(errorBox);

    const actionsRow = Q('<div class="modal-actions">');
    const cancelBtn = Q('<button type="button" class="secondary" data-action="cancel">')
      .text(t('projects_ui.create_cancel', 'Cancel'))
      .on('click', () => closeProjectCreateDialog());
    const submitBtn = Q('<button type="submit" class="primary">')
      .text(t('projects_ui.create_submit', 'Create'));
    actionsRow.append(cancelBtn, submitBtn);
    form.append(actionsRow);

    form.on('submit', handleProjectCreateSubmit);
    dialog.append(form);
    overlay.append(dialog);

    overlay.on('click', (event) => {
      if(event.target === overlay.elements[0]){
        closeProjectCreateDialog();
      }
    });

    const escHandler = (event) => {
      if(event.key === 'Escape'){
        event.preventDefault();
        closeProjectCreateDialog();
      }
    };
    overlay.elements[0]._escHandler = escHandler;
    document.addEventListener('keydown', escHandler);

    Q('body').append(overlay);
    setTimeout(() => {
      const focusEl = document.getElementById('project-create-name');
      if(focusEl){
        focusEl.focus();
        focusEl.select();
      }
    }, 30);
  }

  async function handleProjectCreateSubmit(event){
    event.preventDefault();
    const overlay = document.getElementById('project-create-overlay');
    if(!overlay) return;

    const form = overlay.querySelector('form');
    const input = overlay.querySelector('input[name="project-name"]');
    const errorBox = overlay.querySelector('.modal-error');
    const name = input ? input.value.trim() : '';

    if(errorBox){
      errorBox.textContent = '';
    }

    if(!name){
      const message = t('projects_ui.create_validation_required', 'Project name is required.');
      if(errorBox) errorBox.textContent = message;
      if(input) input.focus();
      return;
    }

    if(name.length < PROJECT_NAME_MIN_LENGTH || name.length > PROJECT_NAME_MAX_LENGTH){
      const message = t('projects_ui.create_validation_length', 'Project name must be between {min} and {max} characters.')
        .replace('{min}', PROJECT_NAME_MIN_LENGTH)
        .replace('{max}', PROJECT_NAME_MAX_LENGTH);
      if(errorBox) errorBox.textContent = message;
      if(input) input.focus();
      return;
    }

    if(!PROJECT_NAME_PATTERN.test(name)){
      const message = t('projects_ui.create_validation_pattern', 'Use letters, numbers, hyphens, and underscores only.');
      if(errorBox) errorBox.textContent = message;
      if(input) input.focus();
      return;
    }

    if(Array.isArray(state.projects) && state.projects.some(project => project?.name === name)){
      const message = t('projects_ui.create_error_exists', 'A project with this name already exists.');
      if(errorBox) errorBox.textContent = message;
      if(input) input.focus();
      return;
    }

    const submitBtn = form ? form.querySelector('button[type="submit"]') : null;
    const cancelBtn = form ? form.querySelector('button[data-action="cancel"]') : null;
    let originalSubmitText = '';
    if(submitBtn){
      originalSubmitText = submitBtn.textContent || '';
      submitBtn.disabled = true;
      submitBtn.textContent = t('projects_ui.create_creating', 'Creating...');
    }
    if(cancelBtn){
      cancelBtn.disabled = true;
    }

    try {
      const response = await comm.createProject(name);
      if(response?.status === 'success'){
        const successMessage = (response.message || t('projects_ui.create_success_status', 'Project {name} created.')).replace('{name}', name);
        setStatus(successMessage);
        closeProjectCreateDialog();
        try {
          await refreshProjects();
        } catch (refreshErr){
          log('Project refresh failed after creation:', refreshErr);
        }
        try {
          await loadProject(name);
        } catch (loadErr){
          log('Auto-load of new project failed:', loadErr);
        }
        return;
      }

      const failureMessage = response?.message || t('projects_ui.create_error_unknown', 'Project creation failed.');
      if(errorBox){
        errorBox.textContent = failureMessage;
      }
      setStatus(failureMessage, true);
    } catch (err){
      const networkMessage = t('projects_ui.create_network_error', 'Network request failed.');
      if(errorBox){
        errorBox.textContent = networkMessage;
      }
      setStatus(networkMessage, true);
      log('Project creation failed:', err);
    } finally {
      if(document.getElementById('project-create-overlay')){
        if(submitBtn){
          submitBtn.disabled = false;
          submitBtn.textContent = originalSubmitText || t('projects_ui.create_submit', 'Create');
        }
        if(cancelBtn){
          cancelBtn.disabled = false;
        }
      }
    }
  }

  async function loadProject(projectName){
    try {
      setStatus(t('status.project_loading', 'Loading project {projectName}...').replace('{projectName}', projectName));
      const project = state.projects.find(p => p.name === projectName);
      if(!project){
        throw new Error(`Project ${projectName} not found in projects list`);
      }

      state.currentProject = project;

      try {
        const res = await comm.fetchProjectConfig(projectName);
        if(res && res.config){
          state.config = res.config;
          syncAugmentationsFromConfig();
          setStatus(t('status.project_loaded_custom', 'Loaded project {projectName} with custom config').replace('{projectName}', projectName));
        } else {
          throw new Error('Project config missing');
        }
      } catch (configErr){
        const fallback = await comm.fetchConfig();
        const config = fallback?.config || fallback || {};
        state.config = config;
        syncAugmentationsFromConfig();
        if(state.currentProject.labels){
          state.config.labels = state.currentProject.labels;
        }
        log('Project config not found, using global config. Reason:', configErr.message);
        setStatus(t('status.project_loaded_defaults', 'Loaded project {projectName} with global defaults').replace('{projectName}', projectName));
      }

      refreshProjects();

      if(Q('.nav-item[data-page="dataset"]').hasClass('active')){
        state.pages.dataset = null;
        hs.components.showPage('dataset');
      }

      state.pages = {};
      if(Q('.nav-item[data-page="training"]').hasClass('active')){
        hs.components.showPage('training');
      }

      if(state.trainingStatus && state.trainingStatus.training_id && state.trainingStatus.project === projectName){
        state.lastAutoLoadedTrainingId = state.trainingStatus.training_id;
      }

      // Refresh page actions to update button states
      if(window.app && window.app.refreshCurrentPageActions){
        window.app.refreshCurrentPageActions();
      }
    } catch (err){
      setStatus(t('status.project_load_error', 'Failed to load project {projectName}: {error}').replace('{projectName}', projectName).replace('{error}', err.message), true);
      log(err);
    }
  }
  async function saveTrainingConfig(){
    if(!state.currentProject){
      setStatus(t('status.no_project_loaded', 'No project loaded'), true);
      return;
    }

    collectCurrentValues();
    const errs = runValidation();
    if(errs.length){
      setStatus(t('status.validation_errors', 'Fix validation errors before saving'), true);
      return;
    }
    const btn = Q('#btn-save-training-config');
    try {
      setStatus(t('status.saving_training_config', 'Saving training config...'));
      if(btn && btn.elements && btn.elements.length){
        btn.prop('disabled', true);
      }
      await comm.saveProjectConfig(state.currentProject.name, state.config);
      state.dirty = false;
      setStatus(t('status.training_config_saved', 'Training config saved'));
      showValidationSummary([]);
    } catch (err){
      setStatus(t('status.save_failed', 'Save failed'), true);
    } finally {
      if(btn && btn.elements && btn.elements.length){
        btn.prop('disabled', false);
      }
    }
  }

  async function refreshDatasetInfo(){
    try {
      const cont = Q('#dataset-cards');
      if(!cont.elements.length){
        return;
      }
      cont.html('');

      if(!state.currentProject){
        const errorCard = Q('<div class="card">');
        Q('<h2 class="card-header">').text(t('ui.no_project_loaded', 'No Project Loaded')).appendTo(errorCard);
        const errorBody = Q('<div class="card-body">');
        Q('<p>').text(t('ui.load_project_first', 'Please load a project first from the Projects tab.')).appendTo(errorBody);
        const goProjects = Q('<button type="button" class="secondary">')
          .text(t('ui.go_to_projects', 'Go to Projects'))
          .on('click', () => Q('.nav-item[data-page="projects"]').click());
        errorBody.append(goProjects);
        errorCard.append(errorBody);
        cont.append(errorCard);
        return;
      }

      const projectInfo = state.currentProject;
      const notAvailableText = t('ui.not_available', 'N/A');
      const unknownValueText = t('ui.unknown', 'Unknown');
      const datasetSummaryLabels = {
        project: t('dataset_ui.summary.project', 'Project'),
        datasetType: t('dataset_ui.summary.dataset_type', 'Dataset Type'),
        totalImages: t('dataset_ui.summary.total_images', 'Total Images'),
        totalLabels: t('dataset_ui.summary.total_labels', 'Total Labels'),
        balanceStatus: t('dataset_ui.summary.balance_status', 'Balance Status'),
        balanceScore: t('dataset_ui.summary.balance_score', 'Balance Score'),
        imagesPerLabelIdeal: t('dataset_ui.summary.images_per_label_ideal', 'Images per Label (Ideal)'),
        minImages: t('dataset_ui.summary.min_images', 'Min Images'),
        maxImages: t('dataset_ui.summary.max_images', 'Max Images'),
        maxMinRatio: t('dataset_ui.summary.max_min_ratio', 'Max/Min Ratio')
      };
      const tableHeaders = {
        label: t('dataset_ui.table.label', 'Label'),
        count: t('dataset_ui.table.count', 'Count'),
        percentage: t('dataset_ui.table.percentage', 'Percentage')
      };

      const overviewCard = Q('<div class="card">');
      Q('<h2 class="card-header">').text(t('ui.dataset_overview', 'Dataset Overview')).appendTo(overviewCard);
      const overviewBody = Q('<div class="card-body column gap">');
      appendStatRow(overviewBody, datasetSummaryLabels.project, formatStatValue(projectInfo.name, unknownValueText));
      appendStatRow(overviewBody, datasetSummaryLabels.datasetType, formatStatValue(projectInfo.dataset_type, unknownValueText));
      appendStatRow(overviewBody, datasetSummaryLabels.totalImages, formatStatValue(projectInfo.image_count, notAvailableText));
      const datasetLabelCount = Array.isArray(projectInfo.labels) ? projectInfo.labels.length : projectInfo.labels;
      appendStatRow(overviewBody, datasetSummaryLabels.totalLabels, formatStatValue(datasetLabelCount, notAvailableText));
      const datasetStatusDisplay = resolveStatusDisplay(projectInfo.balance_status);
      const datasetStatusValue = datasetStatusDisplay.text || formatStatValue(projectInfo.balance_status, notAvailableText);
      appendStatRow(overviewBody, datasetSummaryLabels.balanceStatus, datasetStatusValue, {
        valueClass: datasetStatusDisplay.className || undefined
      });
      appendStatRow(overviewBody, datasetSummaryLabels.balanceScore, formatStatValue(projectInfo.balance_score, notAvailableText));
      overviewCard.append(overviewBody);
      cont.append(overviewCard);

      if(projectInfo.balance_analysis){
        const balanceCard = Q('<div class="card">');
        Q('<h3 class="card-header">').text(t('ui.balance_analysis', 'Balance Analysis')).appendTo(balanceCard);
        const balanceBody = Q('<div class="card-body column gap">');
        const analysis = projectInfo.balance_analysis;
        appendStatRow(balanceBody, datasetSummaryLabels.imagesPerLabelIdeal, formatStatValue(analysis.ideal_per_label, notAvailableText));
        appendStatRow(balanceBody, datasetSummaryLabels.minImages, formatStatValue(analysis.min_images, notAvailableText));
        appendStatRow(balanceBody, datasetSummaryLabels.maxImages, formatStatValue(analysis.max_images, notAvailableText));
        appendStatRow(balanceBody, datasetSummaryLabels.maxMinRatio, formatStatValue(analysis.ratio_max_to_min, notAvailableText));
        balanceCard.append(balanceBody);
        cont.append(balanceCard);
      }

      const labelsCard = Q('<div class="card">');
      Q('<h3 class="card-header">').text(t('ui.label_distribution', 'Label Distribution (Top 20)')).appendTo(labelsCard);
      const labelsBody = Q('<div class="card-body">');

      const sortedLabels = Object.entries(projectInfo.label_distribution)
        .sort(([, a], [, b]) => b - a)
        .slice(0, 20);

      const table = Q('<table class="label-table">');
      const thead = Q('<thead>');
      const headerRow = Q('<tr>');
      [tableHeaders.label, tableHeaders.count, tableHeaders.percentage].forEach(header => {
        Q('<th>').text(header).appendTo(headerRow);
      });
      thead.append(headerRow);
      table.append(thead);
      const tbody = Q('<tbody>');

      sortedLabels.forEach(([label, count]) => {
        const percentage = ((count / projectInfo.image_count) * 100).toFixed(1);
        const row = Q('<tr>');
        Q('<td>').text(label).appendTo(row);
        Q('<td>').text(count.toLocaleString()).appendTo(row);
        Q('<td>').text(`${percentage}%`).appendTo(row);
        tbody.append(row);
      });

      table.append(tbody);
      labelsBody.append(table);
      labelsCard.append(labelsBody);
      cont.append(labelsCard);

      if(projectInfo.recommendations && projectInfo.recommendations.length > 0){
        const recCard = Q('<div class="card">');
        Q('<h3 class="card-header">').text(t('ui.recommendations', 'Recommendations')).appendTo(recCard);
        const recBody = Q('<div class="card-body column gap">');
        projectInfo.recommendations.forEach(item => {
          Q('<div class="recommendation">').text(item).appendTo(recBody);
        });
        recCard.append(recBody);
        cont.append(recCard);
      }
    } catch (err){
      log('Dataset info load failed:', err);
      const errorCard = Q('<div class="card">');
      Q('<h2 class="card-header">').text(t('ui.error', 'Error')).appendTo(errorCard);
      Q('<div class="card-body">').text(t('ui.failed_to_load_dataset', 'Failed to load dataset information.')).appendTo(errorCard);
      Q('#dataset-cards').append(errorCard);
    }
  }

  function ensureUpdatesState(){
    if(!state.updates){
      state.updates = {
        pending: false,
        applying: false,
        files: [],
        orphaned: [],
        lastCheckedAt: null,
        lastError: null,
        message: null,
        summary: null
      };
    }
    return state.updates;
  }

  function selectUpdatesDom(){
    return {
      status: Q('#updates-status'),
      results: Q('#updates-results'),
      orphaned: Q('#updates-orphaned'),
      error: Q('#updates-error'),
      checkBtn: Q('#updates-check-button'),
      applyBtn: Q('#updates-apply-button')
    };
  }

  function truncateChecksum(value){
    if(typeof value !== 'string' || !value.trim()){
      return t('updates_ui.hash_missing', '—');
    }
    return value.trim().slice(0, 10);
  }

  function renderUpdatesState(){
    const updatesState = ensureUpdatesState();
    const domRefs = selectUpdatesDom();
    if(!domRefs.status || !domRefs.status.elements || !domRefs.status.elements.length){
      return;
    }

    const { status, results, orphaned, error, checkBtn, applyBtn } = domRefs;
    const isBusy = !!(updatesState.pending || updatesState.applying);

    if(checkBtn && checkBtn.prop){
      checkBtn.prop('disabled', isBusy);
    }

    if(applyBtn && applyBtn.prop){
      const hasCandidates = Array.isArray(updatesState.files) && updatesState.files.length > 0;
      const disabled = isBusy || !hasCandidates;
      applyBtn.prop('disabled', disabled);
      if(disabled){
        applyBtn.attr('title', t('updates_ui.apply_disabled_hint', 'Run a check to enable updates.'));
      } else {
        applyBtn.removeAttr('title');
      }
    }

    const message = updatesState.message || t('updates_ui.status_idle', 'No update checks have been run yet.');
    status.text(message);
    if(isBusy){
      status.addClass('pending');
    } else {
      status.removeClass('pending');
    }

    if(updatesState.lastError){
      error.text(updatesState.lastError).removeAttr('hidden');
      status.addClass('error-text');
    } else {
      error.text('').attr('hidden', true);
      status.removeClass('error-text');
    }

    if(results && results.html){
      results.html('');
      if(Array.isArray(updatesState.files) && updatesState.files.length){
        const table = Q('<table class="updates-table">');
        const thead = Q('<thead>');
        const headerRow = Q('<tr>');
        Q('<th>').text(t('updates_ui.table_header_file', 'File')).appendTo(headerRow);
        Q('<th>').text(t('updates_ui.table_header_status', 'Status')).appendTo(headerRow);
        Q('<th>').text(t('updates_ui.table_header_local', 'Local')).appendTo(headerRow);
        Q('<th>').text(t('updates_ui.table_header_remote', 'Remote')).appendTo(headerRow);
        thead.append(headerRow);
        table.append(thead);

        const tbody = Q('<tbody>');
        updatesState.files.forEach(entry => {
          const row = Q('<tr>');
          Q('<td>').text(entry.path || '?').appendTo(row);
          const statusLabel = entry.status_label || (entry.status === 'missing'
            ? t('updates_ui.table_row_missing', 'Missing locally')
            : t('updates_ui.table_row_outdated', 'Checksum mismatch'));
          Q('<td>').text(statusLabel).appendTo(row);
          Q('<td>').text(truncateChecksum(entry.local_checksum)).appendTo(row);
          Q('<td>').text(truncateChecksum(entry.remote_checksum)).appendTo(row);
          tbody.append(row);
        });
        table.append(tbody);

        results.append(table);
        Q('<div class="updates-footnote">')
          .text(t('updates_ui.table_footnote', 'Hashes are truncated for readability.'))
          .appendTo(results);
      } else {
        Q('<div class="updates-empty">')
          .text(t('updates_ui.no_updates', 'All tracked files are up to date.'))
          .appendTo(results);
      }
    }

    if(orphaned && orphaned.html){
      orphaned.html('');
      if(Array.isArray(updatesState.orphaned) && updatesState.orphaned.length){
        Q('<h4>').text(t('updates_ui.orphaned_title', 'Untracked local files')).appendTo(orphaned);
        const list = Q('<ul class="updates-orphaned-list">');
        updatesState.orphaned.forEach(path => {
          Q('<li>').text(path).appendTo(list);
        });
        orphaned.append(list);
      } else {
        orphaned.text(t('updates_ui.orphaned_none', 'No extra local files detected.'));
      }
    }
  }

  async function runUpdatesCheck(options = {}){
    const { silentFooter = false } = options || {};
    const updatesState = ensureUpdatesState();
    updatesState.pending = true;
    updatesState.message = t('updates_ui.status_checking', 'Checking for updates...');
    updatesState.lastError = null;
    renderUpdatesState();
    if(!silentFooter){
      setStatus(t('status.checking_updates', 'Checking for updates...'));
    }

    try {
      const response = await comm.checkSystemUpdates();
      updatesState.pending = false;
      updatesState.files = Array.isArray(response.files) ? response.files : [];
      updatesState.orphaned = Array.isArray(response.orphaned) ? response.orphaned : [];
      updatesState.lastCheckedAt = response.checked_at || null;
      updatesState.summary = response;
      updatesState.message = response.message || (updatesState.files.length
        ? t('updates_ui.status_ready', 'Update summary prepared.')
        : t('updates_ui.status_up_to_date', 'Everything is already up to date.'));
      updatesState.lastError = null;
      renderUpdatesState();

      if(!silentFooter){
        const statusKey = updatesState.files.length ? 'status.updates_ready' : 'status.updates_none';
        setStatus(t(statusKey, updatesState.files.length ? 'Updates available' : 'No updates found'));
      }

      return response;
    } catch (err){
      updatesState.pending = false;
      updatesState.lastError = err?.message || String(err);
      updatesState.message = t('updates_ui.status_failed', 'Update check failed.');
      renderUpdatesState();
      if(!silentFooter){
        setStatus(t('status.updates_check_failed', 'Update check failed'), true);
      }
      throw err;
    }
  }

  async function applySystemUpdates(){
    const updatesState = ensureUpdatesState();
    if(updatesState.applying){
      return null;
    }
    const candidates = Array.isArray(updatesState.files) ? updatesState.files.map(entry => entry.path).filter(Boolean) : [];
    if(!candidates.length){
      return null;
    }

    updatesState.applying = true;
    updatesState.lastError = null;
    updatesState.message = t('updates_ui.status_applying', 'Updating files...');
    renderUpdatesState();
    setStatus(t('status.updates_applying', 'Applying updates...'));

    try {
      const response = await comm.applySystemUpdates(candidates);
      updatesState.applying = false;

      const hasFailures = Array.isArray(response.failed) && response.failed.length;
      if(response.status === 'error' || hasFailures){
        const failureMessage = response.message || t('updates_ui.status_apply_failed', 'Some updates failed.');
        updatesState.lastError = hasFailures ? (response.failed[0]?.error || failureMessage) : failureMessage;
        updatesState.message = failureMessage;
        renderUpdatesState();
        setStatus(t('status.updates_apply_failed', 'Update apply failed'), true);
      } else {
        updatesState.message = response.message || t('updates_ui.status_applied', 'Updates applied successfully.');
        updatesState.lastError = null;
        renderUpdatesState();
        setStatus(t('status.updates_applied', 'Updates applied successfully'));
      }

      await runUpdatesCheck({ silentFooter: true }).catch(() => null);
      return response;
    } catch (err){
      updatesState.applying = false;
      updatesState.lastError = err?.message || String(err);
      updatesState.message = t('updates_ui.status_apply_failed', 'Some updates failed.');
      renderUpdatesState();
      setStatus(t('status.updates_apply_failed', 'Update apply failed'), true);
      throw err;
    }
  }

  function ensureDocsState(){
    if(!state.docs){
      state.docs = {
        initialized: false,
        files: [],
        cache: {},
        currentPath: null,
        loading: false,
        error: null,
        pendingAnchor: null
      };
    }
    return state.docs;
  }

  function selectDocsDom(){
    return {
      list: Q('#docs-list'),
      empty: Q('#docs-empty'),
      status: Q('#docs-status'),
      content: Q('#docs-content'),
      title: Q('#docs-current-title'),
      external: Q('#docs-open-external')
    };
  }

  function setDocsStatus(message, { isError = false } = {}){
    const domRefs = selectDocsDom();
    if(!domRefs.status || !domRefs.status.text) return;
    if(!message){
      domRefs.status.text('');
      domRefs.status.attr('hidden', true);
      domRefs.status.removeClass('error-text');
    } else {
      domRefs.status.text(message);
      domRefs.status.removeAttr('hidden');
      if(isError){
        domRefs.status.addClass('error-text');
      } else {
        domRefs.status.removeClass('error-text');
      }
    }
  }

  function slugifyHeading(textValue){
    if(typeof textValue !== 'string') return '';
    const normalized = textValue.trim().toLowerCase();
    if(!normalized) return '';
    const slug = normalized
      .replace(/[^a-z0-9\s-_]+/g, '')
      .replace(/\s+/g, '-');
    return slug;
  }

  function normalizeDocPath(rawPath){
    if(typeof rawPath !== 'string') return null;
    let working = rawPath.trim();
    if(!working) return null;
    working = working.replace(/\\/g, '/');
    working = working.replace(/^\/+/, '');
    if(working.toLowerCase().startsWith('docs/')){
      working = working.slice(5);
    }
    working = working.replace(/^\.\//, '');
    while(working.includes('//')){
      working = working.replace(/\/+/g, '/');
    }
    if(!working) return null;
    return working;
  }

  function formatDocTitle(path){
    if(typeof path !== 'string' || !path.trim()){
      return '';
    }
    const leaf = path.split('/').pop() || path;
    const withoutExt = leaf.replace(/\.md$/i, '');
    const humanized = withoutExt.replace(/[-_]+/g, ' ').trim();
    if(!humanized){
      return leaf;
    }
    return humanized.charAt(0).toUpperCase() + humanized.slice(1);
  }

  function resolveDocTarget(rawHref){
    const docsState = ensureDocsState();
    if(typeof rawHref !== 'string' || !rawHref.trim()){
      return { path: docsState.currentPath, anchor: null, external: null };
    }

    let href = rawHref.trim();
    let anchor = null;
    const hashIndex = href.indexOf('#');
    if(hashIndex >= 0){
      anchor = href.slice(hashIndex + 1);
      href = href.slice(0, hashIndex);
    }

    if(!href){
      return { path: docsState.currentPath, anchor, external: null };
    }

    if(/^https?:\/\//i.test(href) || href.startsWith('mailto:') || href.startsWith('tel:') || href.startsWith('data:')){
      return { path: null, anchor, external: href };
    }

    let isAbsolute = href.startsWith('/');
    let normalized = href.replace(/\\/g, '/');
    if(normalized.startsWith('/')){
      normalized = normalized.replace(/^\/+/, '');
    }
    if(normalized.toLowerCase().startsWith('docs/')){
      normalized = normalized.slice(5);
      isAbsolute = true;
    }

    const segments = normalized.split('/');
    const baseSegments = isAbsolute ? [] : (docsState.currentPath ? docsState.currentPath.split('/').slice(0, -1) : []);
    const resolvedSegments = [];
    baseSegments.forEach(segment => {
      if(segment) resolvedSegments.push(segment);
    });

    segments.forEach(segment => {
      if(!segment || segment === '.') return;
      if(segment === '..'){
        if(resolvedSegments.length){
          resolvedSegments.pop();
        }
        return;
      }
      resolvedSegments.push(segment);
    });

    let resolvedPath = resolvedSegments.join('/');
    if(resolvedPath && !/\.md$/i.test(resolvedPath)){
      resolvedPath = `${resolvedPath}.md`;
    }
    return { path: normalizeDocPath(resolvedPath), anchor, external: null };
  }

  function resolveDocAssetPath(currentPath, assetPath){
    if(typeof assetPath !== 'string' || !assetPath.trim()){
      return null;
    }

    const trimmed = assetPath.trim();
    if(/^https?:\/\//i.test(trimmed) || trimmed.startsWith('data:') || trimmed.startsWith('mailto:') || trimmed.startsWith('tel:')){
      return trimmed;
    }

    let normalized = trimmed.replace(/\\/g, '/');
    let isAbsolute = normalized.startsWith('/');
    if(isAbsolute){
      normalized = normalized.replace(/^\/+/, '');
    }
    if(normalized.toLowerCase().startsWith('docs/')){
      normalized = normalized.slice(5);
      isAbsolute = true;
    }

    const baseSegments = isAbsolute || !currentPath
      ? []
      : currentPath.split('/').slice(0, -1);
    const resolvedSegments = [];
    baseSegments.forEach(seg => {
      if(seg) resolvedSegments.push(seg);
    });
    normalized.split('/').forEach(seg => {
      if(!seg || seg === '.') return;
      if(seg === '..'){
        if(resolvedSegments.length){
          resolvedSegments.pop();
        }
        return;
      }
      resolvedSegments.push(seg);
    });

    if(!resolvedSegments.length){
      return null;
    }

    return `/docs/assets/${resolvedSegments.join('/')}`;
  }

  function decorateDocsContent(){
    const docsState = ensureDocsState();
    const domRefs = selectDocsDom();
    if(!domRefs.content || !domRefs.content.find) return;

    const slugCounts = new Map();
    domRefs.content.find('h1, h2, h3, h4, h5, h6').getAll().forEach((heading) => {
      const slugBase = slugifyHeading(heading.textContent || '');
      if(!slugBase) return;
      let slug = slugBase;
      const current = slugCounts.get(slugBase) || 0;
      if(current > 0){
        slug = `${slugBase}-${current}`;
      }
      slugCounts.set(slugBase, current + 1);
      heading.setAttribute('id', slug);
    });

    domRefs.content.find('a[data-doc-link="true"]').getAll().forEach(anchor => {
      anchor.removeAttribute('target');
      anchor.removeAttribute('rel');
    });

    domRefs.content.find('img').getAll().forEach(image => {
      const src = image.getAttribute('src');
      const resolved = resolveDocAssetPath(docsState.currentPath, src);
      if(resolved){
        image.setAttribute('src', resolved);
      }
      if(!image.getAttribute('alt')){
        image.setAttribute('alt', '');
      }
    });

    if(docsState.pendingAnchor){
      scrollToDocAnchor(docsState.pendingAnchor);
      docsState.pendingAnchor = null;
    }
  }

  function renderDocsList(){
    const docsState = ensureDocsState();
    const domRefs = selectDocsDom();
    if(!domRefs.list) return;

    domRefs.list.html('');
    if(!Array.isArray(docsState.files) || !docsState.files.length){
      if(domRefs.empty){
        domRefs.empty.removeAttr('hidden');
      }
      return;
    }

    if(domRefs.empty){
      domRefs.empty.attr('hidden', true);
    }

    const current = (docsState.currentPath || '').toLowerCase();
    docsState.files.forEach(entry => {
      const li = Q('<li>');
      const button = Q('<button type="button">')
        .attr('data-doc-path', entry.path)
        .text(entry.title || entry.path);
      if(entry.path && entry.path.toLowerCase() === current){
        button.addClass('active');
      }
      li.append(button);
      domRefs.list.append(li);
    });
  }

  function highlightActiveDoc(path){
    const normalized = (path || '').toLowerCase();
    Q('#docs-list button[data-doc-path]').each((_, button) => {
      const btnPath = button.getAttribute('data-doc-path');
      if(btnPath && btnPath.toLowerCase() === normalized){
        button.classList.add('active');
      } else {
        button.classList.remove('active');
      }
    });
  }

  async function refreshDocsList(){
    const docsState = ensureDocsState();
    const domRefs = selectDocsDom();
    if(domRefs.list){
      domRefs.list.addClass('loading');
    }
    setDocsStatus(t('docs_ui.loading_list', 'Loading documentation list...'));
    try {
      const response = await comm.fetchDocsList();
      const entries = Array.isArray(response.docs) ? response.docs : [];
      docsState.files = entries
        .map(item => {
          const normalizedPath = normalizeDocPath(item.path || '');
          if(!normalizedPath) return null;
          const title = item.title || formatDocTitle(normalizedPath) || normalizedPath;
          return { path: normalizedPath, title };
        })
        .filter(Boolean);
      docsState.files.sort((a, b) => a.path.localeCompare(b.path, undefined, { sensitivity: 'base' }));
      if(!docsState.files.length){
        setDocsStatus(t('docs_ui.empty_content', 'No documentation files available in the docs folder.'));
      } else if(!docsState.currentPath){
        setDocsStatus(t('docs_ui.loading_placeholder', 'Select a document to view.'));
      } else {
        setDocsStatus('');
      }
      renderDocsList();
    } catch (err){
      docsState.files = [];
      setDocsStatus(t('docs_ui.list_failed', 'Failed to load documentation list.'), { isError: true });
      if(domRefs.empty){
        domRefs.empty.attr('hidden', true);
      }
      log('Docs list load failed:', err);
    } finally {
      if(domRefs.list){
        domRefs.list.removeClass('loading');
      }
    }
  }

  function renderDocContent(rawContent){
    const domRefs = selectDocsDom();
    if(!domRefs.content) return;

    const rendered = markdown && typeof markdown.render === 'function'
      ? markdown.render(rawContent || '')
      : (rawContent || '');
    domRefs.content.html(rendered);
    decorateDocsContent();
  }

  function scrollToDocAnchor(anchorId){
    if(!anchorId) return;
    const domRefs = selectDocsDom();
    if(!domRefs.content || !domRefs.content.get) return;
    const container = domRefs.content.get();
    if(!container) return;
    const escaped = (typeof CSS !== 'undefined' && CSS.escape)
      ? CSS.escape(anchorId)
      : anchorId.replace(/[^a-zA-Z0-9_-]/g, match => `\\${match}`);
    const target = container.querySelector(`#${escaped}`);
    if(target && typeof target.scrollIntoView === 'function'){
      target.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }

  async function openDoc(path, options = {}){
    const docsState = ensureDocsState();
    const domRefs = selectDocsDom();
    const normalizedPath = normalizeDocPath(path) || docsState.currentPath;
    if(!normalizedPath){
      return;
    }

    if(docsState.loading){
      return;
    }

    docsState.loading = true;
    docsState.error = null;
    docsState.pendingAnchor = options.anchor || null;
    setDocsStatus(t('docs_ui.loading_file', 'Loading document...'));
    if(domRefs.content){
      domRefs.content.html('');
    }

    try {
      let content = docsState.cache[normalizedPath];
      let canonicalPath = normalizedPath;
      if(typeof content !== 'string'){
        const response = await comm.fetchDocPage(normalizedPath);
        if(response.status === 'error'){
          throw new Error(response.message || 'Document load failed');
        }
        content = response.content || '';
        docsState.cache[normalizedPath] = content;
        if(response.path){
          const resolved = normalizeDocPath(response.path);
          if(resolved){
            canonicalPath = resolved;
            docsState.cache[canonicalPath] = content;
          }
        }
      }

      docsState.currentPath = responsePathOrDefault(canonicalPath, docsState.files);
      setDocsStatus('');
      renderDocContent(content);
      highlightActiveDoc(docsState.currentPath);

      if(domRefs.title){
        const currentEntry = docsState.files.find(item => item.path === docsState.currentPath);
        const fallbackTitle = formatDocTitle(docsState.currentPath);
        domRefs.title.text(currentEntry?.title || fallbackTitle || docsState.currentPath);
      }

      if(domRefs.external){
        if(docsState.currentPath){
          const encodedPath = docsState.currentPath.split('/').map(segment => encodeURIComponent(segment)).join('/');
          domRefs.external.attr('href', `/docs/assets/${encodedPath}`);
          domRefs.external.removeAttr('hidden');
        } else {
          domRefs.external.attr('hidden', true);
          domRefs.external.removeAttr('href');
        }
      }
    } catch (err){
      docsState.error = err?.message || String(err);
      setDocsStatus(docsState.error, { isError: true });
      if(domRefs.content){
        domRefs.content.html('');
      }
      if(domRefs.title){
        domRefs.title.text(formatDocTitle(normalizedPath) || normalizedPath || t('docs_ui.placeholder_title', 'Documentation'));
      }
      if(domRefs.external){
        domRefs.external.attr('hidden', true);
        domRefs.external.removeAttr('href');
      }
      log('Doc load failed:', err);
    } finally {
      docsState.loading = false;
    }
  }

  function responsePathOrDefault(candidate, files){
    if(!candidate) return candidate;
    const lower = candidate.toLowerCase();
    const match = Array.isArray(files) ? files.find(entry => entry.path && entry.path.toLowerCase() === lower) : null;
    return match?.path || candidate;
  }

  async function initDocsPage(){
    const docsState = ensureDocsState();
    if(docsState.initialized){
      renderDocsList();
      highlightActiveDoc(docsState.currentPath);
      return;
    }

    docsState.initialized = true;
    await refreshDocsList();

    const defaultPath = (() => {
      if(Array.isArray(docsState.files) && docsState.files.length){
        const preferred = docsState.files.find(item => item.path && item.path.toLowerCase() === 'start.md');
        return preferred?.path || docsState.files[0].path;
      }
      return null;
    })();

    if(defaultPath){
      await openDoc(defaultPath);
    }
  }

  function handleDocsLinkClick(event){
    const anchor = event.target && event.target.closest ? event.target.closest('a') : null;
    if(!anchor) return;
    const href = anchor.getAttribute('href');
    if(!href) return;

    const target = resolveDocTarget(href);
    if(target.external){
      anchor.setAttribute('target', '_blank');
      anchor.setAttribute('rel', 'noopener noreferrer');
      return;
    }

    event.preventDefault();
    if(target.path && target.path !== ensureDocsState().currentPath){
      openDoc(target.path, { anchor: target.anchor });
    } else if(target.anchor){
      scrollToDocAnchor(target.anchor);
    }
  }

  function hexToRGBA(hex, alpha){
    if(typeof hex !== 'string'){
      return `rgba(255, 255, 255, ${alpha ?? 1})`;
    }
    let normalized = hex.trim().replace(/^#/, '');
    if(normalized.length === 3){
      normalized = normalized.split('').map(ch => ch + ch).join('');
    }
    const bigint = parseInt(normalized, 16);
    if(Number.isNaN(bigint)){
      return `rgba(255, 255, 255, ${alpha ?? 1})`;
    }
    const r = (bigint >> 16) & 255;
    const g = (bigint >> 8) & 255;
    const b = bigint & 255;
    const safeAlpha = alpha === undefined ? 1 : Math.max(0, Math.min(1, alpha));
    return `rgba(${r}, ${g}, ${b}, ${safeAlpha})`;
  }

  function createEmptyTelemetry(trainingId = null){
    return {
      activeTrainingId: trainingId || null,
      counter: 0,
      maxPoints: Infinity,
      pendingHistory: !!trainingId,
      series: {
        step_loss: [],
        train_epoch_loss: [],
        val_epoch_loss: [],
        train_epoch_accuracy: [],
        val_epoch_accuracy: []
      },
      timer: null
    };
  }

  function ensureTrainingTelemetry(){
    if(!state.trainingTelemetry){
      state.trainingTelemetry = createEmptyTelemetry();
    }
    return state.trainingTelemetry;
  }

  function resetTrainingTelemetry(trainingId){
    const existing = state.trainingTelemetry;
    if(existing && existing.timer){
      clearTimeout(existing.timer);
    }
    state.trainingTelemetry = createEmptyTelemetry(trainingId);
  }

  function pushMetricPoint(series, x, y, telemetry){
    if(!Array.isArray(series)){
      return;
    }
    const value = Number(y);
    if(!Number.isFinite(value)){
      return;
    }
    series.push({ x, y: value });
    const maxPoints = telemetry?.maxPoints || 1200;
    if(series.length > maxPoints){
      series.splice(0, series.length - maxPoints);
    }
  }

  function ingestTrainingEvents(events){
    if(!Array.isArray(events) || !events.length){
      return;
    }
    const telemetry = ensureTrainingTelemetry();
    events.forEach(event => {
      if(!event || typeof event !== 'object'){
        return;
      }
      const metrics = event.metrics || {};
      const phase = typeof event.phase === 'string' ? event.phase.toLowerCase() : '';
      const x = ++telemetry.counter;
      if(Number.isFinite(Number(metrics.step_loss))){
        pushMetricPoint(telemetry.series.step_loss, x, metrics.step_loss, telemetry);
      } else if(event.type === 'step' && Number.isFinite(Number(metrics.loss))){
        pushMetricPoint(telemetry.series.step_loss, x, metrics.loss, telemetry);
      }
      const epochLoss = Number(metrics.epoch_loss);
      const trainLoss = Number(metrics.train_loss);
      const valLoss = Number(metrics.val_loss);
      if(Number.isFinite(trainLoss)){
        pushMetricPoint(telemetry.series.train_epoch_loss, x, trainLoss, telemetry);
      } else if(Number.isFinite(epochLoss) && phase === 'train'){
        pushMetricPoint(telemetry.series.train_epoch_loss, x, epochLoss, telemetry);
      }
      if(Number.isFinite(valLoss)){
        pushMetricPoint(telemetry.series.val_epoch_loss, x, valLoss, telemetry);
      } else if(Number.isFinite(epochLoss) && phase === 'val'){
        pushMetricPoint(telemetry.series.val_epoch_loss, x, epochLoss, telemetry);
      }

      const trainAcc = Number(metrics.train_accuracy);
      const valAcc = Number(metrics.val_accuracy);
      const epochAcc = Number(metrics.epoch_accuracy);
      if(Number.isFinite(trainAcc)){
        pushMetricPoint(telemetry.series.train_epoch_accuracy, x, trainAcc, telemetry);
      } else if(Number.isFinite(epochAcc) && phase === 'train'){
        pushMetricPoint(telemetry.series.train_epoch_accuracy, x, epochAcc, telemetry);
      }
      if(Number.isFinite(valAcc)){
        pushMetricPoint(telemetry.series.val_epoch_accuracy, x, valAcc, telemetry);
      } else if(Number.isFinite(epochAcc) && phase === 'val'){
        pushMetricPoint(telemetry.series.val_epoch_accuracy, x, epochAcc, telemetry);
      }
    });
  }

  function formatProgress(current, total){
    if(current === undefined || current === null){
      return '—';
    }
    if(total === undefined || total === null || !Number.isFinite(Number(total)) || Number(total) <= 0){
      return `${current}`;
    }
    return `${current} / ${total}`;
  }

  function buildSummaryItem(labelKey, fallback, value){
    const row = Q('<div class="status-summary-item">');
    Q('<span class="status-summary-label">').text(t(labelKey, fallback)).appendTo(row);
    Q('<span class="status-summary-value">').text(value ?? '—').appendTo(row);
    return row;
  }

  function buildGraphElement(labelKey, fallback, points, color){
    const wrapper = Q('<div class="status-graph">');
    const header = Q('<div class="status-graph-header">');
    Q('<div class="status-graph-title">').text(t(labelKey, fallback)).appendTo(header);
    const latestValue = points.length ? points[points.length - 1].y : null;
    Q('<div class="status-graph-value">').text(formatMetricValue(latestValue)).appendTo(header);
    wrapper.append(header);
    const canvas = document.createElement('canvas');
    canvas.className = 'status-graph-canvas';
    canvas.height = 200;
    wrapper.append(canvas);
    if(!points.length){
      wrapper.append(Q('<div class="status-graph-empty">').text(t('status_graph.no_data', 'Waiting for updates')));
    }
    window.requestAnimationFrame(() => drawLineGraph(canvas, points, { color }));
    return wrapper;
  }

  function drawLineGraph(canvas, points, options = {}){
    if(!canvas){
      return;
    }
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || (canvas.parentElement ? canvas.parentElement.clientWidth : 320);
    const height = canvas.clientHeight || 160;
    const deviceWidth = Math.max(1, Math.round(width * ratio));
    const deviceHeight = Math.max(1, Math.round(height * ratio));
    if(canvas.width !== deviceWidth || canvas.height !== deviceHeight){
      canvas.width = deviceWidth;
      canvas.height = deviceHeight;
    }
    const ctx = canvas.getContext('2d');
    if(!ctx){
      return;
    }
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, deviceWidth, deviceHeight);
    ctx.scale(ratio, ratio);

    const drawWidth = width;
    const drawHeight = height;
    const padding = Math.min(28, Math.max(16, drawWidth * 0.08));
    const graphWidth = Math.max(1, drawWidth - padding * 2);
    const graphHeight = Math.max(1, drawHeight - padding * 2);

     ctx.fillStyle = 'rgba(15, 20, 31, 0.85)';
     ctx.fillRect(padding, padding, graphWidth, graphHeight);

    const gridLines = 4;
    ctx.strokeStyle = 'rgba(255, 255, 255, 0.05)';
    ctx.lineWidth = 1;
    ctx.setLineDash([3, 5]);
    const stepY = graphHeight / gridLines;
    for(let i = 1; i < gridLines; i += 1){
      const y = drawHeight - padding - stepY * i;
      ctx.beginPath();
      ctx.moveTo(padding, y);
      ctx.lineTo(drawWidth - padding, y);
      ctx.stroke();
    }
    const stepX = graphWidth / gridLines;
    for(let i = 1; i < gridLines; i += 1){
      const x = padding + stepX * i;
      ctx.beginPath();
      ctx.moveTo(x, padding);
      ctx.lineTo(x, drawHeight - padding);
      ctx.stroke();
    }
    ctx.setLineDash([]);

    ctx.strokeStyle = 'rgba(255, 255, 255, 0.18)';
    ctx.beginPath();
    ctx.moveTo(padding, drawHeight - padding);
    ctx.lineTo(drawWidth - padding, drawHeight - padding);
    ctx.moveTo(padding, drawHeight - padding);
    ctx.lineTo(padding, padding);
    ctx.stroke();

    if(points && points.length){
      let minX = points[0].x;
      let maxX = points[0].x;
      let minY = points[0].y;
      let maxY = points[0].y;
      for(let i = 1; i < points.length; i += 1){
        const pt = points[i];
        if(pt.x < minX) minX = pt.x;
        if(pt.x > maxX) maxX = pt.x;
        if(pt.y < minY) minY = pt.y;
        if(pt.y > maxY) maxY = pt.y;
      }
      if(minX === maxX){
        minX -= 1;
        maxX += 1;
      }
      if(minY === maxY){
        const delta = Math.abs(minY) < 1 ? 1 : Math.abs(minY) * 0.05;
        minY -= delta;
        maxY += delta;
      }
      const spanX = maxX - minX || 1;
      const spanY = maxY - minY || 1;
      const color = options.color || '#7aa9ff';
      const coords = points.map(pt => {
        const normX = (pt.x - minX) / spanX;
        const normY = (pt.y - minY) / spanY;
        return {
          x: padding + normX * graphWidth,
          y: drawHeight - (padding + normY * graphHeight)
        };
      });

      const gradient = ctx.createLinearGradient(0, padding, 0, drawHeight - padding);
      gradient.addColorStop(0, hexToRGBA(color, 0.3));
      gradient.addColorStop(1, hexToRGBA(color, 0.04));

      const baselineY = drawHeight - padding;

      ctx.beginPath();
      coords.forEach((c, idx) => {
        if(idx === 0){
          ctx.moveTo(c.x, c.y);
        } else {
          ctx.lineTo(c.x, c.y);
        }
      });
      ctx.lineTo(coords[coords.length - 1].x, baselineY);
      ctx.lineTo(coords[0].x, baselineY);
      ctx.closePath();
      ctx.fillStyle = gradient;
      ctx.fill();

      ctx.beginPath();
      coords.forEach((c, idx) => {
        if(idx === 0){
          ctx.moveTo(c.x, c.y);
        } else {
          ctx.lineTo(c.x, c.y);
        }
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = 2;
      ctx.lineJoin = 'round';
      ctx.lineCap = 'round';
      ctx.shadowColor = hexToRGBA(color, 0.25);
      ctx.shadowBlur = 8;
      ctx.stroke();
      ctx.shadowBlur = 0;

      const axisFont = '600 10px "Roboto", "Segoe UI", sans-serif';
      const axisColor = 'rgba(255, 255, 255, 0.6)';
      ctx.font = axisFont;
      ctx.fillStyle = axisColor;

      ctx.textAlign = 'right';
      for(let i = 0; i <= gridLines; i += 1){
        const value = minY + (spanY * i) / gridLines;
        const y = drawHeight - padding - (graphHeight / gridLines) * i;
        ctx.textBaseline = i === 0 ? 'bottom' : (i === gridLines ? 'top' : 'middle');
        ctx.fillText(formatAxisTickValue(value), padding - 10, y);
      }

      for(let i = 0; i <= gridLines; i += 1){
        const value = minX + (spanX * i) / gridLines;
        const x = padding + (graphWidth / gridLines) * i;
        ctx.textAlign = i === 0 ? 'left' : (i === gridLines ? 'right' : 'center');
        ctx.textBaseline = 'top';
        ctx.fillText(formatAxisTickValue(value, spanX >= gridLines), x, drawHeight - padding + 8);
      }
    }

    ctx.restore();
  }

  function formatMetricValue(value){
    if(value === undefined || value === null || Number.isNaN(Number(value))){
      return '—';
    }
    const numeric = Number(value);
    const abs = Math.abs(numeric);
    if(abs >= 1000){
      return numeric.toFixed(0);
    }
    if(abs >= 100){
      return numeric.toFixed(1);
    }
    if(abs >= 10){
      return numeric.toFixed(2);
    }
    if(abs >= 1){
      return numeric.toFixed(3);
    }
    if(abs >= 0.01){
      return numeric.toFixed(4);
    }
    return numeric.toExponential(2);
  }

  function formatAxisTickValue(value, preferInteger = false){
    if(!Number.isFinite(value)){
      return '';
    }
    if(preferInteger){
           return Math.round(value).toLocaleString();
    }
    const abs = Math.abs(value);
    if(abs >= 1000){
      return Math.round(value).toLocaleString();
    }
    if(abs >= 100){
      return value.toFixed(1);
    }
    if(abs >= 1){
      return value.toFixed(2);
    }
    if(abs >= 0.01){
      const fixed = value.toFixed(4);
      return fixed.replace(/\.0+$/, '').replace(/(\.\d*?)0+$/, '$1');
    }
    return value.toExponential(1).replace('+', '');
  }

  function determineStatusPollDelay(data, hadError){
    if(hadError){
      return 7000;
    }
    if(data && typeof data === 'object' && data.status){
      const statusText = String(data.status).toLowerCase();
      if(statusText === 'running' || statusText === 'starting'){
        return 5000;
      }
    }
    return 12000;
  }

  async function handleTrainingStatus(data){
    const telemetry = ensureTrainingTelemetry();
    const previousStatusData = state.trainingStatus;
    const previousTrainingId = previousStatusData?.training_id || null;
    const previousStatus = previousStatusData?.status || null;

    state.trainingStatus = data;

    if(!data || data.error){
      if(data && data.error && telemetry.activeTrainingId){
        resetTrainingTelemetry(null);
      }
      clearStoredTrainingContext();
      state.pendingAutoProject = null;
      state.pendingAutoTrainingId = null;
      state.lastAutoLoadedTrainingId = null;
      renderTrainingStatus();
      updateTrainingBadge();
      applyFooterStatus();
      if(previousTrainingId){
        refreshProjects();
      }
      return;
    }

    const currentTrainingId = data.training_id || null;
    let trainingIdChanged = currentTrainingId !== previousTrainingId;

    if(currentTrainingId){
      if(telemetry.activeTrainingId !== currentTrainingId){
        resetTrainingTelemetry(currentTrainingId);
      }
      persistTrainingContext(currentTrainingId, data.project || null);

      if(data.project && (!state.currentProject || state.currentProject.name !== data.project)){
        state.pendingAutoProject = data.project;
        state.pendingAutoTrainingId = currentTrainingId;
      }

      const currentTelemetry = ensureTrainingTelemetry();
      if(Array.isArray(data.updates) && data.updates.length){
        ingestTrainingEvents(data.updates);
        currentTelemetry.pendingHistory = false;
      }
      if(currentTelemetry.pendingHistory && currentTelemetry.counter === 0){
        try {
          const history = await comm.fetchTrainingHistory(currentTrainingId);
          if(history && Array.isArray(history.events)){
            ingestTrainingEvents(history.events);
          }
        } catch (err){
          log('Training history fetch failed:', err);
        } finally {
          currentTelemetry.pendingHistory = false;
        }
      }
    } else {
      if(telemetry.activeTrainingId){
        resetTrainingTelemetry(null);
      }
      clearStoredTrainingContext();
      state.pendingAutoProject = null;
      state.pendingAutoTrainingId = null;
      state.lastAutoLoadedTrainingId = null;
    }

    renderTrainingStatus();
    updateTrainingBadge();
    applyFooterStatus();

    const currentStatus = data.status || null;
    if(trainingIdChanged || (currentTrainingId && currentStatus !== previousStatus)){ 
      refreshProjects();
    }
  }

  function renderTrainingStatus(){
    const el = Q('#status-body');
    if(!el.elements.length){
      return;
    }

    el.html('');

    const telemetry = ensureTrainingTelemetry();
    const statusData = state.trainingStatus;

    if(!statusData){
      el.append(Q('<div class="status-empty">').text(t('status_graph.no_training', 'No active training.')));
      return;
    }

    if(statusData.error){
      el.append(Q('<div class="status-empty">').text(statusData.error));
      return;
    }

    if(!statusData.training_id){
      const count = typeof statusData.count === 'number' ? statusData.count : 0;
      if(count > 0){
        const message = t('status_graph.active_count', 'Active trainings: {count}').replace('{count}', count);
        el.append(Q('<div class="status-note">').text(message));
        const ids = Array.isArray(statusData.active_trainings) ? statusData.active_trainings.join(', ') : '';
        if(ids){
          el.append(Q('<div class="status-note secondary">').text(ids));
        }
      } else {
        el.append(Q('<div class="status-empty">').text(t('status_graph.no_training', 'No active training.')));
      }
      return;
    }

    const summary = Q('<div class="status-summary">');
    summary.append(buildSummaryItem('status_graph.label_training_id', 'Training ID', statusData.training_id));
    summary.append(buildSummaryItem('status_graph.label_status', 'Status', statusData.status || '—'));
    summary.append(buildSummaryItem('status_graph.label_phase', 'Phase', statusData.phase || '—'));
    summary.append(buildSummaryItem('status_graph.label_epoch', 'Epoch', formatProgress(statusData.current_epoch, statusData.total_epochs)));
    summary.append(buildSummaryItem('status_graph.label_step', 'Step', formatProgress(statusData.current_step, statusData.total_steps)));
    el.append(summary);

    const graphsWrap = Q('<div class="status-graphs-grid">');
    const graphs = [
      { key: 'train_epoch_accuracy', label: 'status_graph.train_accuracy', fallback: 'Train Accuracy', color: '#64b5f6' },
      { key: 'val_epoch_accuracy', label: 'status_graph.val_accuracy', fallback: 'Validation Accuracy', color: '#9575cd' },
      { key: 'train_epoch_loss', label: 'status_graph.train_loss', fallback: 'Train Loss', color: '#81c784' },
      { key: 'val_epoch_loss', label: 'status_graph.val_loss', fallback: 'Validation Loss', color: '#ffb74d' }
    ];

    graphs.forEach(item => {
      const series = telemetry.series[item.key] || [];
      graphsWrap.append(buildGraphElement(item.label, item.fallback, series, item.color));
    });

    el.append(graphsWrap);
  }

  async function fetchStatusPayload(){
    let telemetry = ensureTrainingTelemetry();
    let targetId = telemetry.activeTrainingId || null;
    let response = await comm.fetchTrainingStatus(targetId);

    if(response && response.error && targetId){
      resetTrainingTelemetry(null);
      telemetry = ensureTrainingTelemetry();
      targetId = null;
      response = await comm.fetchTrainingStatus();
    }

    if(response && !response.training_id){
      const activeList = Array.isArray(response.active_trainings) ? response.active_trainings : [];
      if(activeList.length){
        const nextId = activeList[0];
        if(telemetry.activeTrainingId !== nextId){
          resetTrainingTelemetry(nextId);
          telemetry = ensureTrainingTelemetry();
        }
        response = await comm.fetchTrainingStatus(nextId);
      }
    }

    return response;
  }

  async function pollStatus(){
    const telemetry = ensureTrainingTelemetry();
    if(telemetry.timer){
      clearTimeout(telemetry.timer);
      telemetry.timer = null;
    }

    let hadError = false;
    try {
      const payload = await fetchStatusPayload();
      await handleTrainingStatus(payload);
    } catch (err){
      hadError = true;
      log('Training status poll failed:', err);
    }

    const delay = determineStatusPollDelay(state.trainingStatus, hadError);
    const updatedTelemetry = ensureTrainingTelemetry();
    updatedTelemetry.timer = window.setTimeout(pollStatus, delay);
  }

  async function generateHeatmap(){
    if(!state.currentProject){
      const message = `${t('ui.no_project_loaded', 'No Project Loaded')}. ${t('ui.load_project_first', 'Please load a project first from the Projects tab.')}`;
      alert(message);
      return;
    }

    try {
      setStatus(t('status.generating_heatmap', 'Generating heatmap...'));
      const data = await comm.evaluateProject(state.currentProject.name);
      setStatus(t('status.heatmap_generated', 'Heatmap generated'));

      const imageCont = Q('#heatmap-image');
      imageCont.html('');
      const img = Q('<img>').attr('src', `data:image/png;base64,${data.heatmap}`);
      imageCont.append(img);
      img.elements[0].addEventListener('load', () => window.adjustImageSize());

      const dataCont = Q('#heatmap-data');
      dataCont.html('');
      const table = Q('<table class="mini-table">');
      const tbody = Q('<tbody>');
      if(data.predictions?.predicted_classes?.length){
        data.predictions.predicted_classes.forEach((cls, index) => {
          const conf = data.predictions.confidence_values[index];
          const row = Q('<tr>');
          Q('<td>').text(t('ui.prediction', 'Prediction')).appendTo(row);
          Q('<td>').text(`${cls}: ${(conf * 100).toFixed(1)}%`).css('word-break', 'break-all').appendTo(row);
          tbody.append(row);
        });
      } else {
        const row = Q('<tr>');
        Q('<td>').text(t('ui.predictions', 'Predictions')).appendTo(row);
        Q('<td>').text(t('ui.no_predictions_above_threshold', 'No predictions above threshold')).css('word-break', 'break-all').appendTo(row);
        tbody.append(row);
      }
      const row1 = Q('<tr>');
      Q('<td>').text(t('ui.image', 'Image')).appendTo(row1);
      Q('<td>').text(data.image_path).css('word-break', 'break-all').appendTo(row1);
      tbody.append(row1);
      const row2 = Q('<tr>');
      Q('<td>').text(t('ui.checkpoint', 'Checkpoint')).appendTo(row2);
      Q('<td>').text(data.checkpoint).css('word-break', 'break-all').appendTo(row2);
      tbody.append(row2);
      table.append(tbody);
      dataCont.append(table);
    } catch (err){
      setStatus(t('status.heatmap_generation_failed', 'Heatmap generation failed'), true);
      log(err);
      alert('Failed to generate heatmap: ' + err.message);
    }
  }

  async function saveSystemConfig(){
    collectCurrentValues();
    const errs = runValidation();
    if(errs.length){
      setStatus(t('status.validation_errors', 'Fix validation errors before saving'), true);
      return;
    }

    const btn = dom.btnSaveSystem;
    try {
      setStatus(t('status.saving_system_settings', 'Saving system settings...'));
      if(btn && btn.elements && btn.elements.length){
        btn.prop('disabled', true);
      }
      await comm.saveSystemConfig(state.config);
      state.dirty = false;
      setStatus(t('status.system_settings_saved', 'System settings saved'));
      showValidationSummary([]);
    } catch (err){
      setStatus(t('status.save_failed', 'Save failed'), true);
    } finally {
      if(btn && btn.elements && btn.elements.length){
        btn.prop('disabled', false);
      }
    }
  }

  async function startTraining(projectName){
    try {
      setStatus(t('status.starting_training', 'Starting training...'));
      const response = await comm.startTraining({ project_name: projectName });
      if(!response || response.started === false){
        const message = response?.error || t('status.training_start_failed', 'Training start failed');
        setStatus(message, true);
        if(response?.error){
          log('Training start failed:', response.error);
        }
        return;
      }

      setStatus(t('status.training_started', 'Training started'));

      if(response.training_id){
        resetTrainingTelemetry(response.training_id);
        persistTrainingContext(response.training_id, response.project || projectName);
        state.pendingAutoProject = response.project || projectName;
        state.pendingAutoTrainingId = response.training_id;
        state.trainingStatus = {
          training_id: response.training_id,
          status: response.status || 'starting',
          project: response.project || projectName
        };
        renderTrainingStatus();
        updateTrainingBadge();
        applyFooterStatus();
      }

      pollStatus();
      refreshProjects();
    } catch (err){
      setStatus(t('status.training_start_failed', 'Training start failed'), true);
      log('Training start error:', err);
    }
  }

  async function stopTraining(trainingId){
    const telemetry = ensureTrainingTelemetry();
    const fallbackId = telemetry.activeTrainingId || state.trainingStatus?.training_id || null;
    const targetId = trainingId || fallbackId;
    if(!targetId){
      setStatus(t('status.training_stop_failed', 'Training stop failed'), true);
      return;
    }

    try {
      setStatus(t('status.training_stopping', 'Stopping training...'));
      const response = await comm.stopTraining(targetId);
      if(response && response.stopped){
        setStatus(t('status.training_stop_requested', 'Training stop requested'));
        pollStatus();
      } else {
        const message = response?.error || t('status.training_stop_failed', 'Training stop failed');
        setStatus(message, true);
      }
    } catch (err){
      setStatus(t('status.training_stop_failed', 'Training stop failed'), true);
      log('Training stop error:', err);
    }
  }

  function registerField(path, el, schema, getter){
    state.fieldIndex[path] = { el, schema, getter };
    if(path === 'training.model_type'){
      state.modelTypeField = { el, schema, getter };
      if(state.modelNameField){
        el.on('change', updateModelNameOptions);
      }
    }

    if(path === 'training.model_name' && state.modelTypeField){
      state.modelTypeField.el.on('change', updateModelNameOptions);
    }
  }

  function updateModelNameOptions(){
    if(!state.modelNameField || !state.modelTypeField) return;
    const modelType = state.modelTypeField.el.val() || 'resnet';
    const variants = state.modelNameField.schema.model_type_variants?.[modelType] || state.modelNameField.schema.enum;
    const currentVal = state.modelNameField.sel.val();

    state.modelNameField.sel.html('');
    variants.forEach(opt => Q('<option>').attr('value', opt).text(opt).appendTo(state.modelNameField.sel));

    if(variants.includes(currentVal)){
      state.modelNameField.sel.val(currentVal);
    } else {
      state.modelNameField.sel.val(variants[0]);
      state.modelNameField.onChange(variants[0]);
      markDirty();
    }

    state.modelNameField.availableOptions = variants;
  }

  function collectCurrentValues(){
    Object.entries(state.fieldIndex).forEach(([path, meta]) => {
      try {
        const val = meta.getter();
        assignPathValue(state.config, path, val);
      } catch {}
    });
  }

  function assignPathValue(root, path, value){
    const parts = path.split('.');
    let cursor = root;
    for(let i = 0; i < parts.length - 1; i += 1){
      const key = parts[i];
      if(!(key in cursor)){
        cursor[key] = {};
      }
      cursor = cursor[key];
    }
    cursor[parts[parts.length - 1]] = value;
  }

  function runValidation(){
    const errors = [];
    Object.entries(state.fieldIndex).forEach(([path, meta]) => {
      const value = meta.getter();
      const schema = meta.schema;
      const fieldEl = meta.el.closest ? meta.el.closest('.field') : null;
      if(fieldEl){
        fieldEl.classList.remove('error');
      }

      if(schema.type === 'number' || schema.type === 'integer'){
        if(value !== null && value !== undefined){
          if(typeof value !== 'number' || Number.isNaN(value)) errors.push({ path, message: 'Not a number' });
          if(schema.minimum !== undefined && value < schema.minimum) errors.push({ path, message: 'Below minimum' });
          if(schema.maximum !== undefined && value > schema.maximum) errors.push({ path, message: 'Above maximum' });
        }
      }
      if(schema.enum && value !== undefined && value !== null && !schema.enum.includes(value)){
        errors.push({ path, message: 'Invalid enum' });
      }
      if(isPathRequired(path) && (value === null || value === undefined || value === '')){
        errors.push({ path, message: 'Required' });
      }
    });

    errors.forEach(error => {
      const meta = state.fieldIndex[error.path];
      if(meta){
        const wrap = meta.el.closest ? meta.el.closest('.field') : null;
        if(wrap){
          wrap.classList.add('error');
        }
      }
    });

    showValidationSummary(errors);
    return errors;
  }

  function isPathRequired(path){
    const parts = path.split('.');
    if(parts.length < 2) return false;
    const sectionName = parts[0];
    const key = parts[parts.length - 1];
    const sectionSchema = state.schema?.properties?.[sectionName];
    if(!sectionSchema) return false;
    if(Array.isArray(sectionSchema.required) && sectionSchema.required.includes(key)){
      return true;
    }
    return false;
  }

  const storedTrainingContext = readStoredTrainingContext();
  if(storedTrainingContext && storedTrainingContext.trainingId){
    resetTrainingTelemetry(storedTrainingContext.trainingId);
    state.pendingAutoProject = storedTrainingContext.project || null;
    state.pendingAutoTrainingId = storedTrainingContext.trainingId;
  } else {
    ensureTrainingTelemetry();
  }

  applyFooterStatus();

  hs.actions = {
    setStatus,
    showValidationSummary,
    markDirty,
    navInit,
    initLanguageSelector,
    switchLanguage,
    refreshProjects,
  showProjectCreateDialog,
  closeProjectCreateDialog,
    loadProject,
    saveTrainingConfig,
    syncAugmentationsFromConfig,
    getAugmentationOptions,
    isAugmentationEnabled,
    setAugmentationEnabled,
    getCustomAugmentations,
    getAugmentationFieldValue,
    setAugmentationFieldValue,
    renderAugmentationPreview,
    requestAugmentationPreview,
    refreshDatasetInfo,
    renderUpdatesState,
    runUpdatesCheck,
    applySystemUpdates,
    pollStatus,
    generateHeatmap,
    saveSystemConfig,
    startTraining,
    stopTraining,
    registerField,
    updateModelNameOptions,
    collectCurrentValues,
    assignPathValue,
    runValidation,
    isPathRequired,
    initDocsPage,
    openDoc,
    handleDocsLinkClick
  };
})();
