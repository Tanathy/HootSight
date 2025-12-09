from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

from system.common.deep_merge import deep_merge_json, load_json_optional
from system.log import info, error


def _determine_root() -> Path:
    run_module = sys.modules.get('run') or sys.modules.get('__main__')
    if run_module and hasattr(run_module, 'ROOT'):
        try:
            root_path = getattr(run_module, 'ROOT')
            if isinstance(root_path, Path) and root_path.exists():
                return root_path
        except Exception:
            pass

    return Path(__file__).resolve().parents[1]


ROOT = _determine_root()
SYSTEM_CONFIG_PATH = ROOT / "config" / "system_config.json"
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.json"
PACKAGES_CONFIG_PATH = ROOT / "config" / "packages.jsonc"
MAPPINGS_CONFIG_PATH = ROOT / "config" / "mappings.json"
LOCALIZATIONS_DIR = ROOT / "config" / "localizations"


def _first_existing(*paths: Path) -> str | None:
    for p in paths:
        if p.exists():
            return str(p)
    return None




def _load_file_if_exists(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        return str(path)
    except Exception:
        return None


def reload_settings() -> Any:
    system_cfg = _load_file_if_exists(SYSTEM_CONFIG_PATH)
    training_cfg = _load_file_if_exists(DEFAULT_CONFIG_PATH)

    # Start with empty dict
    merged = {}
    
    # Merge system config first (general, api, ui, paths, system, memory, dataset_editor)
    if system_cfg:
        merged = deep_merge_json(system_cfg)
    
    # Then merge training config (training, dataset, activations, augmentations, optimizers, etc.)
    if training_cfg:
        training_data = deep_merge_json(training_cfg)
        # Manual merge to combine both configs
        for key, value in training_data.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = {**merged[key], **value}
            else:
                merged[key] = value

    global SETTINGS
    SETTINGS = merged
    reload_packages()
    reload_mappings()
    try:
        reload_localization()
    except Exception:
        pass
    
    try:
        import torch
        try:
            cache_dir = SETTINGS['paths']['cache_dir']
        except Exception:
            raise ValueError("Missing 'paths.cache_dir' in config/system_config.json")
        cache_path = ROOT / cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_path))
        import os
        os.environ['TORCH_HOME'] = str(cache_path)
    except ImportError:
        pass
    
    return merged


def _collect_module_package_paths() -> list[str]:
    return []


def reload_mappings() -> Any:
    global MAPPINGS

    if MAPPINGS_CONFIG_PATH.exists():
        merged = deep_merge_json(str(MAPPINGS_CONFIG_PATH))
    else:
        merged = {}

    MAPPINGS = merged or {}
    return MAPPINGS


def reload_packages() -> Any:
    global PACKAGES

    base = None
    if PACKAGES_CONFIG_PATH.exists():
        base = str(PACKAGES_CONFIG_PATH)

    module_paths = _collect_module_package_paths()

    if base is None and not module_paths:
        merged = {}
    else:
        sources = [s for s in ([base] if base else []) + module_paths]
        merged = None
        for idx, src in enumerate(sources):
            if idx == 0:
                merged = deep_merge_json(src, replace_lists=False)
            else:
                merged = deep_merge_json(merged, src, replace_lists=False)

    PACKAGES = merged or {}
    return PACKAGES


def _collect_module_localization_paths(lang: str) -> list[str]:
    return []


def reload_localization() -> Any:
    global LOCALIZATION, _language_data

    lang_code = DEFAULT_LANG_CODE
    if isinstance(SETTINGS, dict):
        lang_code = SETTINGS.get("general", {}).get("language") or lang_code

    lang_path = get_language_file_path(lang_code)
    data = load_json_optional(lang_path, {}) or {}

    if _premerged_language_data and isinstance(_premerged_language_data, dict):
        data = deep_merge_json(data, _premerged_language_data, replace_lists=False)

    data = _merge_enabled_module_languages(data, lang_code)

    LOCALIZATION = data
    _language_data = data
    return LOCALIZATION


PACKAGES: Any = {}


MAPPINGS: Any = {}


LOCALIZATION_DIR = LOCALIZATIONS_DIR
LOCALIZATION: Any = {}


SETTINGS: Any = {}
PACKAGES = PACKAGES
MAPPINGS = MAPPINGS
LOCALIZATION = LOCALIZATION
reload_settings()

DEFAULT_LANG_CODE = "en"
_language_data: Any = None
_premerged_language_data: Any = None


def get_language_file_path(lang_code: str) -> str:
    found = _first_existing(
        ROOT / "config" / "localizations" / f"{lang_code}.json",
        ROOT / "config" / "localizations" / f"{lang_code}.jsonc",
    )
    return found or str(ROOT / "config" / "languages" / f"{lang_code}.json")


def _merge_enabled_module_languages(base_data: dict, lang_code: str) -> dict:
    return dict(base_data)


def set_premerged_language_data(data: dict) -> None:
    global _premerged_language_data
    _premerged_language_data = data


def load_language_data() -> dict:
    global _language_data
    if _language_data is None:
        lang_code = DEFAULT_LANG_CODE
        if isinstance(SETTINGS, dict):
            lang_code = SETTINGS.get("general", {}).get("language") or lang_code

        lang_path = get_language_file_path(lang_code)
        data = load_json_optional(lang_path, {}) or {}

        if _premerged_language_data and isinstance(_premerged_language_data, dict):
            data = deep_merge_json(data, _premerged_language_data, replace_lists=False)

        data = _merge_enabled_module_languages(data, lang_code)

        _language_data = data
    return _language_data


def _persist_language_to_system_config(lang_code: str) -> None:
    import json
    try:
        SYSTEM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        if SYSTEM_CONFIG_PATH.exists():
            with open(SYSTEM_CONFIG_PATH, 'r', encoding='utf-8') as f:
                system_config = json.load(f)
        else:
            system_config = {}
        
        if "general" not in system_config:
            system_config["general"] = {}
        system_config["general"]["language"] = lang_code
        
        with open(SYSTEM_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(system_config, f, indent=4, ensure_ascii=False)
        
        info(f"Language persisted to system config: {lang_code}")
    except Exception as ex:
        error(f"Failed to persist language to system config: {ex}")


def _get_localized_text(data: dict, key: str, **kwargs) -> str | None:
    if not data or not isinstance(data, dict):
        return None
    parts = key.split('.')
    cur = data
    for part in parts:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    if isinstance(cur, str):
        try:
            return cur.format(**kwargs)
        except Exception:
            return cur
    return None


def lang(key: str, **kwargs) -> str:
    data = load_language_data()
    
    if data:
        result = _get_localized_text(data, key, **kwargs)
        if result is not None:
            return result
    
    try:
        lang_code = DEFAULT_LANG_CODE
        if isinstance(SETTINGS, dict):
            lang_code = SETTINGS.get("general", {}).get("language") or lang_code
        
        if lang_code != DEFAULT_LANG_CODE:
            en_path = get_language_file_path(DEFAULT_LANG_CODE)
            en_data = load_json_optional(en_path, {}) or {}
            if en_data:
                result = _get_localized_text(en_data, key, **kwargs)
                if result is not None:
                    return result
    except Exception:
        pass
    
    return key


def _extract_language_name(file_path: Path, code: str) -> str:
    data = load_json_optional(str(file_path), {}) or {}
    if isinstance(data, dict):
        name = data.get("language_name")
        if isinstance(name, str):
            cleaned = name.strip()
            if cleaned:
                return cleaned
    return code.upper()


def available_languages() -> list[dict[str, str]]:
    directories = [
        ROOT / "config" / "languages",
        ROOT / "config" / "localizations",
    ]

    discovered: dict[str, dict[str, str]] = {}

    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            continue

        for file_path in sorted(directory.iterdir(), key=lambda p: p.name.lower()):
            if not file_path.is_file() or file_path.suffix.lower() not in (".json", ".jsonc"):
                continue

            code = file_path.stem.lower()
            if not code:
                continue

            name = _extract_language_name(file_path, code)

            existing = discovered.get(code)
            should_override = (
                existing is None
                or directory.name.lower() == "localizations"
                or existing["name"].upper() == code.upper()
            )

            if should_override:
                discovered[code] = {"code": code, "name": name}

    if not discovered:
        return [{"code": "en", "name": "English"}]

    return sorted(discovered.values(), key=lambda item: (item["name"].casefold(), item["code"]))


def switch_language(lang_code: str) -> dict:
    if isinstance(SETTINGS, dict):
        if "general" not in SETTINGS:
            SETTINGS["general"] = {}
        SETTINGS["general"]["language"] = lang_code
    
    _persist_language_to_system_config(lang_code)
    
    global _language_data
    _language_data = None
    
    data = load_language_data()
    global LOCALIZATION
    LOCALIZATION = data
    return data


def save_system_settings() -> None:
    """Save current system settings to system_config.json."""
    import json
    
    SYSTEM_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Extract only system-related settings
    system_keys = ["general", "api", "ui", "paths", "system", "memory", "dataset_editor"]
    system_settings = {k: v for k, v in SETTINGS.items() if k in system_keys}
    
    try:
        with open(SYSTEM_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(system_settings, f, indent=4, ensure_ascii=False)
        info(f"System settings saved to {SYSTEM_CONFIG_PATH}")
    except Exception as ex:
        error(f"Failed to save system settings: {str(ex)}")
        raise


__all__ = [
    "SETTINGS",
    "reload_settings",
    "save_system_settings",
    "PACKAGES",
    "reload_packages",
    "MAPPINGS",
    "reload_mappings",
    "LOCALIZATION",
    "reload_localization",
    "get_language_file_path",
    "load_language_data",
    "lang",
    "switch_language",
    "available_languages",
    "set_premerged_language_data",
]
def get_current_language() -> str:
    lang_code = DEFAULT_LANG_CODE
    if isinstance(SETTINGS, dict):
        lang_code = SETTINGS.get("general", {}).get("language") or lang_code
    return lang_code

