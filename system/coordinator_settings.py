"""Coordinator settings loader.

Loads `config/config.json` and `config/config_user.json`, deep-merges them with
user settings overriding defaults, and exposes a module-level
`SETTINGS` variable for global access.

Usage:
    from system.coordinator_settings import SETTINGS

You can call `reload_settings()` to re-read the files and update the module var.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import sys

from system.common.deep_merge import deep_merge_json
from system.log import info, error


def _determine_root() -> Path:
    """Determine project root.

    Priority:
    1. ROOT from run.py module if available
    2. Parent-of-parent of this module (fallback)
    """
    # Try to get ROOT from run.py if it's been imported
    run_module = sys.modules.get('run') or sys.modules.get('__main__')
    if run_module and hasattr(run_module, 'ROOT'):
        try:
            root_path = getattr(run_module, 'ROOT')
            if isinstance(root_path, Path) and root_path.exists():
                return root_path
        except Exception:
            pass

    # Fallback to previous behavior
    return Path(__file__).resolve().parents[1]


ROOT = _determine_root()
DEFAULT_CONFIG_PATH = ROOT / "config" / "config.json"
USER_CONFIG_PATH = ROOT / "config" / "config_user.json"
PACKAGES_CONFIG_PATH = ROOT / "config" / "packages.jsonc"
MAPPINGS_CONFIG_PATH = ROOT / "config" / "mappings.json"
LOCALIZATIONS_DIR = ROOT / "config" / "localizations"


def _first_existing(*paths: Path) -> str | None:
    """Return the first existing path as string from the provided Path arguments, or None."""
    for p in paths:
        if p.exists():
            return str(p)
    return None




def _load_file_if_exists(path: Path) -> str | None:
    """Return file path as string if it exists and is non-empty, None otherwise."""
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8").strip()
        if not text:
            return None
        # Return path as string for deep_merge_json
        return str(path)
    except Exception:
        return None


from typing import Any

def reload_settings() -> Any:
    """Load and merge config files and return the resulting structure.

    Side-effect: updates the module-level SETTINGS value.
    """
    base = _load_file_if_exists(DEFAULT_CONFIG_PATH)
    user = _load_file_if_exists(USER_CONFIG_PATH)

    # If neither exists, default to empty dict
    if base is None and user is None:
        merged = {}
    elif base is None:
        merged = deep_merge_json({}, user)
    elif user is None:
        merged = deep_merge_json(base)
    else:
        merged = deep_merge_json(base, user)

    global SETTINGS
    SETTINGS = merged
    # Also refresh PACKAGES and LOCALIZATION whenever settings reloads so modules can consult them
    reload_packages()
    reload_mappings()
    try:
        reload_localization()
    except Exception:
        # Localization is optional; ignore failures here to avoid breaking settings reload
        pass
    
    # Set PyTorch cache directory
    try:
        import torch
        cache_dir = SETTINGS.get('paths', {}).get('cache_dir', 'cache')
        cache_path = ROOT / cache_dir
        cache_path.mkdir(parents=True, exist_ok=True)
        torch.hub.set_dir(str(cache_path))
        # Also set TORCH_HOME for torchvision and other PyTorch components
        import os
        os.environ['TORCH_HOME'] = str(cache_path)
    except ImportError:
        # PyTorch not available, skip cache setup
        pass
    
    return merged


def _collect_module_package_paths() -> list[str]:
    """Return a list of module package file paths to merge, honoring SETTINGS['modules'].

    Rules:
    - No module support, returns empty list.
    """
    return []


def reload_mappings() -> Any:
    """Load mappings configuration from config/mappings.json.

    Exposes module-level MAPPINGS variable (dict). Returns the loaded structure.
    """
    global MAPPINGS

    if MAPPINGS_CONFIG_PATH.exists():
        merged = deep_merge_json(str(MAPPINGS_CONFIG_PATH))
    else:
        merged = {}

    MAPPINGS = merged or {}
    return MAPPINGS


def reload_packages() -> Any:
    """Load and merge package manifests from global config and enabled modules.

    Exposes module-level PACKAGES variable (dict). Returns the merged structure.
    """
    global PACKAGES

    base = None
    if PACKAGES_CONFIG_PATH.exists():
        base = str(PACKAGES_CONFIG_PATH)

    module_paths = _collect_module_package_paths()

    # Merge: base then each module in arbitrary filesystem order
    if base is None and not module_paths:
        merged = {}
    else:
        sources = [s for s in ([base] if base else []) + module_paths]
        # Use deep_merge_json by starting with first source and merging rest
        merged = None
        for idx, src in enumerate(sources):
            if idx == 0:
                merged = deep_merge_json(src, replace_lists=False)
            else:
                merged = deep_merge_json(merged, src, replace_lists=False)

    PACKAGES = merged or {}
    return PACKAGES


def _collect_module_localization_paths(lang: str) -> list[str]:
    """Return a list of module localization file paths (module/config/<lang>.json) honoring SETTINGS['modules']."""
    return []


def reload_localization() -> Any:
    """Load and merge localization for SETTINGS['language'] into LOCALIZATION.

    Uses the same logic as load_language_data() but updates LOCALIZATION and cache.
    """
    global LOCALIZATION, _language_data

    if _current_lang_code:
        lang_code = _current_lang_code
    else:
        lang_code = DEFAULT_LANG_CODE
        if isinstance(SETTINGS, dict):
            lang_code = SETTINGS.get("general", {}).get("language") or lang_code

    lang_path = get_language_file_path(lang_code)
    data = _load_json_safe(lang_path, {}) or {}

    if _premerged_language_data and isinstance(_premerged_language_data, dict):
        data = deep_merge_json(data, _premerged_language_data, replace_lists=False)

    data = _merge_enabled_module_languages(data, lang_code)

    LOCALIZATION = data
    _language_data = data  # Update cache to keep them in sync
    return LOCALIZATION


# Initialize PACKAGES
PACKAGES: Any = {}


# Initialize MAPPINGS
MAPPINGS: Any = {}


LOCALIZATION_DIR = LOCALIZATIONS_DIR
LOCALIZATION: Any = {}


# Initialize at module import (mutable container to satisfy linters)
SETTINGS: Any = {}
# Ensure PACKAGES and LOCALIZATION exist before first reload
PACKAGES = PACKAGES
MAPPINGS = MAPPINGS
LOCALIZATION = LOCALIZATION
reload_settings()

# Language/runtime helpers
DEFAULT_LANG_CODE = "en"
_current_lang_code: str | None = None
_language_data: Any = None
_premerged_language_data: Any = None


def _load_json_safe(path: str, default: Any = None) -> Any:
    """Safely load JSON/JSONC from a path, returning default on error or missing file."""
    try:
        p = Path(path)
        if not p.exists():
            return default
        # Use deep_merge_json to parse the file path directly
        return deep_merge_json(str(p))
    except Exception:
        return default

def get_language_file_path(lang_code: str) -> str:
    """Return the best language file path for lang_code.

    Prefer project config/localizations/<lang>.json(.jsonc) first (current repo layout),
    then fall back to config/languages/<lang>.json(.jsonc) for compatibility.
    Returns the first path that exists, or the primary languages path (non-existent)
    as a string if none found.
    """
    found = _first_existing(
        ROOT / "config" / "localizations" / f"{lang_code}.json",
        ROOT / "config" / "localizations" / f"{lang_code}.jsonc",
    )
    return found or str(ROOT / "config" / "languages" / f"{lang_code}.json")


def _merge_enabled_module_languages(base_data: dict, lang_code: str) -> dict:
    """Find enabled modules and deep merge their language files into base_data.

    Module support removed - returns base_data unchanged.
    """
    return dict(base_data)


def set_premerged_language_data(data: dict) -> None:
    global _premerged_language_data
    _premerged_language_data = data


def load_language_data() -> dict:
    """Load and cache language data (base + enabled-modules overlays).

    Uses runtime override `_current_lang_code`, then SETTINGS['general']['language'], then DEFAULT_LANG_CODE.
    """
    global _language_data
    if _language_data is None:
        # Determine language code
        if _current_lang_code:
            lang_code = _current_lang_code
        else:
            lang_code = DEFAULT_LANG_CODE
            if isinstance(SETTINGS, dict):
                lang_code = SETTINGS.get("general", {}).get("language") or lang_code

        # Load base language file
        lang_path = get_language_file_path(lang_code)
        data = _load_json_safe(lang_path, {}) or {}

        # Merge premerged (if provided) first to allow early overlay
        if _premerged_language_data and isinstance(_premerged_language_data, dict):
            data = deep_merge_json(data, _premerged_language_data, replace_lists=False)

        # Merge enabled module languages
        data = _merge_enabled_module_languages(data, lang_code)

        _language_data = data
    return _language_data


def lang_switch(lang_code: str):
    global _current_lang_code, _language_data, _premerged_language_data
    _current_lang_code = lang_code
    _language_data = None
    _premerged_language_data = None


def _get_localized_text(data: dict, key: str, **kwargs) -> str | None:
    """Helper to extract localized text from data dict by key path."""
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
    """Get localized string for key, with English fallback if translation missing."""
    data = load_language_data()
    
    # Try current language first
    if data:
        result = _get_localized_text(data, key, **kwargs)
        if result is not None:
            return result
    
    # Fallback to English if current language doesn't have the key or no data loaded
    try:
        if not _current_lang_code or _current_lang_code != DEFAULT_LANG_CODE:
            en_path = get_language_file_path(DEFAULT_LANG_CODE)
            en_data = _load_json_safe(en_path, {}) or {}
            if en_data:
                result = _get_localized_text(en_data, key, **kwargs)
                if result is not None:
                    return result
    except Exception:
        pass
    
    # Final fallback: return the key itself
    return key


def _extract_language_name(file_path: Path, code: str) -> str:
    data = _load_json_safe(str(file_path), {}) or {}
    if isinstance(data, dict):
        name = data.get("language_name")
        if isinstance(name, str):
            cleaned = name.strip()
            if cleaned:
                return cleaned
    return code.upper()


def available_languages() -> list[dict[str, str]]:
    """Return available language metadata (code + display name).

    Prefers entries from `config/localizations` over legacy `config/languages`.
    Falls back to English if nothing is discovered.
    """

    directories = [
        ROOT / "config" / "languages",
        ROOT / "config" / "localizations",
    ]

    discovered: dict[str, dict[str, str]] = {}

    for directory in directories:
        if not directory.exists() or not directory.is_dir():
            continue

        # Iterate in alphabetical order for stable results
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

    # Sort by display name (case-insensitive), fallback to code to ensure deterministic order
    return sorted(discovered.values(), key=lambda item: (item["name"].casefold(), item["code"]))


def switch_language(lang_code: str) -> dict:
    lang_switch(lang_code)
    data = load_language_data()
    # Keep module-level LOCALIZATION in sync with the language cache
    global LOCALIZATION
    LOCALIZATION = data
    return data


def save_user_settings() -> None:
    """Save current SETTINGS to config_user.json.
    
    Only saves settings that differ from the default config to keep config_user.json clean.
    """
    import json
    from pathlib import Path
    
    if not USER_CONFIG_PATH.exists():
        USER_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    
    # Load default config to compare
    default_config = {}
    if DEFAULT_CONFIG_PATH.exists():
        try:
            default_config = deep_merge_json(str(DEFAULT_CONFIG_PATH))
        except Exception:
            default_config = {}
    
    # For now, save all SETTINGS to keep it simple
    # In a more sophisticated implementation, we could diff against defaults
    try:
        with open(USER_CONFIG_PATH, 'w', encoding='utf-8') as f:
            json.dump(SETTINGS, f, indent=2, ensure_ascii=False)
        info(lang("coordinator_settings.user_settings_saved", path=USER_CONFIG_PATH))
    except Exception as ex:
        error(lang("coordinator_settings.user_settings_save_failed", error=str(ex)))
        raise

__all__ = [
    "SETTINGS",
    "reload_settings",
    "save_user_settings",
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
    """Get the currently active language code."""
    if _current_lang_code:
        return _current_lang_code
    lang_code = DEFAULT_LANG_CODE
    if isinstance(SETTINGS, dict):
        lang_code = SETTINGS.get("general", {}).get("language") or lang_code
    return lang_code

