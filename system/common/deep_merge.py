from __future__ import annotations
import json
from pathlib import Path
from typing import Any


def _strip_jsonc_comments(src: str) -> str:
	"""
	Strip // and /* */ comments from JSONC while preserving string contents.
	Simple FSM-based stripper that avoids removing comment tokens inside JSON strings.
	"""
	out: list[str] = []
	i = 0
	n = len(src)
	in_string = False
	escape = False

	while i < n:
		ch = src[i]

		if in_string:
			out.append(ch)
			if escape:
				escape = False
			elif ch == "\\":
				escape = True
			elif ch == '"':
				in_string = False
			i += 1
			continue

		# Not in a string
		if ch == '"':
			in_string = True
			out.append(ch)
			i += 1
			continue

		# Line comment //
		if ch == "/" and i + 1 < n and src[i + 1] == "/":
			i += 2
			while i < n and src[i] not in ("\n", "\r"):
				i += 1
			continue

		# Block comment /* ... */
		if ch == "/" and i + 1 < n and src[i + 1] == "*":
			i += 2
			while i + 1 < n and not (src[i] == "*" and src[i + 1] == "/"):
				i += 1
			i += 2 if i + 1 < n else 1
			continue

		out.append(ch)
		i += 1

	return "".join(out)


def _load_json_from_path(path: str) -> Any:
	"""
	Load JSON/JSONC from a filesystem path. Expands user (~) and resolves relative paths.
	Note: json.loads requires the whole document in memory. For very large files consider a streaming parser.
	"""
	p = Path(path).expanduser()
	if not p.is_absolute():
		p = p.resolve()
	if not p.exists():
		raise FileNotFoundError(f"JSON file not found: {path}")

	with p.open("r", encoding="utf-8") as fh:
		content = fh.read()

	try:
		return json.loads(content)
	except json.JSONDecodeError:
		clean = _strip_jsonc_comments(content)
		return json.loads(clean)


def load_json_optional(path: str | Path, default: Any = None) -> Any:
	"""Load JSON/JSONC from disk, returning a default on failure or missing file.

	This mirrors the behavior used in settings/localization loaders while keeping
	the parsing logic centralized.
	"""
	try:
		p = Path(path)
		if not p.exists():
			return default
		return deep_merge_json(str(p))
	except Exception:
		return default


def _parse_jsonish(value: Any) -> Any:
	"""
	Parse a JSON-ish input.

	Accepted:
	- dict/list/primitives -> returned as-is
	- JSON string
	- JSONC string (JSON with comments)
	- Filesystem path to a file containing JSON/JSONC

	The function first tries JSON parsing when the string starts with '{' or '[' (after whitespace),
	then falls back to file loading, then a final attempt with JSONC stripping.
	"""
	if not isinstance(value, str):
		return value

	trimmed = value.lstrip()
	looks_like_json = bool(trimmed) and trimmed[0] in ("{", "[")

	if looks_like_json:
		try:
			return json.loads(value)
		except json.JSONDecodeError:
			try:
				return json.loads(_strip_jsonc_comments(value))
			except json.JSONDecodeError:
				# fall through to path-checking
				pass

	p = Path(value).expanduser()
	if p.exists():
		return _load_json_from_path(str(p))

	# Last attempt: maybe JSONC with comments even if it didn't parse earlier
	try:
		return json.loads(_strip_jsonc_comments(value))
	except Exception as exc:
		raise ValueError(f"Unable to parse input as JSON/JSONC or find a file at path: {value}") from exc


def _deep_merge_two(left: Any, right: Any, replace_lists: bool = True) -> Any:
	"""
	Merge right into left and return a new value (inputs are not mutated).

	Rules:
	- If both are dicts: merge keys recursively. When both values are dicts they merge recursively.
	- If both are lists and replace_lists is True: right replaces left.
	- Otherwise: right wins (overwrite).
	"""
	if isinstance(left, dict) and isinstance(right, dict):
		result = dict(left)
		for key, right_val in right.items():
			if key in result:
				result[key] = _deep_merge_two(result[key], right_val, replace_lists=replace_lists)
			else:
				result[key] = right_val
		return result

	# Lists: by default perform unique union (preserve order).
	# If replace_lists is True, fall back to replacement behavior for callers who want it.
	if isinstance(left, list) and isinstance(right, list):
		if replace_lists:
			return right

		# Merge lists while keeping items unique. Since list items may be unhashable
		# (dicts/lists), perform a linear dedupe using recursive equality.
		result: list[Any] = []

		def _items_equal(a: Any, b: Any) -> bool:
			"""Recursively compare JSON-like structures for equality."""
			if a is b:
				return True
			if type(a) != type(b):
				return False
			if isinstance(a, dict):
				if a.keys() != b.keys():
					return False
				return all(_items_equal(a[k], b[k]) for k in a.keys())
			if isinstance(a, list):
				if len(a) != len(b):
					return False
				return all(_items_equal(x, y) for x, y in zip(a, b))
			return a == b

		for item in left:
			result.append(item)

		for item in right:
			found = False
			for existing in result:
				if _items_equal(item, existing):
					found = True
					break
			if not found:
				result.append(item)

		return result

	# Otherwise right wins
	return right


def deep_merge_json(base: Any, *others: Any, replace_lists: bool = False) -> Any:
	"""
	Deep-merge multiple JSON-like sources with an explicit base.

	Usage:
		merged = deep_merge_json(json1, json2, json3)
	where:
	- json1 is the base (dict/list/primitive or JSON string or path)
	- json2, json3, ... are merged into the base in order

	Parameters:
	- base: the base JSON-like input (dict/list/primitive or JSON string or path)
	- *others: additional JSON-like inputs to merge into base (in order)
	- replace_lists: if True (default) lists from later inputs replace earlier lists.
	                 (If you want list concatenation/dedupe, tell me and I'll add an explicit strategy param.)

	Behavior:
	- Later sources overwrite earlier ones on conflicts (applied left-to-right).
	- The function parses and merges on-the-fly (doesn't keep all parsed sources in memory).
	- Accepts dicts, lists, primitives, JSON strings, JSONC strings, or filesystem paths.

	Returns:
	- The merged JSON-like structure (dict/list/primitive).
	"""
	# Validate base presence
	if base is None:
		# None as an explicit base is allowed; but require that at least one positional argument present
		# (this protects accidental empty-call usage)
		if not others:
			raise ValueError("deep_merge_json requires at least one argument (base).")
	# Parse base
	try:
		acc = _parse_jsonish(base)
	except Exception as exc:
		raise ValueError(f"Failed to parse base input: {exc}") from exc

	# Merge remaining sources in order
	for idx, src in enumerate(others, start=1):
		try:
			parsed = _parse_jsonish(src)
		except Exception as exc:
			raise ValueError(f"Failed to parse source #{idx}: {exc}") from exc
		acc = _deep_merge_two(acc, parsed, replace_lists=replace_lists)

	return acc


__all__ = ["deep_merge_json", "load_json_optional"]
