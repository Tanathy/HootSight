from system.common.dice import roll

__all__ = ["roll"]

def hex16() -> str:
	"""Return a random 16-character lowercase hexadecimal string.

	Uses local entropy mixing via roll() to generate 8 random bytes, then
	formats as 16 hex chars. Not cryptographic; good enough for stable
	identifiers in the app.
	"""
	# Generate 8 bytes by rolling 8 times 0..255
	vals = [int(roll(0, 255)) for _ in range(8)]
	return ''.join(f"{b:02x}" for b in vals)

__all__ = ["roll", "hex16"]
