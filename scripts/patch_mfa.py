"""Fix MFA 3.3.9's unconditional underscore stripping for non-positional phones.

Run with the MFA environment's Python. A Telugu label such as te_aa must not
be interpreted as base 'te' with position 'aa'. Only B/I/E/S are Kaldi position
suffixes. This changes label parsing, not acoustic weights or phone boundaries.
"""

from importlib.metadata import version
from importlib.util import find_spec
from pathlib import Path

if version("montreal-forced-aligner") != "3.3.9":
    raise SystemExit(
        "This patch is verified only for MFA 3.3.9; inspect newer versions before patching."
    )
package = Path(find_spec("montreal_forced_aligner").origin).parent
path = package / "helper.py"
source = path.read_text()
start = source.index("def split_phone_position(")
end = source.index("\ndef ", start + 4)
function = source[start:end]
marker = "    return phone, pos"
replacement = """    # LTRC: underscores are allowed inside non-positional phone names.
    if pos not in {"B", "I", "E", "S"}:
        return phone_label, None
    return phone, pos"""
if "# LTRC:" not in function:
    if function.count(marker) != 1:
        raise SystemExit(
            "Unexpected MFA helper implementation; refusing an ambiguous patch."
        )
    source = source[:start] + function.replace(marker, replacement) + source[end:]
    path.write_text(source)
from montreal_forced_aligner.helper import split_phone_position  # noqa: E402

assert split_phone_position("te_aa") == ("te_aa", None)
assert split_phone_position("te_r_v") == ("te_r_v", None)
assert split_phone_position("AA0_B") == ("AA0", "B")
assert split_phone_position("te_aa_E") == ("te_aa", "E")
print("MFA phone-position parser verified:", path)
