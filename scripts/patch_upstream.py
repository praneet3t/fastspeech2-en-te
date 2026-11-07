"""Apply the small, documented changes to a pristine pinned upstream copy.

The resulting vendor files are committed; this script explains the exact patch.
"""

from pathlib import Path

root = Path(__file__).resolve().parents[1] / "vendor/fastspeech2"
p = root / "transformer/Models.py"
s = p.read_text().replace("from text.symbols import symbols\n", "")
s = s.replace("n_src_vocab = len(symbols) + 1", 'n_src_vocab = config["vocab_size"]')
s = s.replace(
    "if not self.training and src_seq.shape[1] > self.max_seq_len:",
    "if src_seq.shape[1] > self.max_seq_len:",
)
s = s.replace(
    "if not self.training and enc_seq.shape[1] > self.max_seq_len:",
    "if enc_seq.shape[1] > self.max_seq_len:",
)
p.write_text(s)
p = root / "model/modules.py"
s = p.read_text().replace(
    "x, energy_target, src_mask, p_control", "x, energy_target, src_mask, e_control"
)
s = s.replace(
    "x, energy_target, mel_mask, p_control", "x, energy_target, mel_mask, e_control"
)
s = s.replace(
    "return output, torch.LongTensor(mel_len).to(device)",
    "return output, torch.tensor(mel_len, dtype=torch.long, device=x.device)",
)
# Expansion dominates Python overhead; repeat_interleave implements identical math.
a = s.index("        out = list()\n", s.index("    def expand("))
b = s.index("\n    def forward(", a)
s = (
    s[:a]
    + """        durations = predicted.long()
        if (durations < 0).any():
            raise ValueError("Negative duration")
        return torch.repeat_interleave(batch, durations, dim=0)
"""
    + s[b:]
)
s = s.replace(
    "            x, mel_len = self.length_regulator(x, duration_rounded, max_len)",
    """            duration_rounded = duration_rounded.masked_fill(src_mask, 0)
            if not torch.isfinite(duration_rounded).all():
                raise ValueError("Non-finite predicted durations")
            if (duration_rounded.sum(1) == 0).any():
                raise ValueError("All predicted durations are zero; model is not ready for inference")
            if (duration_rounded.sum(1) > 12000).any():
                raise ValueError("Predicted output exceeds 12000 frames; inspect duration predictor")
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)""",
)
p.write_text(s)
p = root / "model/__init__.py"
p.write_text(
    "from .fastspeech2 import FastSpeech2\nfrom .loss import FastSpeech2Loss\n"
)
p = root / "model/optimizer.py"
p.unlink(missing_ok=True)
