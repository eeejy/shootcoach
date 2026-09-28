"""Write the printable target (PNG + PDF, true size at 300 dpi). Print at 100% scale."""
import argparse

from shootcoach.config import load_target_spec
from shootcoach.target.template import save_printable

ap = argparse.ArgumentParser()
ap.add_argument("--config", default=None)
ap.add_argument("--out", default="samples")
args = ap.parse_args()
png, pdf = save_printable(load_target_spec(args.config), args.out)
print(png, pdf)
