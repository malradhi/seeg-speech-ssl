#!/bin/sh
# Regenerate every table of the supplementary material from the released analysis outputs.
# Usage (from paper/supplement): sh build.sh [results directory, default ../../results]
# Writes supp_tables.tex (all tables) and tables/<name>.tex (one file per table).
set -e
R="${1:-../../results}"
python3 make_supp_tables.py "$R/stats" "$R" supp_tables.tex
python3 - <<'PY'
import re, os
s = open("supp_tables.tex").read()
parts = re.split(r"(?m)^% ---- (.*)$", s)
os.makedirs("tables", exist_ok=True)
for i in range(1, len(parts), 2):
    name = re.sub(r"[^a-z0-9]+", "_", parts[i].lower()).strip("_")
    open(f"tables/{name}.tex", "w").write(parts[i + 1])
PY
echo "tables written to $(pwd)/tables"
