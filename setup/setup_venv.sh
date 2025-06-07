#!/bin/bash
if [ ! -d ".venv" ]; then
  # Suche passende Python-Version <= 3.12
  for py in python3.12 python3.11 python3.10 python3.9 python3; do
    if command -v $py >/dev/null 2>&1; then
      ver=$($py -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
      # Vergleich der Version (<= 3.12)
      ver_major=$(echo $ver | cut -d. -f1)
      ver_minor=$(echo $ver | cut -d. -f2)
      if [ $ver_major -eq 3 ] && [ $ver_minor -le 12 ]; then
        echo "Using Python $ver for venv creation"
        $py -m venv .venv
        .venv/bin/pip install --upgrade pip
        .venv/bin/pip install -r requirements.txt
        exit 0
      fi
    fi
  done
  echo "Error: No suitable Python version (<= 3.12) found!" >&2
  exit 1
else
  echo "venv already exists"
fi
