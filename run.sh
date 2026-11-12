#!/usr/bin/env bash
cd "$(dirname "$0")"
[[ -f settings.sh ]] && source settings.sh
MUSICAPP_OPEN_BROWSER=1 exec .venv/bin/python -m app
