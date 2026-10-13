#!/bin/bash
# Exports the settings config.py resolves; safe to source from any directory.
eval "$(python "$(dirname "${BASH_SOURCE[0]}")/config.py" --sh-vars)"
