#!/usr/bin/env bash
# Ollama server for Indira — explicit env vars, identical for every
# machine. Use this instead of the desktop app (whose launchctl-based config
# silently resets on reboot).
#
#   brew install ollama          # one-time
#   ./scripts/start_ollama.sh    # run in its own terminal
#
# For the installation rig this gets wrapped in a launchd plist with
# KeepAlive=true (auto-restart doubles as the 72h watchdog — see TODO Phase 5).

# Two KV-cache slots so background memory-consolidation calls don't evict
# the live conversation's prompt cache (Ollama routes each request to the
# slot with the longest matching prefix).
export OLLAMA_NUM_PARALLEL=2

# Note: model keep-alive is NOT set here — the app requests keep_alive: -1
# per-request (config llm.keep_alive), so it works with any server setup.

exec ollama serve
