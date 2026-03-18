#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PORT=2555
CONFIG="$DIR/litellm_config.yaml"
VENV="$DIR/.venv"

echo ""
echo "  Claude Code Alias Setup"
echo "  ========================"
echo ""
echo "  This creates a shell alias so you can run Claude Code"
echo "  backed by a model from your LiteLLM proxy."
echo ""

# --- Detect shell profile ---

SHELL_NAME="$(basename "$SHELL")"
case "$SHELL_NAME" in
    zsh)  PROFILE="$HOME/.zshrc" ;;
    bash) PROFILE="$HOME/.bashrc" ;;
    *)    PROFILE="$HOME/.${SHELL_NAME}rc" ;;
esac

echo "  Detected shell: $SHELL_NAME ($PROFILE)"
echo ""

# --- Ensure venv exists for Python helpers ---

if [ ! -d "$VENV" ] || [ ! -x "$VENV/bin/python" ]; then
    echo "  Setting up Python environment..."
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install -q -r "$DIR/requirements.txt"
fi

# --- Get all available models from provider catalogs + config ---

MODELS_JSON=$("$VENV/bin/python" -c "
import json, sys
sys.path.insert(0, '$DIR')
import config
import providers

# Configured models (already in proxy)
configured = {m['alias']: m['model'] for m in config.list_models()}

# All models from provider catalogs
catalog = {}
for p in providers.all_providers():
    if p.name == 'ollama':
        continue  # Skip ollama — not useful for Claude Code
    for alias, model_str in p.models.items():
        catalog[alias] = {'model': model_str, 'provider': p.display_name, 'configured': alias in configured}

# Merge: configured models first, then unconfigured from catalog
result = []
for alias, model in configured.items():
    prov = catalog.get(alias, {}).get('provider', '')
    result.append({'alias': alias, 'model': model, 'provider': prov, 'configured': True})
for alias, info in catalog.items():
    if alias not in configured:
        result.append({'alias': alias, 'model': info['model'], 'provider': info['provider'], 'configured': False})

print(json.dumps(result))
" 2>/dev/null)

if [ -z "$MODELS_JSON" ] || [ "$MODELS_JSON" = "[]" ]; then
    echo "  ✗ No models available."
    exit 1
fi

# Parse JSON into arrays
ALIASES=()
DISPLAY=()
CONFIGURED=()
MODEL_STRS=()

while IFS= read -r line; do
    alias=$(echo "$line" | "$VENV/bin/python" -c "import json,sys; d=json.load(sys.stdin); print(d['alias'])")
    provider=$(echo "$line" | "$VENV/bin/python" -c "import json,sys; d=json.load(sys.stdin); print(d['provider'])")
    configured=$(echo "$line" | "$VENV/bin/python" -c "import json,sys; d=json.load(sys.stdin); print(d['configured'])")
    model_str=$(echo "$line" | "$VENV/bin/python" -c "import json,sys; d=json.load(sys.stdin); print(d['model'])")

    ALIASES+=("$alias")
    MODEL_STRS+=("$model_str")
    CONFIGURED+=("$configured")

    if [ "$configured" = "True" ]; then
        DISPLAY+=("$alias ($provider) ✓ configured")
    else
        DISPLAY+=("$alias ($provider)")
    fi
done < <(echo "$MODELS_JSON" | "$VENV/bin/python" -c "import json,sys; [print(json.dumps(x)) for x in json.load(sys.stdin)]")

echo "  Available models:"
echo ""
for i in "${!DISPLAY[@]}"; do
    echo "    [$((i+1))] ${DISPLAY[$i]}"
done
echo ""

read -p "  Choose model [1]: " MODEL_CHOICE
MODEL_CHOICE="${MODEL_CHOICE:-1}"
MODEL_IDX=$((MODEL_CHOICE - 1))

if [ "$MODEL_IDX" -lt 0 ] || [ "$MODEL_IDX" -ge "${#ALIASES[@]}" ]; then
    echo "  ✗ Invalid choice."
    exit 1
fi

MODEL="${ALIASES[$MODEL_IDX]}"
MODEL_STR="${MODEL_STRS[$MODEL_IDX]}"
IS_CONFIGURED="${CONFIGURED[$MODEL_IDX]}"

echo "  Selected: $MODEL"
echo ""

# --- Auto-add to config if not configured ---

if [ "$IS_CONFIGURED" = "False" ]; then
    echo "  Model not in proxy config yet. Adding..."
    "$VENV/bin/python" -c "
import sys
sys.path.insert(0, '$DIR')
import config
ok, msg = config.add_model('$MODEL', '$MODEL_STR')
print(f'  ✓ {msg}' if ok else f'  ✗ {msg}')
" 2>/dev/null
    echo "  (Run ./litellm.sh restart to activate)"
    echo ""
fi

# --- Choose alias name ---

DEFAULT_ALIAS="claude-${MODEL}"
read -p "  Alias name [$DEFAULT_ALIAS]: " ALIAS_NAME
ALIAS_NAME="${ALIAS_NAME:-$DEFAULT_ALIAS}"

# Validate alias name (alphanumeric, hyphens, underscores, dots)
if [[ ! "$ALIAS_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "  ✗ Invalid alias name. Use letters, numbers, hyphens, underscores, dots."
    exit 1
fi

echo ""

# --- Read master key from .env ---

MASTER_KEY="sk-1234"
if [ -f "$DIR/.env" ]; then
    KEY=$(grep "^LITELLM_MASTER_KEY=" "$DIR/.env" | cut -d= -f2- | tr -d '"' | tr -d "'")
    if [ -n "$KEY" ]; then
        MASTER_KEY="$KEY"
    fi
fi

# --- Build the function ---

# Use a function instead of alias so we can properly isolate env vars.
# env -i starts clean, then we pass only what's needed + PATH/HOME/TERM.
FUNC_BLOCK=$(cat <<FUNCEOF
# Claude Code via LiteLLM Proxy (${MODEL})
${ALIAS_NAME}() {
  ANTHROPIC_BASE_URL="http://localhost:${PORT}" \\
  ANTHROPIC_MODEL="${MODEL}" \\
  ANTHROPIC_API_KEY="${MASTER_KEY}" \\
  CLAUDE_CODE_DISABLE_1M_CONTEXT=1 \\
  CLAUDE_CODE_SKIP_OAUTH=1 \\
  claude "\$@"
}
FUNCEOF
)

MARKER="# Claude Code via LiteLLM Proxy (${ALIAS_NAME})"

# --- Check for existing function ---

if grep -q "^${ALIAS_NAME}()" "$PROFILE" 2>/dev/null || grep -q "alias ${ALIAS_NAME}=" "$PROFILE" 2>/dev/null; then
    echo "  ⚠ '${ALIAS_NAME}' already exists in $PROFILE."
    read -p "  Overwrite? [y/N]: " OVERWRITE
    if [[ "${OVERWRITE,,}" != "y" ]]; then
        echo "  Cancelled."
        exit 0
    fi
    # Remove old function/alias block
    sed -i.bak "/# Claude Code via LiteLLM Proxy.*${ALIAS_NAME}/d; /^${ALIAS_NAME}()/,/^}/d; /alias ${ALIAS_NAME}=/d" "$PROFILE"
    echo "  Removed old entry."
fi

# --- Write to profile ---

echo "" >> "$PROFILE"
echo "$FUNC_BLOCK" >> "$PROFILE"

echo "  ✓ Added to $PROFILE:"
echo ""
echo "$FUNC_BLOCK" | sed 's/^/    /'
echo ""

# --- Source it ---

echo "  To activate now, run:"
echo ""
echo "    source $PROFILE"
echo ""
echo "  Then use it:"
echo ""
echo "    $ALIAS_NAME"
echo ""
echo "  Make sure the proxy is running first:"
echo ""
echo "    cd $DIR && ./litellm.sh up && ./litellm.sh login openai"
echo ""
