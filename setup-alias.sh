#!/bin/bash
set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PORT=2555
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

# --- Step 1: Choose provider ---

PROVIDER_JSON=$(DIR="$DIR" "$VENV/bin/python" -c '
import json, os, sys
sys.path.insert(0, os.environ["DIR"])
import providers

result = []
for p in providers.all_providers():
    status, msg = p.validate()
    result.append({
        "name": p.name,
        "display": p.display_name,
        "auth_ok": status.value == "ok",
        "auth_msg": msg
    })
print(json.dumps(result))
')

# Parse providers
PROV_NAMES=()
PROV_DISPLAYS=()
PROV_AUTH=()

while IFS=$'\t' read -r name display auth_ok auth_msg; do
    PROV_NAMES+=("$name")
    if [ "$auth_ok" = "True" ]; then
        PROV_DISPLAYS+=("$display  ✓ $auth_msg")
    else
        PROV_DISPLAYS+=("$display  ✗ $auth_msg")
    fi
    PROV_AUTH+=("$auth_ok")
done < <("$VENV/bin/python" -c '
import json, sys
data = json.loads(sys.stdin.read())
for p in data:
    print(f"{p[\"name\"]}\t{p[\"display\"]}\t{p[\"auth_ok\"]}\t{p[\"auth_msg\"]}")
' <<< "$PROVIDER_JSON")

echo "  Select a provider:"
echo ""
for i in "${!PROV_DISPLAYS[@]}"; do
    echo "    [$((i+1))] ${PROV_DISPLAYS[$i]}"
done
echo ""

read -p "  Choose: " PROV_CHOICE
if [ -z "$PROV_CHOICE" ]; then
    echo "  ✗ No choice made."
    exit 1
fi
PROV_IDX=$((PROV_CHOICE - 1))

if [ "$PROV_IDX" -lt 0 ] || [ "$PROV_IDX" -ge "${#PROV_NAMES[@]}" ]; then
    echo "  ✗ Invalid choice."
    exit 1
fi

PROVIDER="${PROV_NAMES[$PROV_IDX]}"
PROVIDER_AUTH="${PROV_AUTH[$PROV_IDX]}"

# --- Check auth ---

if [ "$PROVIDER_AUTH" != "True" ]; then
    echo ""
    echo "  ✗ $PROVIDER is not authenticated."
    echo "    Run: ./litellm.sh login $PROVIDER"
    exit 1
fi

# --- Step 2: Get models for selected provider ---

MODELS_JSON=$(DIR="$DIR" PROVIDER="$PROVIDER" "$VENV/bin/python" -c '
import json, os, sys
sys.path.insert(0, os.environ["DIR"])
import config
import providers

provider = providers.get_provider(os.environ["PROVIDER"])
configured = {m["alias"]: m["model"] for m in config.list_models()}

# Get models from provider
if provider.name == "ollama":
    catalog = provider.discover_models()
    if catalog is None:
        print("Error: Cannot reach Ollama. Check that it is running.", file=sys.stderr)
        sys.exit(1)
else:
    catalog = provider.models

result = []
# Configured models for this provider first
for alias, model in configured.items():
    if model.startswith("ollama/") and provider.name == "ollama":
        result.append({"alias": alias, "model": model, "configured": True})
    elif not model.startswith("ollama/") and alias in catalog:
        result.append({"alias": alias, "model": model, "configured": True})

# Unconfigured from catalog
for alias, model_str in catalog.items():
    if alias not in configured:
        result.append({"alias": alias, "model": model_str, "configured": False})

print(json.dumps(result))
')

# Parse models
ALIASES=()
DISPLAY=()
CONFIGURED=()
MODEL_STRS=()

while IFS=$'\t' read -r alias model configured; do
    ALIASES+=("$alias")
    MODEL_STRS+=("$model")
    CONFIGURED+=("$configured")
    if [ "$configured" = "True" ]; then
        DISPLAY+=("$alias  ✓ configured")
    else
        DISPLAY+=("$alias")
    fi
done < <("$VENV/bin/python" -c '
import json, sys
data = json.loads(sys.stdin.read())
for m in data:
    print(f"{m[\"alias\"]}\t{m[\"model\"]}\t{m[\"configured\"]}")
' <<< "$MODELS_JSON")

if [ "${#ALIASES[@]}" -eq 0 ]; then
    if [ "$PROVIDER" = "ollama" ]; then
        echo ""
        echo "  No models found in Ollama."
        echo "  Pull one first: ollama pull <model>"
        echo "  Or login for cloud models: ollama login"
    else
        echo ""
        echo "  ✗ No models available for this provider."
    fi
    exit 1
fi

echo ""
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
    DIR="$DIR" PROVIDER="$PROVIDER" MODEL="$MODEL" MODEL_STR="$MODEL_STR" \
    "$VENV/bin/python" <<'PYEOF'
import os, sys
sys.path.insert(0, os.environ["DIR"])
import config
import providers

provider = providers.get_provider(os.environ["PROVIDER"])
model = os.environ["MODEL"]
model_str = os.environ["MODEL_STR"]
extra = {}
if provider.name == "ollama":
    extra = provider.get_extra_params()
ok, msg = config.add_model(model, model_str, extra)
print(f"  \u2713 {msg}" if ok else f"  \u2717 {msg}")
PYEOF
    echo "  (Run ./litellm.sh restart to activate)"
    echo ""
fi

# --- Choose alias name ---

DEFAULT_ALIAS="claude-${MODEL}"
# Sanitize default: replace colons and slashes with hyphens
DEFAULT_ALIAS=$(echo "$DEFAULT_ALIAS" | tr ':/' '-')
read -p "  Alias name [$DEFAULT_ALIAS]: " ALIAS_NAME
ALIAS_NAME="${ALIAS_NAME:-$DEFAULT_ALIAS}"

# Validate alias name (alphanumeric, hyphens, underscores, dots)
if [[ ! "$ALIAS_NAME" =~ ^[a-zA-Z0-9_.-]+$ ]]; then
    echo "  ✗ Invalid alias name. Use letters, numbers, hyphens, underscores, dots."
    exit 1
fi

echo ""

# --- Build the function ---

# Use a function instead of alias so we can properly isolate env vars.
# The master key is read from .env at invocation time rather than embedded
# in the shell profile, so it stays in the chmod-600 .env file.
ENV_PATH="$DIR/.env"

FUNC_BLOCK=$(cat <<FUNCEOF
# Claude Code via LiteLLM Proxy (${MODEL})
# Note: master key is read from .env at runtime (not embedded in profile)
${ALIAS_NAME}() {
  local _mk
  _mk=\$("${VENV}/bin/python" -c "
import sys; sys.path.insert(0, '${DIR}')
from config import load_env_file
v = load_env_file('${ENV_PATH}').get('LITELLM_MASTER_KEY', '')
print(v)
" 2>/dev/null)
  if [ \$? -ne 0 ]; then
    echo "Error: Failed to read ${ENV_PATH} — check Python environment at ${VENV}" >&2
    return 1
  fi
  if [ -z "\$_mk" ]; then
    echo "Error: LITELLM_MASTER_KEY not found in ${ENV_PATH} — set it with ./litellm.sh or add it manually." >&2
    return 1
  fi
  ANTHROPIC_BASE_URL="http://localhost:${PORT}" \\
  ANTHROPIC_MODEL="${MODEL}" \\
  ANTHROPIC_AUTH_TOKEN="\$_mk" \\
  CLAUDE_CODE_DISABLE_1M_CONTEXT=1 \\
  claude "\$@"
}
FUNCEOF
)

# --- Check for existing function ---

if grep -q "^${ALIAS_NAME}()" "$PROFILE" 2>/dev/null || grep -q "alias ${ALIAS_NAME}=" "$PROFILE" 2>/dev/null; then
    echo "  ⚠ '${ALIAS_NAME}' already exists in $PROFILE."
    read -p "  Overwrite? [y/N]: " OVERWRITE
    if [[ "${OVERWRITE,,}" != "y" ]]; then
        echo "  Cancelled."
        exit 0
    fi
    # Remove old function/alias block using grep -v (portable, no GNU sed needed)
    # Removes: comment lines, function definition through closing brace, legacy aliases
    TMPFILE=$(mktemp)
    awk -v alias="${ALIAS_NAME}" '
        /^# Claude Code via LiteLLM Proxy/ { skip_comment=1; next }
        /^# Note: master key is read/ && skip_comment { next }
        skip_comment && $0 ~ "^"alias"\\(\\)" { skip_comment=0; in_func=1; next }
        $0 ~ "^"alias"\\(\\)" { in_func=1; next }
        in_func && /^}/ { in_func=0; next }
        in_func { next }
        $0 ~ "^alias "alias"=" { next }
        { skip_comment=0; print }
    ' "$PROFILE" > "$TMPFILE" && mv "$TMPFILE" "$PROFILE"
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
echo "    cd $DIR && ./litellm.sh up"
echo ""
