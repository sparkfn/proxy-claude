#!/usr/bin/env python3
import json
import logging
import os
import sys
import time

from container import PROXY_PORT as PORT
from providers.base import Status

log = logging.getLogger("litellm-cli")
MODEL_ALIAS_STATE_FILE = os.environ.get("PROXY_MODEL_ALIAS_STATE", "/tmp/litellm-proxy-model-alias.json")

_STATUS_ICON = {
    Status.OK: "\u2713",
    Status.UNVERIFIED: "?",
    Status.NOT_CONFIGURED: "\u2717",
    Status.UNREACHABLE: "\u2717",
    Status.FAILED: "\u2717",
    Status.INVALID: "\u2717",
    Status.NOT_FOUND: "\u2717",
}

def _icon(status):
    """Return a single-char icon for a Status value."""
    return _STATUS_ICON.get(status, "\u2717")


def _setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="  [%(levelname)s] %(name)s: %(message)s",
    )


def _eprint(*args, **kwargs):
    """Print to stderr (for interactive prompts that shouldn't mix with stdout data)."""
    print(*args, file=sys.stderr, **kwargs)


def _thinking_level_label(level, contract):
    labels = (contract or {}).get("level_labels") or {}
    return labels.get(level, level.replace("_", " ").title())


def _thinking_level_description(level):
    descriptions = {
        "none": "No additional reasoning",
        "minimal": "Fastest responses with minimal reasoning",
        "low": "Fast responses with lighter reasoning",
        "medium": "Balances speed and reasoning depth for everyday tasks",
        "high": "Greater reasoning depth for complex problems",
        "xhigh": "Extra high reasoning depth for complex problems",
    }
    return descriptions.get(level, "")


def _persist_selected_model_state(alias):
    """Persist the active Claude-tier mapping for the proxy control plane."""
    state = {
        "selected_model": alias,
        "anthropic_defaults": {
            "haiku": alias,
            "sonnet": alias,
            "opus": alias,
        },
        "updated_at": int(time.time()),
    }
    tmp_path = f"{MODEL_ALIAS_STATE_FILE}.tmp.{os.getpid()}"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(state, f)
    os.replace(tmp_path, MODEL_ALIAS_STATE_FILE)


def show_help():
    name = os.environ.get("LITELLM_CLI_NAME", os.path.basename(sys.argv[0]) or "./proclaude.sh")
    print("LiteLLM Gateway CLI")
    print(f"Usage: {name} <command> [options]")
    print()
    print("Status:")
    print("  status            Backend and model status")
    print()
    print("Models:")
    print("  model add         Add models (interactive)")
    print("  model rm          Remove a configured model")
    print("  model list        List configured models")
    print()
    print("Providers:")
    print("  provider list     Show available providers")
    print("  provider status   Show auth status per provider")
    print("  provider login    Authenticate with a provider")
    print("  provider logout   Remove provider credentials")
    print()
    print("Launch:")
    print("  launch claude     Launch Claude Code through the proxy")
    print()
    print("Options:")
    print("  --verbose, -v     Enable debug logging")


SUBCOMMAND_REGISTRY = {
    "model": {
        "add":  (None, "add [--provider X]", "Add models (interactive)"),
        "rm":   (None, "rm [--provider X]", "Remove a configured model"),
        "list": (None, "list [--provider X]", "List configured models"),
    },
    "provider": {
        "list":   (None, "list", "Show available providers"),
        "status": (None, "status", "Show auth status per provider"),
        "login":  (None, "login [name]", "Authenticate with a provider"),
        "logout": (None, "logout [name]", "Remove provider credentials"),
        "openai-browser-trigger": (None, "openai-browser-trigger", "Trigger OpenAI device-code flow"),
    },
    "launch": {
        "claude": (None, "claude [--provider X] [--model Y] [--thinking <level>] [--telegram|--no-telegram] [--emit-env <path>] [-- args...]", "Launch Claude Code through the proxy"),
    },
}


def _show_group_help(group):
    """Print subcommand help for a command group."""
    name = os.environ.get("LITELLM_CLI_NAME", os.path.basename(sys.argv[0]) or "./proclaude.sh")
    entries = SUBCOMMAND_REGISTRY.get(group)
    if not entries:
        return
    print(f"Usage: {name} {group} <subcommand>\n")
    for _sub, (_handler, usage_str, desc) in entries.items():
        print(f"  {name} {group} {usage_str:<40} {desc}")


def cmd_openai_browser_trigger(provider_flag=None, model_flag=None, extra_args=None, **_kwargs):
    """Trigger OpenAI browser OAuth device-code flow by sending a dummy request."""
    import config
    import requests
    master_key = config.get_env("LITELLM_MASTER_KEY") or ""
    chatgpt_models = [m["alias"] for m in config.list_models() if m.get("model", "").startswith("chatgpt/")]
    if not chatgpt_models:
        print("  ✗ No chatgpt/ models configured.")
        sys.exit(1)
    try:
        requests.post(
            f"http://localhost:{PORT}/v1/chat/completions",
            json={"model": chatgpt_models[0], "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1},
            headers={"Authorization": f"Bearer {master_key}", "Content-Type": "application/json"},
            timeout=10,
        )
    except Exception:
        pass  # Expected to fail — just triggers the device code in LiteLLM logs


def cmd_start_status():
    """Quick post-start status: model auth readiness (no network calls)."""
    import config
    import providers

    env_data = config.load_env_file(config.ENV_PATH)
    auth_dir = os.path.join(config.DIR, "auth")
    print("")
    print("  Models:")
    for provider in providers.all_providers():
        if not provider.models:
            continue
        ready, reason = provider.check_ready(env_data, auth_dir=auth_dir)
        alias = next(iter(provider.models))
        icon = "✓" if ready else "✗"
        label = "ready" if ready else reason
        print(f"    {icon} {alias:<16} {provider.display_name:<10} {label}")


def cmd_status():
    import container
    import config
    import providers

    cs, _ = container.status()
    state = "reachable" if cs == Status.OK else "unreachable"
    print(f"LiteLLM:    {state}")
    print(f"Port:       localhost:{PORT}")
    print()

    models = config.list_models()
    if not models:
        print("Models:     (none configured)")
        return

    # Cache validation per provider to avoid redundant network calls
    auth_cache = {}
    print("Models:")
    for m in models:
        provider = providers.get_provider(m["provider"])
        if provider and cs == Status.OK:
            if m["provider"] not in auth_cache:
                auth_cache[m["provider"]] = provider.validate()
            auth_status, _ = auth_cache[m["provider"]]
            icon = _icon(auth_status)
            if auth_status == Status.OK:
                label = "authenticated" if m["provider"] != "ollama" else "reachable"
            elif auth_status == Status.UNVERIFIED:
                label = "unverified"
            elif auth_status == Status.NOT_CONFIGURED:
                label = "not configured"
            elif auth_status == Status.UNREACHABLE:
                label = "unreachable"
            else:
                label = "invalid"
        else:
            icon = "-"
            label = "unknown" if cs != Status.OK else "unknown provider"
        print(f"  {m['alias']:<12} {m['provider']:<10} {icon} {label}")


def _prompt_credentials(provider, auth_type):
    """Prompt user for credentials based on provider.login_prompts. Returns dict."""
    prompts = getattr(provider, "login_prompts", {}).get(auth_type)
    if not prompts:
        return None
    print(f"\n  {prompts['instructions']}\n")
    credentials = {}
    for env_var, prompt_text in prompts["fields"]:
        credentials[env_var] = input(f"  {prompt_text}").strip()
    return credentials


def _ollama_interactive_login(provider):
    """Drive the interactive Ollama login flow (cloud login, model discovery, pull)."""
    # Offer cloud login
    choice = input("\n  Login to ollama.com for cloud models? [y/N]: ").strip()
    if choice.lower() == "y":
        print()
        s, msg = provider.ollama_cloud_login()
        print(f"  {_icon(s)} {msg}")

    # Show available models
    models = provider.discover_models()
    if models is None:
        print("\n  Warning: Could not discover models (check Ollama status).")
    elif models:
        print(f"\n  Available models ({len(models)}):\n")
        for alias in models:
            print(f"    - {alias}")
    else:
        print("\n  No models found. Pull one: ollama pull <model>")

    # Offer to pull
    pull = input("\n  Pull a model? Enter name (or Enter to skip): ").strip()
    if pull:
        print()
        ps, pull_msg = provider.pull_model(pull)
        print(f"  {_icon(ps)} {pull_msg}")


def _choose_auth_type(provider):
    """Prompt user to choose auth type. Returns auth_type string or None."""
    if not provider.auth_types:
        return None
    if len(provider.auth_types) == 1:
        return provider.auth_types[0]
    print(f"\n  Auth method for {provider.display_name}:\n")
    for i, at in enumerate(provider.auth_types, 1):
        label = at.replace("_", " ").title()
        print(f"    [{i}] {label}")
    print()
    choice = input("  Choose [1]: ").strip() or "1"
    try:
        return provider.auth_types[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid choice.")
        sys.exit(1)


def _print_login_result(login_status, msg):
    """Print login result with status icon. Returns the status for caller control flow."""
    print(f"\n  {_icon(login_status)} {msg}")
    return login_status


def _restart_and_report(context_msg, provider=None, added=None):
    """Report change and suggest restart."""
    log.debug("Config changed: %s", context_msg)
    if added:
        print(f"  Added: {', '.join(added)}")
    if provider:
        status, msg = provider.validate()
        if status == Status.OK:
            print(f"  {_icon(status)} {msg}")
    print(f"  Restart to apply: ./proclaude.sh restart")


def _ollama_manual_input(provider, catalog):
    """Prompt for a model name manually. Offer to pull if not found.
    Returns (selected_list, updated_catalog)."""
    model_name = input("\n  Model name: ").strip()
    if not model_name:
        print("  Cancelled.")
        sys.exit(1)

    if model_name not in catalog:
        pull = input(f"  '{model_name}' not found in Ollama. Pull it? [Y/n]: ").strip()
        if pull.lower() != "n":
            print()
            ps, msg = provider.pull_model(model_name)
            if ps != Status.OK:
                print(f"  \u2717 {msg}")
                sys.exit(1)
            print(f"  \u2713 {msg}")

    catalog[model_name] = f"ollama/{model_name}"
    return [model_name], catalog


# --- Provider commands ---

def cmd_provider_list(provider_flag=None, model_flag=None, extra_args=None):
    """Show available providers."""
    import providers
    print("\n  Available providers:\n")
    for p in providers.all_providers():
        print(f"    {p.name:<12} {p.display_name}")


def cmd_provider_status(provider_flag=None, model_flag=None, extra_args=None):
    """Show auth status per provider."""
    import providers
    print("\n  Provider auth status:\n")
    for p in providers.all_providers():
        status, msg = p.validate()
        if status == Status.OK:
            print(f"    {p.display_name:<20} \u2713 {msg}")
        elif status == Status.UNVERIFIED:
            print(f"    {p.display_name:<20} ? {msg}")
        else:
            print(f"    {p.display_name:<20} \u2717 {msg}")


def cmd_provider_login(provider_flag=None, model_flag=None, extra_args=None):
    """Authenticate with a provider."""
    import providers

    # Provider name from flag, positional arg, or interactive
    provider_name = provider_flag or (extra_args[0] if extra_args else None)

    if provider_name is None:
        all_provs = providers.all_providers()
        print("\n  Select a provider:\n")
        for i, p in enumerate(all_provs, 1):
            print(f"    [{i}] {p.display_name}")
        print()
        choice = input("  Choose: ").strip()
        try:
            provider = all_provs[int(choice) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice.")
            sys.exit(1)
    else:
        provider = providers.get_provider(provider_name)
        if not provider:
            print(f"  Unknown provider: {provider_name}")
            print(f"  Available: {', '.join(p.name for p in providers.all_providers())}")
            sys.exit(1)

    # Providers with no auth_types (e.g. Ollama)
    if len(provider.auth_types) == 0:
        ls, msg = provider.login()
        print(f"  {_icon(ls)} {msg}")
        if provider.name == "ollama" and ls == Status.OK:
            _ollama_interactive_login(provider)
        return

    # Check if already authenticated (only short-circuit on OK, not UNVERIFIED --
    # UNVERIFIED users may want to re-authenticate with better credentials)
    status, msg = provider.validate()
    if status == Status.OK:
        print(f"  \u2713 Already authenticated with {provider.display_name}. {msg}")
        return

    auth_type = _choose_auth_type(provider)
    credentials = _prompt_credentials(provider, auth_type)
    try:
        result = _print_login_result(*provider.login(auth_type, credentials=credentials))
    except RuntimeError as e:
        print(f"  ✗ {e}")
        sys.exit(1)
    if result not in (Status.OK, Status.UNVERIFIED):
        sys.exit(1)

    # Restart container so it picks up the new .env credentials
    import container
    cs, _ = container.status()
    if cs == Status.OK:
        _restart_and_report("provider login", provider=provider)


def cmd_provider_logout(provider_flag=None, model_flag=None, extra_args=None):
    """Remove provider credentials."""
    import providers
    import config

    provider_name = provider_flag or (extra_args[0] if extra_args else None)

    if provider_name is None:
        all_provs = providers.all_providers()
        print("\n  Select a provider:\n")
        for i, p in enumerate(all_provs, 1):
            print(f"    [{i}] {p.display_name}")
        print()
        choice = input("  Choose: ").strip()
        try:
            provider = all_provs[int(choice) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice.")
            sys.exit(1)
    else:
        provider = providers.get_provider(provider_name)
        if not provider:
            print(f"  Unknown provider: {provider_name}")
            sys.exit(1)

    # Detect active auth type and handle accordingly
    if provider.name == "ollama":
        print("  Ollama has no stored credentials.")
        return

    auth_type = provider.detect_auth_type()
    env_vars_for_auth = provider.env_vars.get(auth_type, []) if auth_type else []

    if not env_vars_for_auth:
        # Browser OAuth or no env vars -- can't clear programmatically
        print(f"  {provider.display_name} browser OAuth credentials are managed by the container.")
        print(f"  Restart with './proclaude.sh restart' to reset.")
        return

    confirm = input(
        f"  Remove {', '.join(env_vars_for_auth)} from .env? [y/N]: "
    ).strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    try:
        for var in env_vars_for_auth:
            config.remove_env(var)
    except RuntimeError as e:
        print(f"  ✗ {e}")
        sys.exit(1)
    print(f"  \u2713 Removed credentials for {provider.display_name}.")


# --- Model commands ---

def cmd_model_list(provider_flag=None, model_flag=None, extra_args=None):
    """List configured models, optionally filtered by provider."""
    import config
    models = config.list_models()
    if provider_flag:
        models = [m for m in models if m["provider"] == provider_flag]
    if not models:
        print("  No models configured." if not provider_flag
              else f"  No models configured for provider '{provider_flag}'.")
        return
    print("  Configured models:")
    for m in models:
        print(f"    {m['alias']:<12} {m['provider']:<10} ({m['model']})")


def cmd_model_add(provider_flag=None, model_flag=None, extra_args=None):
    """Add models for a provider."""
    import config
    import providers

    # Step 1: Pick provider
    if provider_flag:
        provider = providers.get_provider(provider_flag)
        if not provider:
            print(f"  Unknown provider: {provider_flag}")
            print(f"  Available: {', '.join(p.name for p in providers.all_providers())}")
            sys.exit(1)
    else:
        all_provs = providers.all_providers()
        print("\n  Select a provider:\n")
        for i, p in enumerate(all_provs, 1):
            print(f"    [{i}] {p.display_name}")
        print()
        choice = input("  Choose: ").strip()
        try:
            provider = all_provs[int(choice) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice.")
            sys.exit(1)

    # Step 2: Check provider is ready
    if provider.name == "ollama":
        status, msg = provider.validate()
        if status != Status.OK:
            print(f"\n  \u2717 Ollama is not running. Start it first.")
            sys.exit(1)
        catalog = provider.discover_models()
        if catalog is None:
            print(f"\n  \u2717 Cannot reach Ollama.")
            sys.exit(1)
    else:
        status, msg = provider.validate()
        if status not in (Status.OK, Status.UNVERIFIED):
            print(f"\n  \u2717 {provider.display_name} is not authenticated.")
            print(f"    Run: ./proclaude.sh provider login {provider.name}")
            sys.exit(1)

        auth_type = provider.detect_auth_type()
        if hasattr(provider, "get_models_for_auth") and auth_type:
            catalog = provider.get_models_for_auth(auth_type)
        else:
            catalog = provider.models

    if not catalog:
        print("\n  No models available for this provider.")
        sys.exit(1)

    # Step 3: Show catalog with checkmarks for already-configured
    existing_aliases = {m["alias"] for m in config.list_models()}
    aliases = list(catalog.keys())

    if provider.name == "ollama":
        print(f"\n  Available Ollama models:\n")
        for i, alias in enumerate(aliases, 1):
            mark = " \u2713" if alias in existing_aliases else ""
            print(f"    [{i}] {alias}{mark}")
        print(f"    [m] Enter model name manually")
        print(f"    [a] All")
    else:
        print(f"\n  Available models for {provider.display_name}:\n")
        for i, alias in enumerate(aliases, 1):
            mark = " \u2713" if alias in existing_aliases else ""
            print(f"    [{i}] {alias}{mark}")
        print(f"    [a] All")

    print()
    model_choice = input("  Choose (comma-separated, e.g. 1,3): ").strip()

    # Step 4: Parse selection
    if provider.name == "ollama" and model_choice.lower() == "m":
        selected, catalog = _ollama_manual_input(provider, catalog)
    elif model_choice.lower() == "a":
        selected = aliases
    else:
        selected = []
        for part in model_choice.split(","):
            try:
                idx = int(part.strip()) - 1
                selected.append(aliases[idx])
            except (ValueError, IndexError):
                print(f"  Skipping invalid choice: {part.strip()}")

    if not selected:
        print("  No models selected.")
        sys.exit(1)

    # Step 5: Add each model
    added = []
    for alias in selected:
        entry = catalog[alias]
        model_str = entry["model"] if isinstance(entry, dict) else entry
        final_alias = alias
        if alias in existing_aliases:
            print(f"\n  Alias '{alias}' already exists.")
            final_alias = input(f"  Enter a different alias (or Enter to skip): ").strip()
            if not final_alias or final_alias in existing_aliases:
                print(f"  Skipping {alias}.")
                continue

        extra = {}
        if hasattr(provider, "get_extra_params"):
            extra = provider.get_extra_params()

        s, msg = config.add_model(final_alias, model_str, extra)
        if s == Status.OK:
            added.append(final_alias)
            existing_aliases.add(final_alias)
            print(f"  \u2713 {msg}")
        else:
            print(f"  \u2717 {msg}")

    if not added:
        print("\n  No models added.")
        return

    # Step 6: Restart if container is running
    import container
    cs, _ = container.status()
    if cs == Status.OK:
        _restart_and_report("adding models", provider=provider, added=added)
    else:
        print(f"\n  Added: {', '.join(added)}. Start the proxy with './proclaude.sh start'.")


def cmd_model_rm(provider_flag=None, model_flag=None, extra_args=None):
    """Remove configured models."""
    import config
    import providers

    models = config.list_models()
    if provider_flag:
        models = [m for m in models if m["provider"] == provider_flag]
    if not models:
        print("  No models configured." if not provider_flag
              else f"  No models configured for provider '{provider_flag}'.")
        return

    print(f"\n  Configured models:\n")
    for i, m in enumerate(models, 1):
        print(f"    [{i}] {m['alias']} ({m['provider']})")
    print()
    choice = input("  Remove which model(s)? (comma-separated, e.g. 1,3): ").strip()

    selected = []
    for part in choice.split(","):
        try:
            idx = int(part.strip()) - 1
            selected.append(models[idx])
        except (ValueError, IndexError):
            print(f"  Skipping invalid choice: {part.strip()}")

    if not selected:
        print("  No models selected.")
        return

    names = ", ".join(f"'{m['alias']}'" for m in selected)
    confirm = input(
        f"  Remove {names}? [y/N]: "
    ).strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    removed_providers = set()
    for model in selected:
        provider_name = model["provider"]
        s, msg = config.remove_model(model["alias"])
        if s == Status.OK:
            print(f"  \u2713 {msg}")
            if provider_name:
                removed_providers.add(provider_name)
        else:
            print(f"  \u2717 {msg}")

    # Offer to clean up env vars if last model for a provider was removed
    for pname in removed_providers:
        if not config.provider_has_models(pname):
            provider = providers.get_provider(pname)
            if provider:
                all_env_vars = []
                for env_list in provider.env_vars.values():
                    all_env_vars.extend(env_list)
                if all_env_vars:
                    cleanup = input(
                        f"  No models left for {provider.display_name}. "
                        f"Remove {', '.join(all_env_vars)} from .env? [y/N]: "
                    ).strip().lower()
                    if cleanup == "y":
                        try:
                            for var in all_env_vars:
                                config.remove_env(var)
                        except RuntimeError as e:
                            print(f"  ✗ {e}")
                            sys.exit(1)
                        print(f"  \u2713 Cleaned up env vars.")

    # Restart only if container is running
    import container
    cs, _ = container.status()
    if cs == Status.OK:
        _restart_and_report("removing models")
    else:
        print("\n  Models removed. Proxy is not running.")


# --- Launch commands ---

# Credentials collected during inline setup — written to emit-env for host-side .env update
_pending_credentials = {}
_needs_browser_oauth = False


def _launch_inline_setup(entry, out):
    """Prompt for credentials inline during launch. Returns True if successful.

    Does NOT write to .env (bind-mount prevents it). Instead stores
    credentials in _pending_credentials for the emit-env output.
    The shell script writes them to .env on the host side.
    """
    provider = entry["provider_obj"]

    # Browser OAuth (OpenAI) — must check BEFORE api_key prompts
    if "browser_oauth" in getattr(provider, "auth_types", []):
        global _needs_browser_oauth
        out("  OpenAI requires browser login. Will set up after prompts...")
        entry["ready"] = True
        entry["_needs_restart"] = True
        _needs_browser_oauth = True
        return True

    # API key auth — prompt for credentials
    for auth_type, prompts in getattr(provider, "login_prompts", {}).items():
        if not prompts or not prompts.get("fields"):
            continue
        instructions = prompts.get("instructions", "")
        if instructions:
            out(f"\n  {instructions}\n")
        for env_var, prompt_text in prompts["fields"]:
            out(f"  {prompt_text}", end="")
            sys.stderr.flush()
            value = input().strip()
            if not value:
                out("  Skipped.")
                return False
            _pending_credentials[env_var] = value
            out(f"  ✓ {env_var} collected")
        entry["ready"] = True
        entry["_needs_restart"] = True
        return True

    out(f"  {entry['ready_reason']}")
    return False


def cmd_launch_claude(provider_flag=None, model_flag=None, extra_args=None, thinking=None, **_kwargs):
    """Launch Claude Code through the LiteLLM proxy."""
    import shutil
    import config
    import providers

    extra_args = extra_args or []
    emit_env = _kwargs.get("emit_env")
    # Use _eprint for interactive output so it doesn't pollute stdout in --emit-env mode
    out = _eprint if emit_env else print
    _credentials_changed = False  # Set True if inline setup wrote new keys to .env

    # Step 1: Check claude binary (skip in --emit-env mode)
    claude_bin = None
    if not emit_env:
        # Detect if running inside container (claude binary won't be here)
        if os.environ.get("PROXY_LITELLM_HOST"):
            print("  \u2717 'launch claude' must be run via ./proclaude.sh on the host")
            print("    Run: ./proclaude.sh launch claude")
            sys.exit(1)
        claude_bin = shutil.which("claude")
        if not claude_bin:
            print("  \u2717 Claude Code CLI not found. Install it first:")
            print("    npm install -g @anthropic-ai/claude-code")
            sys.exit(1)

    # Step 2: Build catalog from provider registry with readiness checks.
    # Uses check_ready() for fast local checks instead of network-calling validate().
    env_data = config.load_env_file(config.ENV_PATH)
    auth_dir = os.path.join(config.DIR, "auth")
    all_entries = []
    for provider in providers.all_providers():
        ready, reason = provider.check_ready(env_data, auth_dir=auth_dir)
        for alias, entry in provider.models.items():
            litellm_model = entry["model"] if isinstance(entry, dict) else entry
            all_entries.append({
                "alias": alias,
                "provider": provider.name,
                "display_name": provider.display_name,
                "model": litellm_model,
                "ready": ready,
                "ready_reason": reason,
                "provider_obj": provider,
            })

    if not all_entries:
        out("  No models available.")
        sys.exit(1)

    # Filter by provider/model flags
    if provider_flag:
        all_entries = [e for e in all_entries if e["provider"] == provider_flag]
        if not all_entries:
            out(f"  No models for provider '{provider_flag}'.")
            sys.exit(1)

    if model_flag:
        match = [e for e in all_entries if e["alias"] == model_flag]
        if not match:
            out(f"  Model '{model_flag}' not found.")
            out(f"  Available: {', '.join(e['alias'] for e in all_entries)}")
            sys.exit(1)
        entry = match[0]
        if not entry["ready"]:
            out(f"  {entry['alias']} is not ready: {entry['ready_reason']}")
            if not _launch_inline_setup(entry, out):
                sys.exit(1)
            _credentials_changed = True
        model = {"alias": entry["alias"], "model": entry["model"], "provider": entry["provider"],
                 "litellm_params": {"model": entry["model"]}}
    else:
        ready = [e for e in all_entries if e["ready"]]
        not_ready = [e for e in all_entries if not e["ready"]]

        if not ready and not not_ready:
            out("  No models available.")
            sys.exit(1)

        if not ready:
            # No models ready — offer inline setup
            out("\n  No models are ready yet. Let's set one up:\n")
            for i, e in enumerate(all_entries, 1):
                out(f"    [{i}] {e['alias']:<16} {e['display_name']:<10}  {e['ready_reason']}")
            out("")
            out("  Choose: ", end="")
            sys.stderr.flush()
            choice = input().strip()
            try:
                entry = all_entries[int(choice) - 1]
            except (ValueError, IndexError):
                out("  Invalid choice.")
                sys.exit(1)
            if not _launch_inline_setup(entry, out):
                sys.exit(1)
            _credentials_changed = True
            model = {"alias": entry["alias"], "model": entry["model"], "provider": entry["provider"],
                     "litellm_params": {"model": entry["model"]}}
        elif len(ready) == 1 and not not_ready:
            entry = ready[0]
            out(f"  Using {entry['alias']} ({entry['display_name']})")
            model = {"alias": entry["alias"], "model": entry["model"], "provider": entry["provider"],
                     "litellm_params": {"model": entry["model"]}}
        else:
            # Show all models in one numbered list — ready and not-ready
            combined = ready + not_ready
            out("")
            for i, e in enumerate(combined, 1):
                if e["ready"]:
                    out(f"    [{i}] {e['alias']:<16} {e['display_name']}")
                else:
                    out(f"    [{i}] {e['alias']:<16} {e['display_name']:<10}  {e['ready_reason']}")
            out("")
            out("  Choose [1]: ", end="")
            sys.stderr.flush()
            choice = input().strip() or "1"
            try:
                idx = int(choice) - 1
                if idx < 0 or idx >= len(combined):
                    out("  Invalid choice.")
                    sys.exit(1)
                entry = combined[idx]
            except ValueError:
                out("  Invalid choice.")
                sys.exit(1)
            if not entry["ready"]:
                if not _launch_inline_setup(entry, out):
                    sys.exit(1)
                _credentials_changed = True
            model = {"alias": entry["alias"], "model": entry["model"], "provider": entry["provider"],
                     "litellm_params": {"model": entry["model"]}}

    # Ensure selected model exists in litellm_config.yaml
    existing_aliases = {m["alias"] for m in config.list_models()}
    if model["alias"] not in existing_aliases:
        provider = providers.get_provider(model["provider"])
        extra = {}
        if hasattr(provider, "get_extra_params"):
            extra = provider.get_extra_params()
        s, msg = config.add_model(model["alias"], model["model"], extra)
        if s == Status.OK:
            out(f"  Added {model['alias']} to config")
            _credentials_changed = True  # Config changed — need restart

    # Step 4: skip — check_ready() already validated credentials locally.
    # Full network validation (validate()) is too slow and depends on LiteLLM
    # being reachable, which isn't guaranteed during first boot.

    # Step 5: Thinking effort (interactive if the selected model has a verified contract)
    thinking_contract = config.resolve_thinking_contract(model)
    supported_levels = tuple((thinking_contract or {}).get("levels") or ())
    if thinking and not thinking_contract:
        out(f"  ✗ Thinking effort is not supported for model '{model['alias']}'.")
        out("    The configured upstream route is not verified for thinking control.")
        sys.exit(1)
    if thinking and supported_levels and thinking not in supported_levels:
        labels = ", ".join(_thinking_level_label(level, thinking_contract) for level in supported_levels)
        out(f"  ✗ Invalid thinking effort '{thinking}' for model '{model['alias']}'.")
        out(f"    Valid options: {labels}")
        sys.exit(1)

    if not thinking and thinking_contract:
        default_level = thinking_contract.get("default_level")
        if len(supported_levels) == 1:
            thinking = supported_levels[0]
            out(f"  Thinking effort: {_thinking_level_label(thinking, thinking_contract)}")
        else:
            out(f"\n  Thinking effort:\n")
            for i, level in enumerate(supported_levels, 1):
                label = _thinking_level_label(level, thinking_contract)
                default_suffix = " (default)" if level == default_level else ""
                description = _thinking_level_description(level)
                if description:
                    out(f"    [{i}] {label}{default_suffix}  {description}")
                else:
                    out(f"    [{i}] {label}{default_suffix}")
            out("")
            out("  Choose (Enter for default): ", end="")
            sys.stderr.flush()
            tc = input().strip()
            if not tc:
                thinking = default_level
            else:
                try:
                    thinking = supported_levels[int(tc) - 1]
                except (ValueError, IndexError):
                    out("  Invalid choice.")
                    sys.exit(1)

    # Step 6: Read master key
    try:
        master_key = config.ensure_master_key()
    except RuntimeError as e:
        out(f"  ✗ {e}")
        sys.exit(1)

    # Step 7: Telegram binding (skip prompt if flag was passed)
    telegram = _kwargs.get("telegram")
    if telegram is None:
        out("\n  Bind session to Telegram? [y/N]: ", end="")
        sys.stderr.flush()
        telegram = input().strip().lower() == "y"

    # Step 8: Launch or print env
    log.debug("Launching Claude Code: model=%s provider=%s thinking=%s telegram=%s",
              model["alias"], model["provider"], thinking or "default", telegram)

    # Check for --emit-env mode (used by proclaude.sh to get env without exec'ing)
    alias = model["alias"]
    _persist_selected_model_state(alias)

    # Resolve model limits for the selected model
    provider = providers.get_provider(model["provider"])
    limits = getattr(provider, "model_limits", {}).get(alias) if provider else None

    if emit_env:
        with open(emit_env, "w") as f:
            f.write(f"export ANTHROPIC_BASE_URL='http://localhost:{PORT}'\n")
            f.write(f"export ANTHROPIC_AUTH_TOKEN='{master_key}'\n")
            f.write(f"export ANTHROPIC_MODEL='{alias}'\n")
            f.write(f"export CLAUDE_SELECTED_PROVIDER='{model['provider']}'\n")
            f.write(f"export ANTHROPIC_DEFAULT_HAIKU_MODEL='{alias}'\n")
            f.write(f"export ANTHROPIC_DEFAULT_SONNET_MODEL='{alias}'\n")
            f.write(f"export ANTHROPIC_DEFAULT_OPUS_MODEL='{alias}'\n")
            if thinking:
                f.write(f"export ANTHROPIC_CUSTOM_HEADERS='x-thinking-effort: {thinking}'\n")
            if telegram:
                channel = os.environ.get("TELEGRAM_CHANNEL", "plugin:telegram@claude-plugins-official")
                f.write(f"export CLAUDE_CHANNELS='{channel}'\n")
            if limits:
                if limits.get("context"):
                    f.write(f"export CLAUDE_CODE_MAX_MODEL_TOKENS='{limits['context']}'\n")
                if limits.get("max_output"):
                    f.write(f"export MAX_OUTPUT_TOKENS='{limits['max_output']}'\n")
            # Emit the command to run — shell just sources this and exec's LAUNCH_CMD
            cmd_parts = "claude --dangerously-skip-permissions"
            if telegram:
                channel = os.environ.get("TELEGRAM_CHANNEL", "plugin:telegram@claude-plugins-official")
                cmd_parts += f" --channels '{channel}'"
            f.write(f"LAUNCH_CMD='{cmd_parts}'\n")
            if _credentials_changed:
                f.write("NEEDS_RESTART=1\n")
            if _needs_browser_oauth:
                f.write("NEEDS_BROWSER_OAUTH=1\n")
            # Pass collected credentials to shell for host-side .env write
            for cred_key, cred_val in _pending_credentials.items():
                f.write(f"export SET_ENV_{cred_key}='{cred_val}'\n")
        return

    # Normal mode: exec claude
    out(f"  Launching Claude Code ({alias})...")

    os.environ["ANTHROPIC_BASE_URL"] = f"http://localhost:{PORT}"
    os.environ["ANTHROPIC_AUTH_TOKEN"] = master_key
    os.environ["ANTHROPIC_MODEL"] = alias
    os.environ["ANTHROPIC_DEFAULT_HAIKU_MODEL"] = alias
    os.environ["ANTHROPIC_DEFAULT_SONNET_MODEL"] = alias
    os.environ["ANTHROPIC_DEFAULT_OPUS_MODEL"] = alias
    if thinking:
        os.environ["ANTHROPIC_CUSTOM_HEADERS"] = f"x-thinking-effort: {thinking}"
        print(f"  Thinking effort: {thinking}")
    cmd = [claude_bin, "--dangerously-skip-permissions"]
    if telegram:
        channel = os.environ.get("TELEGRAM_CHANNEL", "plugin:telegram@claude-plugins-official")
        cmd += ["--channels", channel]
    cmd += extra_args
    os.execvp(claude_bin, cmd)


# --- Router ---

def _init_registry():
    """Populate SUBCOMMAND_REGISTRY with handler references (called after all functions are defined)."""
    handlers = {
        "model": {"add": cmd_model_add, "rm": cmd_model_rm, "list": cmd_model_list},
        "provider": {"list": cmd_provider_list, "status": cmd_provider_status,
                     "login": cmd_provider_login, "logout": cmd_provider_logout,
                     "openai-browser-trigger": cmd_openai_browser_trigger},
        "launch": {"claude": cmd_launch_claude},
    }
    for group, subs in handlers.items():
        for sub, handler in subs.items():
            entry = SUBCOMMAND_REGISTRY[group][sub]
            SUBCOMMAND_REGISTRY[group][sub] = (handler, entry[1], entry[2])


def _parse_flags(args):
    """Extract --provider, --model, --thinking flags from args.
    Returns (provider, model, remaining_args, extra_flags)."""
    provider = None
    model = None
    extra_flags = {}
    remaining = []
    i = 0
    while i < len(args):
        if args[i] == "--provider" and i + 1 < len(args):
            provider = args[i + 1]
            i += 2
        elif args[i] == "--model" and i + 1 < len(args):
            model = args[i + 1]
            i += 2
        elif args[i] == "--thinking" and i + 1 < len(args):
            extra_flags["thinking"] = args[i + 1]
            i += 2
        elif args[i] == "--telegram":
            extra_flags["telegram"] = True
            i += 1
        elif args[i] == "--no-telegram":
            extra_flags["telegram"] = False
            i += 1
        elif args[i] == "--emit-env" and i + 1 < len(args):
            extra_flags["emit_env"] = args[i + 1]
            i += 2
        elif args[i] == "--":
            remaining.extend(args[i + 1:])
            break
        else:
            remaining.append(args[i])
            i += 1
    return provider, model, remaining, extra_flags


def main():
    import config

    args = sys.argv[1:]

    # Parse --verbose / -v from anywhere in args
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]

    _setup_logging(verbose)
    log.debug("CLI started with args: %s", args)

    if not args or args[0] in ("help", "-h", "--help"):
        show_help()
        return

    config._ensure_env()

    cmd = args[0]
    rest = args[1:]
    log.debug("Executing command: %s %s", cmd, rest)

    # --- Infrastructure commands redirected to proclaude.sh ---
    if cmd in ("start", "stop", "restart", "logs"):
        print(f"  '{cmd}' is handled by proclaude.sh directly.")
        print(f"  Run: ./proclaude.sh {cmd}")
        return

    if cmd == "start-status":
        cmd_start_status()
        return

    if cmd == "status":
        if any(a in ("-h", "--help") for a in rest):
            show_help()
            return
        if rest:
            print(f"  'status' does not accept arguments: {' '.join(rest)}")
            sys.exit(1)
        cmd_status()

    # --- Subcommand groups (help only before -- boundary) ---
    elif cmd in SUBCOMMAND_REGISTRY:
        if not rest or rest[0] in ("-h", "--help"):
            _show_group_help(cmd)
            return

        sub = rest[0]
        sub_rest = rest[1:]

        # Check for help flags only before the -- passthrough boundary
        for a in sub_rest:
            if a == "--":
                break
            if a in ("-h", "--help"):
                _show_group_help(cmd)
                return

        provider_flag, model_flag, remaining, extra_flags = _parse_flags(sub_rest)

        entry = SUBCOMMAND_REGISTRY.get(cmd, {}).get(sub)
        if not entry or entry[0] is None:
            print(f"  Unknown subcommand: {cmd} {sub}")
            _show_group_help(cmd)
            sys.exit(1)

        handler = entry[0]
        handler(provider_flag=provider_flag, model_flag=model_flag, extra_args=remaining, **extra_flags)

    else:
        print(f"  Unknown command: {cmd}")
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    _init_registry()
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(130)
    except EOFError:
        print("\n  Cancelled (not interactive).")
        sys.exit(1)
    except Exception as e:
        log.debug("Unhandled exception", exc_info=True)
        print(f"  \u2717 Unexpected error: {e}")
        sys.exit(1)
