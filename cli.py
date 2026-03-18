#!/usr/bin/env python3
import logging
import sys
import os

DIR = os.path.dirname(os.path.abspath(__file__))
PORT = 2555

log = logging.getLogger("litellm-cli")


def _setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="  [%(levelname)s] %(name)s: %(message)s",
    )


def show_help():
    name = os.path.basename(sys.argv[0]) if sys.argv[0] else "./litellm.sh"
    print("LiteLLM Gateway CLI")
    print(f"Usage: {name} [COMMAND] [OPTIONS]")
    print()
    print("Lifecycle:")
    print("  up              Start the proxy container")
    print("  down            Stop and remove the container")
    print("  restart         Restart the container")
    print("  status          Container and model status")
    print("  logs            Stream container logs")
    print()
    print("Models:")
    print("  add             Add a model or provider (interactive wizard)")
    print("  remove          Remove a configured model")
    print("  models          List configured models")
    print()
    print("Auth:")
    print("  login [provider]  Authenticate with a provider")
    print("                    No arg: show auth status for all providers")
    print()
    print("Options:")
    print("  --verbose, -v   Enable debug logging")


def cmd_status():
    import container
    import config
    import providers

    running, output = container.status()
    state = "running" if running else "stopped"
    print(f"Container:  litellm-proxy  [{state}]")
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
        if provider and running:
            if m["provider"] not in auth_cache:
                auth_cache[m["provider"]] = provider.validate()
            auth_status, _ = auth_cache[m["provider"]]
            if auth_status.value == "ok":
                icon = "✓"
                label = "authenticated" if m["provider"] != "ollama" else "reachable"
            elif auth_status.value == "not_configured":
                icon = "✗"
                label = "not configured"
            elif auth_status.value == "unreachable":
                icon = "✗"
                label = "unreachable"
            else:
                icon = "✗"
                label = "invalid"
        else:
            icon = "-"
            label = "unknown" if not running else "unknown provider"
        print(f"  {m['alias']:<12} {m['provider']:<10} {icon} {label}")


def cmd_models():
    import config
    models = config.list_models()
    if not models:
        print("No models configured.")
        return
    print("Configured models:")
    for m in models:
        print(f"  {m['alias']:<12} {m['provider']:<10} ({m['model']})")


def cmd_login(provider_name=None):
    import providers

    if provider_name is None:
        print("Provider auth status:\n")
        for p in providers.all_providers():
            status, msg = p.validate()
            if status.value == "ok":
                print(f"  {p.display_name:<20} ✓ {msg}")
            else:
                print(f"  {p.display_name:<20} ✗ {msg}")
        return

    provider = providers.get_provider(provider_name)
    if not provider:
        print(f"Unknown provider: {provider_name}")
        print(f"Available: {', '.join(p.name for p in providers.all_providers())}")
        sys.exit(1)

    status, msg = provider.validate()
    if status.value == "ok":
        print(f"  ✓ Already authenticated with {provider.display_name}. {msg}")
        return

    auth_type = None
    if len(provider.auth_types) == 0:
        print(f"  {provider.display_name} doesn't require authentication.")
        ok, msg = provider.login()
        print(f"  {'✓' if ok else '✗'} {msg}")
        return
    elif len(provider.auth_types) == 1:
        auth_type = provider.auth_types[0]
    else:
        print(f"\n  {provider.display_name} supports multiple auth methods:\n")
        for i, at in enumerate(provider.auth_types, 1):
            label = at.replace("_", " ").title()
            print(f"    [{i}] {label}")
        print()
        choice = input("  Choose [1]: ").strip() or "1"
        try:
            idx = int(choice) - 1
            auth_type = provider.auth_types[idx]
        except (ValueError, IndexError):
            print("  Invalid choice.")
            sys.exit(1)

    ok, msg = provider.login(auth_type)
    if ok:
        print(f"\n  ✓ {msg}")
    else:
        print(f"\n  ✗ {msg}")
        sys.exit(1)


def _print_restart_failure():
    """Print container failure message with backup info if available."""
    import config
    print(f"  ✗ Container failed to start. Check './litellm.sh logs' for details.")
    if os.path.exists(config.CONFIG_BACKUP):
        print(f"    Your previous config was backed up to litellm_config.yaml.bak")


def cmd_add():
    import config
    import container
    import providers

    print("\n  What would you like to add?\n")
    print("    [1] A provider (then pick models)")
    print("    [2] A specific model")
    print()
    choice = input("  Choose [1]: ").strip() or "1"

    if choice == "1":
        _add_provider_first()
    elif choice == "2":
        _add_model_first()
    else:
        print("  Invalid choice.")
        sys.exit(1)


def _add_provider_first():
    import config
    import container
    import providers

    all_provs = providers.all_providers()
    print(f"\n  Select a provider:\n")
    for i, p in enumerate(all_provs, 1):
        print(f"    [{i}] {p.display_name}")
    print()
    choice = input("  Choose: ").strip()
    try:
        provider = all_provs[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid choice.")
        sys.exit(1)

    auth_type = None
    if provider.auth_types:
        if len(provider.auth_types) == 1:
            auth_type = provider.auth_types[0]
        else:
            print(f"\n  Auth method for {provider.display_name}:\n")
            for i, at in enumerate(provider.auth_types, 1):
                label = at.replace("_", " ").title()
                print(f"    [{i}] {label}")
            print()
            at_choice = input("  Choose [1]: ").strip() or "1"
            try:
                auth_type = provider.auth_types[int(at_choice) - 1]
            except (ValueError, IndexError):
                print("  Invalid choice.")
                sys.exit(1)

        status, msg = provider.validate()
        if status.value != "ok":
            print(f"\n  Need to authenticate with {provider.display_name}.")
            ok, msg = provider.login(auth_type)
            if not ok:
                print(f"\n  ✗ {msg}")
                sys.exit(1)
            print(f"  ✓ {msg}")

    if provider.name == "ollama":
        catalog = provider.discover_models()
        if not catalog:
            print("\n  ✗ Ollama is not running. Start it and try again.")
            sys.exit(1)
    else:
        if auth_type and hasattr(provider, "get_models_for_auth"):
            catalog = provider.get_models_for_auth(auth_type)
        else:
            catalog = provider.models

    if not catalog:
        print("\n  No models available for this provider.")
        sys.exit(1)

    aliases = list(catalog.keys())
    print(f"\n  Available models for {provider.display_name}:\n")
    for i, alias in enumerate(aliases, 1):
        print(f"    [{i}] {alias}")
    print(f"    [a] All")
    print()
    model_choice = input("  Choose (comma-separated, e.g. 1,3): ").strip()

    selected = []
    if model_choice.lower() == "a":
        selected = aliases
    else:
        for part in model_choice.split(","):
            try:
                idx = int(part.strip()) - 1
                selected.append(aliases[idx])
            except (ValueError, IndexError):
                print(f"  Skipping invalid choice: {part.strip()}")

    if not selected:
        print("  No models selected.")
        sys.exit(1)

    added = []
    for alias in selected:
        model_str = catalog[alias]
        existing = config.list_models()
        existing_aliases = [m["alias"] for m in existing]
        final_alias = alias
        if alias in existing_aliases:
            print(f"\n  Alias '{alias}' already exists.")
            final_alias = input(f"  Enter a different alias (or Enter to skip): ").strip()
            if not final_alias or final_alias in existing_aliases:
                print(f"  Skipping {alias}.")
                continue

        extra = {}
        if provider.name == "ollama":
            extra = provider.get_extra_params()

        ok, msg = config.add_model(final_alias, model_str, extra)
        if ok:
            added.append(final_alias)
            print(f"  ✓ {msg}")
        else:
            print(f"  ✗ {msg}")

    if not added:
        print("\n  No models added.")
        return

    print(f"\n  Restarting container...")
    log.debug("Restarting after adding models: %s", added)
    if not container.restart():
        _print_restart_failure()
        sys.exit(1)
    if container.wait_healthy():
        status, msg = provider.validate()
        if status.value == "ok":
            print(f"  ✓ Container is running. Added: {', '.join(added)}. {msg}")
        else:
            print(f"  ⚠ Container is running. Added: {', '.join(added)}")
            print(f"    Auth check: {msg}")
    else:
        _print_restart_failure()
        sys.exit(1)


def _add_model_first():
    import config
    import container
    import providers

    combined = {}
    for p in providers.all_providers():
        if p.name == "ollama":
            ollama_models = p.discover_models()
            if not ollama_models:
                print("  (Ollama not running — skipping its models)")
            else:
                for alias, model_str in ollama_models.items():
                    key = f"{alias} ({p.display_name})"
                    combined[key] = (p, alias, model_str)
        else:
            for alias, model_str in p.models.items():
                key = f"{alias} ({p.display_name})"
                combined[key] = (p, alias, model_str)

    if not combined:
        print("  No models available from any provider.")
        sys.exit(1)

    keys = list(combined.keys())
    print(f"\n  Available models:\n")
    for i, key in enumerate(keys, 1):
        print(f"    [{i}] {key}")
    print()
    choice = input("  Choose: ").strip()
    try:
        key = keys[int(choice) - 1]
    except (ValueError, IndexError):
        print("  Invalid choice.")
        sys.exit(1)

    provider, alias, model_str = combined[key]

    auth_type = None
    if provider.auth_types:
        status, msg = provider.validate()
        if status.value != "ok":
            if len(provider.auth_types) > 1:
                print(f"\n  Auth method for {provider.display_name}:\n")
                for i, at in enumerate(provider.auth_types, 1):
                    label = at.replace("_", " ").title()
                    print(f"    [{i}] {label}")
                print()
                at_choice = input("  Choose [1]: ").strip() or "1"
                try:
                    auth_type = provider.auth_types[int(at_choice) - 1]
                except (ValueError, IndexError):
                    print("  Invalid choice.")
                    sys.exit(1)
            else:
                auth_type = provider.auth_types[0]
            print(f"\n  Need to authenticate with {provider.display_name}.")
            ok, msg = provider.login(auth_type)
            if not ok:
                print(f"\n  ✗ {msg}")
                sys.exit(1)
            print(f"  ✓ {msg}")
        else:
            # Already authenticated — detect which auth type is active
            auth_type = provider.detect_auth_type()

        # Resolve model string based on auth type
        if auth_type and hasattr(provider, "get_model_string"):
            new_model_str = provider.get_model_string(alias, auth_type)
            if new_model_str:
                model_str = new_model_str

    existing_aliases = [m["alias"] for m in config.list_models()]
    final_alias = alias
    if alias in existing_aliases:
        print(f"\n  Alias '{alias}' already exists.")
        final_alias = input(f"  Enter a different alias: ").strip()
        if not final_alias or final_alias in existing_aliases:
            print("  Cancelled.")
            sys.exit(1)
    else:
        custom = input(f"  Alias [{alias}]: ").strip()
        if custom:
            final_alias = custom

    extra = {}
    if provider.name == "ollama":
        extra = provider.get_extra_params()

    ok, msg = config.add_model(final_alias, model_str, extra)
    if not ok:
        print(f"  ✗ {msg}")
        sys.exit(1)
    print(f"  ✓ {msg}")

    print(f"\n  Restarting container...")
    log.debug("Restarting after adding model: %s", final_alias)
    if not container.restart():
        _print_restart_failure()
        sys.exit(1)
    if container.wait_healthy():
        status, msg = provider.validate()
        if status.value == "ok":
            print(f"  ✓ Container is running with '{final_alias}'. {msg}")
        else:
            print(f"  ⚠ Container is running with '{final_alias}'")
            print(f"    Auth check: {msg}")
    else:
        _print_restart_failure()
        sys.exit(1)


def cmd_remove():
    import config
    import container
    import providers

    models = config.list_models()
    if not models:
        print("  No models configured.")
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
        f"  Remove {names}? This will restart the container. [y/N]: "
    ).strip().lower()
    if confirm != "y":
        print("  Cancelled.")
        return

    removed_providers = set()
    for model in selected:
        ok, msg, provider_name = config.remove_model(model["alias"])
        if ok:
            print(f"  ✓ {msg}")
            if provider_name:
                removed_providers.add(provider_name)
        else:
            print(f"  ✗ {msg}")

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
                        for var in all_env_vars:
                            config.remove_env(var)
                        print(f"  ✓ Cleaned up env vars.")

    print(f"\n  Restarting container...")
    log.debug("Restarting after removing models")
    if not container.restart():
        _print_restart_failure()
        sys.exit(1)
    if container.wait_healthy():
        print(f"  ✓ Container is running.")
    else:
        _print_restart_failure()
        sys.exit(1)


def main():
    import config

    args = sys.argv[1:]

    # Parse --verbose / -v from anywhere in args
    verbose = "--verbose" in args or "-v" in args
    args = [a for a in args if a not in ("--verbose", "-v")]

    _setup_logging(verbose)
    log.debug("CLI started with args: %s", args)

    config._ensure_env()

    if not args or args[0] in ("help", "-h", "--help"):
        show_help()
        return

    cmd = args[0]
    log.debug("Executing command: %s", cmd)

    if cmd == "up":
        import container
        if not container.up():
            sys.exit(1)
    elif cmd == "down":
        import container
        if not container.down():
            sys.exit(1)
    elif cmd == "restart":
        import container
        if not container.restart():
            sys.exit(1)
    elif cmd == "status":
        cmd_status()
    elif cmd == "logs":
        import container
        container.logs()
    elif cmd == "models":
        cmd_models()
    elif cmd == "login":
        provider_name = args[1] if len(args) > 1 else None
        cmd_login(provider_name)
    elif cmd == "add":
        cmd_add()
    elif cmd == "remove":
        cmd_remove()
    else:
        print(f"Unknown command: {cmd}")
        show_help()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n  Cancelled.")
        sys.exit(130)
