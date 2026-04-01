"""Routing helpers extracted from gateway/proxy.py."""

from dataclasses import dataclass
import importlib
import logging
import os
import urllib.parse
from typing import Callable, Iterable

log = logging.getLogger("litellm-proxy.v2.routes")


@dataclass(frozen=True)
class NativeAnthropicRoute:
    host: str
    port: int
    path: str
    api_key_env: str | None
    auth_header: str

    def as_dict(self):
        return {
            "host": self.host,
            "port": self.port,
            "path": self.path,
            "api_key_env": self.api_key_env,
            "auth_header": self.auth_header,
        }


@dataclass
class RouteState:
    translated_models: set
    all_models: list
    native_routes: dict
    thinking_contracts: dict

    def __getitem__(self, key):
        if key == "translated":
            return self.translated_models
        if key == "all_models":
            return self.all_models
        if key == "native":
            return {
                model_name: route.as_dict()
                for model_name, route in self.native_routes.items()
            }
        if key == "thinking_contracts":
            return self.thinking_contracts
        raise KeyError(key)


@dataclass(frozen=True)
class RouteDependencies:
    provider_registry: Callable[[], Iterable]
    provider_from_model: Callable
    thinking_contract_resolver: Callable


def _default_route_dependencies():
    config_mod = _load_module("config")
    providers_mod = _load_module("providers")
    return RouteDependencies(
        provider_registry=providers_mod.all_providers,
        provider_from_model=config_mod._provider_from_model,
        thinking_contract_resolver=config_mod.resolve_thinking_contract,
    )


def _load_module(module_name):
    candidates = (
        f"gateway.{module_name}",
        module_name,
    )
    last_error = None
    for candidate in candidates:
        try:
            return importlib.import_module(candidate)
        except ImportError as exc:
            last_error = exc
    raise last_error


def build_route_state(
    entries,
    *,
    dependencies=None,
    provider_registry=None,
    provider_from_model=None,
    thinking_contract_resolver=None,
):
    """Build cached routing and thinking state from config model entries."""
    translated = set()
    all_models = []
    native = {}
    thinking_contracts = {}

    if dependencies is None:
        dependencies = _default_route_dependencies()
    if provider_registry is None:
        provider_registry = getattr(dependencies, "provider_registry", None)
    if provider_from_model is None:
        provider_from_model = getattr(dependencies, "provider_from_model", None)
    if thinking_contract_resolver is None:
        thinking_contract_resolver = getattr(dependencies, "thinking_contract_resolver", None)

    if provider_registry is None or provider_from_model is None or thinking_contract_resolver is None:
        raise ValueError(
            "route dependencies must provide provider_registry, provider_from_model, and thinking_contract_resolver"
        )

    provider_for_model = {}
    for provider in provider_registry():
        if provider.anthropic_base_url:
            for model_name in provider.models:
                provider_for_model[model_name] = provider

    for entry in entries:
        name = entry.get("model_name", "")
        if not name:
            continue

        all_models.append(name)
        litellm_params = dict(entry.get("litellm_params", {}) or {})
        litellm_model = litellm_params.get("model", "")
        provider_name = provider_from_model(litellm_model, litellm_params)
        model_entry = {
            "alias": name,
            "model": litellm_model,
            "provider": provider_name,
            "litellm_params": litellm_params,
        }
        thinking_contract = thinking_contract_resolver(model_entry)
        if thinking_contract:
            thinking_contracts[name] = thinking_contract
            if thinking_contract.get("requires_openai_translation"):
                translated.add(name)

        if name in provider_for_model:
            provider = provider_for_model[name]
            if not provider.native_auth:
                log.warning(
                    "Provider %s has anthropic_base_url but no native_auth — skipping native route",
                    provider.name,
                )
                continue
            api_key_env = provider.native_auth.get("env")
            auth_header = provider.native_auth.get("header", "x-api-key")
            parsed = urllib.parse.urlparse(provider.anthropic_base_url)
            native[name] = NativeAnthropicRoute(
                host=parsed.hostname,
                port=parsed.port or 443,
                path=parsed.path.rstrip("/"),
                api_key_env=api_key_env,
                auth_header=auth_header,
            )
            log.info("Native Anthropic route: %s → %s", name, provider.anthropic_base_url)

    return RouteState(
        translated_models=translated,
        all_models=all_models,
        native_routes=native,
        thinking_contracts=thinking_contracts,
    )


def resolve_config_path(module_dir):
    """Resolve the config path for host and container layouts."""
    local_config = os.path.join(module_dir, "litellm_config.yaml")
    if os.path.exists(local_config):
        return local_config
    return os.path.join(os.path.dirname(module_dir), "litellm_config.yaml")
