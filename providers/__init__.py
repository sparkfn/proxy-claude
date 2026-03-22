from providers.ollama import OllamaProvider
from providers.minimax import MiniMaxProvider
from providers.openai import OpenAIProvider

_PROVIDERS = {}


def _register(cls):
    instance = cls()
    _PROVIDERS[instance.name] = instance


def get_provider(name):
    """Look up a provider by name. Returns None if not found."""
    return _PROVIDERS.get(name)


def all_providers():
    """Return list of all registered provider instances."""
    return list(_PROVIDERS.values())


# Register providers (order matters for display)
_register(OpenAIProvider)
_register(MiniMaxProvider)
_register(OllamaProvider)
