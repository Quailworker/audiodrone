"""Where the language model and the site data live. Environment variables override the defaults.

LLM_BASE_URL   OpenAI-compatible endpoint. Meterless Nightshift serves one on the LAN
               (e.g. http://<pc>:8088/v1, key "rt-local"). Default: local Ollama, the stand-in on Linux.
LLM_API_KEY    API key for that endpoint.
LLM_MODEL      Nemotron model name as the endpoint knows it (Ollama: nemotron-mini; Nightshift: the
               name its model library shows, e.g. a Nemotron 3 Nano build).
LLM_TIMEOUT_S  Seconds the Nemotron layer may take per message before the template is used.
LLM_EXTRA_BODY JSON added to every model request, e.g. {"chat_template_kwargs": {"enable_thinking": false}}:
               Nemotron 3.5 through Nightshift's NVIDIA provider otherwise spends a small token budget on
               reasoning and returns an empty reply.
SITE_MCP       Command that starts the site-data MCP server (stdio). Default: the local stand-in
               (alerting/sitedata_mcp.py). For Zetaris, the command or URL of its MCP server.
SITE_FILE      JSON site file for the stand-in.

The same variables can live in a private env file (KEY=VALUE lines), ALERTING_ENV or by default
~/.config/audiodrone/alerting.env, so keys stay out of the repo. Real environment variables win.
"""
import json
import os
import shlex
import sys
from dataclasses import dataclass, field

from alerting.sitedata import DEFAULT_SITE


def _load_env_file():
    path = os.path.expanduser(os.environ.get("ALERTING_ENV", "~/.config/audiodrone/alerting.env"))
    if not os.path.isfile(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())


_load_env_file()


@dataclass
class Config:
    llm_base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "rt-local"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "nemotron-mini"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT_S", "20")))
    # whole operator answer (all tool rounds); after it the facts-only fallback answer is given
    answer_deadline_s: float = field(default_factory=lambda: float(os.environ.get("ANSWER_DEADLINE_S", "45")))
    llm_max_tokens: int = field(default_factory=lambda: int(os.environ.get("LLM_MAX_TOKENS", "300")))
    llm_extra_body: dict = field(default_factory=lambda: json.loads(os.environ.get("LLM_EXTRA_BODY", "{}") or "{}"))
    site_file: str = field(default_factory=lambda: os.environ.get("SITE_FILE", str(DEFAULT_SITE)))
    site_mcp: str = field(default_factory=lambda: os.environ.get("SITE_MCP", ""))

    def site_mcp_command(self):
        """(command, args) for the MCP stdio client."""
        if self.site_mcp:
            parts = shlex.split(self.site_mcp)
            return parts[0], parts[1:]
        return sys.executable, ["-m", "alerting.sitedata_mcp", "--site", self.site_file]
