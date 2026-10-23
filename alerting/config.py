"""Where the language model and the site data live. Environment variables override the defaults.

LLM_BASE_URL   OpenAI-compatible endpoint. Meterless Nightshift serves one on the LAN
               (e.g. http://<pc>:8088/v1, key "rt-local"). Default: local Ollama, the stand-in on Linux.
LLM_API_KEY    API key for that endpoint.
LLM_MODEL      Nemotron model name as the endpoint knows it (Ollama: nemotron-mini; Nightshift: the
               name its model library shows, e.g. a Nemotron 3 Nano build).
LLM_TIMEOUT_S  Seconds the Nemotron layer may take per message before the template is used.
SITE_MCP       Command that starts the site-data MCP server (stdio). Default: the local stand-in
               (alerting/sitedata_mcp.py). For Zetaris, the command or URL of its MCP server.
SITE_FILE      JSON site file for the stand-in.
"""
import os
import shlex
import sys
from dataclasses import dataclass, field

from alerting.sitedata import DEFAULT_SITE


@dataclass
class Config:
    llm_base_url: str = field(default_factory=lambda: os.environ.get("LLM_BASE_URL", "http://127.0.0.1:11434/v1"))
    llm_api_key: str = field(default_factory=lambda: os.environ.get("LLM_API_KEY", "rt-local"))
    llm_model: str = field(default_factory=lambda: os.environ.get("LLM_MODEL", "nemotron-mini"))
    llm_timeout_s: float = field(default_factory=lambda: float(os.environ.get("LLM_TIMEOUT_S", "20")))
    site_file: str = field(default_factory=lambda: os.environ.get("SITE_FILE", str(DEFAULT_SITE)))
    site_mcp: str = field(default_factory=lambda: os.environ.get("SITE_MCP", ""))

    def site_mcp_command(self):
        """(command, args) for the MCP stdio client."""
        if self.site_mcp:
            parts = shlex.split(self.site_mcp)
            return parts[0], parts[1:]
        return sys.executable, ["-m", "alerting.sitedata_mcp", "--site", self.site_file]
