"""Stand-in for the Zetaris MCP server: the site-data tools over a local JSON file (stdio transport).

    python -m alerting.sitedata_mcp [--site alerting/example_site.json]

The Nemotron layer (alerting/agent.py) connects to whatever MCP server SITE_MCP names. With real
Zetaris, expose these same tool names over its federated tables (personnel, units, shelters, sites).
"""
import argparse
import json

from mcp.server.mcpserver import MCPServer

from alerting.sitedata import DEFAULT_SITE, LocalSiteData

mcp = MCPServer("site-data")
_db: LocalSiteData | None = None


@mcp.tool()
def list_sites() -> str:
    """Protected sites with their position (m, local frame) and danger radius."""
    return json.dumps(_db.list_sites())


@mcp.tool()
def list_shelters() -> str:
    """Shelters with position (m, local frame) and capacity."""
    return json.dumps(_db.list_shelters())


@mcp.tool()
def people_in_zone(x: float, y: float, radius: float) -> str:
    """Groups and units within `radius` metres of point (x, y), nearest first."""
    return json.dumps(_db.people_in_zone(x, y, radius))


@mcp.tool()
def nearest_shelter(x: float, y: float, avoid_x: float | None = None, avoid_y: float | None = None,
                    avoid_radius: float = 0.0) -> str:
    """Nearest shelter to (x, y) outside the circle (avoid_x, avoid_y, avoid_radius), with walking time."""
    return json.dumps(_db.nearest_shelter(x, y, avoid_x, avoid_y, avoid_radius))


@mcp.tool()
def get_group(group_id: str) -> str:
    """One group or unit: name, kind, position, size and alert channel."""
    return json.dumps(_db.get_group(group_id))


def main():
    global _db
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", default=str(DEFAULT_SITE))
    a = ap.parse_args()
    _db = LocalSiteData(a.site)
    mcp.run()


if __name__ == "__main__":
    main()
