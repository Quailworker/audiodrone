"""Build mode (terra/build.py, terra/server.py): listening ranges, run-time estimate, request -> scenario."""
from terra import server
from terra.build import estimate_s, listening_range_m, ranges


def test_ranges_follow_loudness_and_noise():
    r = ranges()["ranges"]
    for amb in r:
        assert r[amb]["mid"] < r[amb]["fpv"] < r[amb]["ts_mr"] < r[amb]["ts_fw"]     # quieter drones, shorter range
    for cls in r["desert"]:
        assert r["desert"][cls] > r["desert_wind"][cls] > r["town"][cls] > r["highway"][cls]


def test_ranges_match_the_full_simulation():
    # results/spacing: the 7-12 in quad first heard 0.9-1.2 km out in quiet desert;
    # results/swarm_abudhabi: in town the quads went unheard at 200-400 m, the petrol multirotor was tracked from 506 m
    assert 700 <= listening_range_m("mid", "desert") <= 1200
    assert listening_range_m("mid", "town") < 200
    assert 500 <= listening_range_m("ts_mr", "town") <= 900


def test_estimate_grows_with_clusters_drones_and_path():
    one = {"clusters": [[0, 0], [500, 0]], "drones": [{"cls": "mid", "speed": 14, "waypoints": [[0, -1400], [0, 0]]}]}
    more = {**one, "clusters": one["clusters"] * 4}
    two = {**one, "drones": one["drones"] * 2}
    assert estimate_s(one) < estimate_s(more) and estimate_s(one) < estimate_s(two)
    assert estimate_s({"clusters": [], "drones": []}) == 0


def test_build_request_becomes_a_checked_scenario():
    q = {"origin": [22.8205, 53.5935], "clusters": [[0, 0], [700, 0]], "ambient": "desert",
         "drones": [{"cls": "mid", "agl": 60, "speed": 14, "waypoints": [[-400, -2600], [0, 0]]}],
         "sites": [{"name": "Protected site", "x": 0, "y": 0, "radius_m": 150}]}
    scn = server._build_scenario(q)
    assert scn["chain"] == "v2" and scn["terrain"] and server._build_check(scn) is None
    assert scn["border"] and scn["_sector"].startswith("SAU")            # near the Saudi border: crossing alerts on
    assert 2600 < scn["half_m"] <= server.BUILD_MAX["half_m"]
    city = server._build_scenario({**q, "origin": [24.47, 54.37]})
    assert city["border"] == [] and city["_sector"] == "SITE"            # Abu Dhabi: no border nearby
    assert "two directions" in server._build_check({**scn, "clusters": [[0, 0]]})
    assert "path" in server._build_check({**scn, "drones": [{**scn["drones"][0], "waypoints": [[0, 0]]}]})
