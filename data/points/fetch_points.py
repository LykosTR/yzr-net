"""Fetch the Space Radar point layers (recent earthquakes, VIIRS fires of the last 24 h) from the public Esri
Living Atlas feature services and write compact JSON: {"at": <epoch ms>, "fields": [...], "rows": [[x, y, ...], ...]}.
Every row is [lon, lat, <fields in order>]; time-dependent fields (hoursOld / hours_old) are the value AT `at`."""
import json, os, sys, time, urllib.parse, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
BASE = "https://services9.arcgis.com/RHVPKKiFTONKtxq3/arcgis/rest/services/"
LAYERS = {
    "quakes": dict(
        svc="USGS_Seismic_Data_v1", where="1=1", page=2000, min_rows=50,
        fields=["OBJECTID", "mag", "place", "eventTime", "depth", "hoursOld", "alert", "tsunami", "url"]),
    "fires": dict(
        svc="Satellite_VIIRS_Thermal_Hotspots_and_Fire_Activity", page=4000, min_rows=500,
        where="hours_old<=24 AND (confidence='high' OR frp>=10)",
        fields=["OBJECTID", "frp", "hours_old"]),
}


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "yzr-net-points/1.0"})
            return json.load(urllib.request.urlopen(req, timeout=90))
        except Exception as e:
            print("retry", attempt, e)
            time.sleep(3)
    sys.exit("request failed: " + url[:120])


def fetch(name, cfg):
    rows, off = [], 0
    while True:
        q = urllib.parse.urlencode({
            "where": cfg["where"], "outFields": ",".join(cfg["fields"]), "orderByFields": "OBJECTID",
            "returnGeometry": "true", "outSR": 4326, "geometryPrecision": 4,
            "resultOffset": off, "resultRecordCount": cfg["page"], "f": "json"})
        j = get(BASE + cfg["svc"] + "/FeatureServer/0/query?" + q)
        if "error" in j:
            sys.exit("%s: %s" % (name, j["error"]))
        feats = j.get("features", [])
        for f in feats:
            g, a = f.get("geometry"), f.get("attributes", {})
            if not g or g.get("x") is None or g.get("y") is None:
                continue
            rows.append([round(g["x"], 4), round(g["y"], 4)] + [a.get(k) for k in cfg["fields"]])
        if j.get("exceededTransferLimit") and feats and len(rows) < 60000:
            off += len(feats)
            continue
        return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    for name, cfg in LAYERS.items():
        at = int(time.time() * 1000)
        rows = fetch(name, cfg)
        if len(rows) < cfg["min_rows"]:
            sys.exit("%s: only %d rows, refusing to publish" % (name, len(rows)))
        with open(os.path.join(OUT, name + ".json"), "w", encoding="utf-8") as f:
            json.dump({"at": at, "fields": cfg["fields"], "rows": rows}, f, separators=(",", ":"), ensure_ascii=False)
        print(name, len(rows), "rows")


main()
