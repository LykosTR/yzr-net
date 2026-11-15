"""Fetch the NOAA SWPC OVATION aurora nowcast (1 degree global grid) and write a compact JSON for the browser:
{"at": <epoch ms>, "obs": "...", "fc": "...", "w": 360, "h": 181, "b64": <base64 of w*h bytes, row-major (lat+90)*360+lon, value 0-100>}."""
import base64, json, os, sys, time, urllib.request

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
URL = "https://services.swpc.noaa.gov/json/ovation_aurora_latest.json"
W, H = 360, 181


def get(url):
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "yzr-net-aurora/1.0"})
            return json.load(urllib.request.urlopen(req, timeout=60))
        except Exception as e:
            print("retry", attempt, e)
            time.sleep(3)
    sys.exit("request failed: " + url)


def main():
    j = get(URL)
    coords = j.get("coordinates") or []
    if len(coords) < W * H * 0.9:
        sys.exit("only %d cells, refusing to publish" % len(coords))
    grid = bytearray(W * H)
    for lon, lat, v in coords:
        if 0 <= lon < W and -90 <= lat <= 90:
            grid[(lat + 90) * W + lon] = max(0, min(100, int(v)))
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "ovation.json"), "w", encoding="utf-8") as f:
        json.dump({"at": int(time.time() * 1000), "obs": j.get("Observation Time"), "fc": j.get("Forecast Time"),
                   "w": W, "h": H, "b64": base64.b64encode(bytes(grid)).decode("ascii")}, f, separators=(",", ":"))


main()
