"""
Fetch daily max/min temperature history for a NOAA GHCN-Daily station.

Default station: GHCND:USW00094728 (NYC Central Park), which is the
settlement source for Kalshi's NYC daily-high-temperature contracts.

Uses NOAA's public Climate Data Online (CDO) "access/services/data" endpoint,
which does not require an API token for the plain-text/CSV format used here.
If NOAA changes this, an NCEI token can be passed via --token or the
NOAA_TOKEN environment variable and used with the v2 CDO API instead
(see `fetch_via_cdo_v2` below).

Run from a machine with normal internet access:

    python src/fetch_data.py --station GHCND:USW00094728 \
        --start 2000-01-01 --end 2026-07-01 \
        --out data/central_park_daily.csv
"""
import argparse
import io
import os
import sys
import time

import pandas as pd
import requests

NCEI_ACCESS_URL = "https://www.ncei.noaa.gov/access/services/data/v1"
CDO_V2_URL = "https://www.ncdc.noaa.gov/cdo-web/api/v2/data"


def fetch_via_access_api(station: str, start: str, end: str) -> pd.DataFrame:
    """
    Pull daily TMAX/TMIN via the NCEI 'access/services/data' endpoint.
    This endpoint returns CSV directly and does not require a token.
    """
    params = {
        "dataset": "daily-summaries",
        "stations": station.replace("GHCND:", ""),
        "startDate": start,
        "endDate": end,
        "dataTypes": "TMAX,TMIN,PRCP,SNOW",
        "units": "standard",
        "format": "csv",
        "includeAttributes": "false",
    }
    resp = requests.get(NCEI_ACCESS_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.read_csv(io.StringIO(resp.text), parse_dates=["DATE"])
    return df


def fetch_via_cdo_v2(station: str, start: str, end: str, token: str) -> pd.DataFrame:
    """
    Fallback using the token-authenticated CDO v2 API. Paginates in
    1-year chunks since the v2 API caps date ranges and result counts.
    """
    headers = {"token": token}
    frames = []
    start_year = int(start[:4])
    end_year = int(end[:4])
    for year in range(start_year, end_year + 1):
        y_start = f"{year}-01-01"
        y_end = f"{year}-12-31"
        offset = 1
        while True:
            params = {
                "datasetid": "GHCND",
                "stationid": station,
                "datatypeid": "TMAX,TMIN",
                "startdate": y_start,
                "enddate": y_end,
                "units": "standard",
                "limit": 1000,
                "offset": offset,
            }
            r = requests.get(CDO_V2_URL, headers=headers, params=params, timeout=60)
            r.raise_for_status()
            payload = r.json()
            results = payload.get("results", [])
            if not results:
                break
            frames.append(pd.DataFrame(results))
            if len(results) < 1000:
                break
            offset += 1000
            time.sleep(0.3)  # CDO v2 rate limit: 5 req/sec, 10k req/day
    if not frames:
        return pd.DataFrame()
    long_df = pd.concat(frames, ignore_index=True)
    wide = long_df.pivot_table(
        index="date", columns="datatype", values="value", aggfunc="first"
    ).reset_index()
    wide["date"] = pd.to_datetime(wide["date"])
    return wide


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--station", default="GHCND:USW00094728")
    p.add_argument("--start", default="2000-01-01")
    p.add_argument("--end", default=pd.Timestamp.today().strftime("%Y-%m-%d"))
    p.add_argument("--out", default="data/central_park_daily.csv")
    p.add_argument("--token", default=os.environ.get("NOAA_TOKEN"))
    p.add_argument(
        "--method",
        choices=["access", "cdo_v2"],
        default="access",
        help="'access' = no-token CSV endpoint (default); 'cdo_v2' = token-based fallback",
    )
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    if args.method == "access":
        df = fetch_via_access_api(args.station, args.start, args.end)
    else:
        if not args.token:
            sys.exit("--method cdo_v2 requires --token or NOAA_TOKEN env var")
        df = fetch_via_cdo_v2(args.station, args.start, args.end, args.token)

    if df.empty:
        sys.exit("No data returned — check station id, date range, and network access.")

    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} rows to {args.out}")


if __name__ == "__main__":
    main()
