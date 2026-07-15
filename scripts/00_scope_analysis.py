"""Scope check: forecast skill by lead, adaptive correction on 0-36h vs 168-360h."""
import numpy as np, pandas as pd, pvlib
from src.data import load_config, load_raw, prepare_forecasts, hourly_actuals

def main():
    cfg = load_config(); f, ac = load_raw(cfg); f = prepare_forecasts(f); ah = hourly_actuals(ac)
    loc = pvlib.location.Location(cfg["location"]["lat"], cfg["location"]["lon"], tz=cfg["location"]["tz"])
    rmse = lambda e: float(np.sqrt(np.mean(e**2)))
    df = f.merge(ah, left_on="step", right_index=True, how="inner").dropna(subset=["actual_mw"])
    df["elev"] = loc.get_solarposition(pd.DatetimeIndex(df["step"]))["elevation"].values
    df = df[df.elev > cfg["daytime_elevation_deg"]].copy()
    clim = df.groupby([df.step.dt.month, df.step.dt.hour])["actual_mw"].transform("mean")

    print("=== forecast skill vs climatology, by lead ===")
    for lo, hi in [(0,36),(48,72),(96,120),(168,240),(240,360)]:
        s = df[(df.op_lead_h>lo)&(df.op_lead_h<=hi)]
        if len(s) < 200: continue
        fr, cr = rmse(s.actual_mw-s.fc_mw), rmse(s.actual_mw-clim[s.index])
        print(f"  {lo:3d}-{hi:3d}h: RMSE {fr:5.0f} MW | skill vs clim {(1-fr/cr)*100:+.0f}% | corr {np.corrcoef(s.fc_mw,s.actual_mw)[0,1]:.2f}")

    print("\n=== adaptive-correction skill (walk-forward eval), by lead ===")
    for lo, hi, name in [(0,36,"0-36h"),(168,360,"168-360h")]:
        d = df[(df.op_lead_h>lo)&(df.op_lead_h<=hi)].sort_values(["issued_at","step"]).reset_index(drop=True)
        d["resid"]=d.actual_mw-d.fc_mw; d["iday"]=d.issued_at.dt.floor("D"); d["hour"]=d.step.dt.hour
        piv=d.pivot_table(index=d.step.dt.floor("D"),columns="hour",values="resid",aggfunc="mean").sort_index()
        b=piv.rolling("60D",min_periods=3).mean().shift(1).ffill().stack().rename("b").reset_index()
        b.columns=["bday","hour","b"]; d=d.merge(b,left_on=["iday","hour"],right_on=["bday","hour"],how="left")
        d["corr"]=np.clip(d.fc_mw+d["b"].fillna(0.0),0,None)
        t=d.issued_at; cut=t.min()+(t.max()-t.min())*0.8; te=d[t>=cut]
        base,mod=rmse(te.actual_mw-te.fc_mw),rmse(te.actual_mw-te["corr"])
        print(f"  {name}: baseline RMSE {base:5.0f} -> {mod:5.0f} MW | skill {(1-mod/base)*100:+.2f}%")

if __name__ == "__main__":
    main()
