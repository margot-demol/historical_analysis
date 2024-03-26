import xarray as xr
import numpy as np

import histlib.cstes as cstes
import histlib.box as box

"""
FUNCTIONS STRESS -> WIND TERM 
---------------------------------------------------------------------------------------------------------

EKMAN RIO 2014
---------------------------------
return wind term 
"""

def cst_rio_z0(taue, taun, f, theta_lon, theta_lat, rot=True):
    
    theta0 = - 30.75 * np.pi / 180 * np.sign(f)
    beta0 = 0.61
    
    uek_e = beta0 * (np.cos(theta0) * taue - np.sin(theta0) * taun)
    uek_n = beta0 * (np.sin(theta0) * taue + np.cos(theta0) * taun)

    if rot:
        uek_x, uek_y = box.vevn2vxvy(theta_lon, theta_lat, uek_e, uek_n)
        return -f * uek_y, f * uek_x
    else:
        return -f * uek_n, f * uek_e


def cst_rio_z15(taue, taun, f, theta_lon, theta_lat, rot=True):
    theta15 = - 48.18 * np.pi / 180 * np.sign(f)
    beta15 = 0.25
    
    uek_e = beta15 * (np.cos(theta15) * taue - np.sin(theta15) * taun)
    uek_n = beta15 * (np.sin(theta15) * taue + np.cos(theta15) * taun)

    if rot:
        uek_x, uek_y = box.vevn2vxvy(theta_lon, theta_lat, uek_e, uek_n)
        return f * uek_y, -f * uek_x
    else:
        return f * uek_n, -f * uek_e

"""
WIND TERM DATASET 
---------------------------------------------------------------------------------------------------------
"""

list_wd_srce_suffix = ["es", "e5"]
list_func = [cst_rio_z0, cst_rio_z15]
list_func_suffix = ["cstrio_z0", "cstrio_z15"]


def compute_wd_from_stress(
    ds,
    list_wd_srce_suffix=list_wd_srce_suffix,
    list_func=list_func,
    list_func_suffix=list_func_suffix,
    east_north=False,
):
    _ds = xr.Dataset()
    for i in range(len(list_func)):
        func, suf = list_func[i], list_func_suffix[i]
        for src in list_wd_srce_suffix:
            alti_matchup_theta_lon, alti_matchup_theta_lat = ds["box_theta_lon"].sel(
                box_x=0, box_y=0
            ), ds["box_theta_lat"].sel(box_x=0, box_y=0)

            try:
                idx = ds.__site_matchup_indice.astype(int).compute()

                drifter_theta_lon = ds["drifter_theta_lon"].sel(
                    site_obs=idx)
            
                drifter_theta_lat = ds["drifter_theta_lat"].sel(
                    site_obs=idx)

            except:
                drifter_theta_lon = ds["drifter_theta_lon"]
                drifter_theta_lat = ds["drifter_theta_lat"]

            # ALTI AND DRIFTER MATCHUP
            (
                _ds[src + "_" + suf + "_alti_wd_x"],
                _ds[src + "_" + suf + "_alti_wd_y"],
            ) = func(
                ds[src + "_alti_matchup_taue"],
                ds[src + "_alti_matchup_taun"],
                ds["f"],
                alti_matchup_theta_lon,
                alti_matchup_theta_lat,
                rot=True,
            )
            (
                _ds[src + "_" + suf + "_drifter_wd_x"],
                _ds[src + "_" + suf + "_drifter_wd_y"],
            ) = func(
                ds[src + "_drifter_matchup_taue"],
                ds[src + "_drifter_matchup_taun"],
                ds["f"],
                drifter_theta_lon,
                drifter_theta_lat,
                rot=True,
            )

            # attrs ADD HERE IF MORE FUNCTIONS TO COMPUTE WIND FROM STRESS
            if src == "es":
                source = "erastar "
            if src == "e5":
                source = "era5 "
            if suf == "cstrio_z0":
                fonction = r"computed with the method described in Rio 2014 for constant $\theta$ and $\beta$ parameters at z=0m"
            if suf == "cstrio_z15":
                fonction = r"computed with the method described in Rio 2014 for constant $\theta$ and $\beta$ parameters at z=15m"

            _ds[src + "_" + suf + "_alti_wd_x"].attrs = {
                "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_x)_{altimatchup}$ from "
                + suf,
                "description": "along track wind related term interpolated at the altimeter's matchup from "
                + source
                + fonction,
                "units": r"$m.s^{-2}$",
            }
            _ds[src + "_" + suf + "_alti_wd_y"].attrs = {
                "long_name": r" $(-\frac{1}{\rho}\partial_z\tau_y)_{altimatchup}$ from "
                + suf,
                "description": "cross track wind related term interpolated at the altimeter's matchup from "
                + source
                + fonction,
                "units": r"$m.s^{-2}$",
            }
            _ds[src + "_" + suf + "_drifter_wd_x"].attrs = {
                "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_x)_{driftermatchup}$ from "
                + suf,
                "description": "along track wind related term interpolated at the drifter's matchup from "
                + source
                + fonction,
                "units": r"$m.s^{-2}$",
            }
            _ds[src + "_" + suf + "_drifter_wd_y"].attrs = {
                "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_y)_{driftermatchup}$ from "
                + suf,
                "description": "cross track wind related term interpolated at the drifter's matchup from "
                + source
                + fonction,
                "units": r"$m.s^{-2}$",
            }
            # TRAJ
            if src + "_traj_taue" in ds:
                (
                    _ds[src + "_" + suf + "_traj_wd_x"],
                    _ds[src + "_" + suf + "_traj_wd_y"],
                ) = func(
                    ds[src + "_traj_taue"],
                    ds[src + "_traj_taun"],
                    ds["f"],
                    ds["drifter_theta_lon"],
                    ds["drifter_theta_lat"],
                    rot=True,
                )
                # attrs
                _ds[src + "_" + suf + "_traj_wd_x"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_x)_{traj}$ from "
                    + suf,
                    "description": "along track wind related term interpolated on the drifter's trajectory from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }
                _ds[src + "_" + suf + "_traj_wd_y"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_y)_{traj}$ from "
                    + suf,
                    "description": "cross track wind related term interpolated on the drifter's trajectory from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }

            if src + "_box_taue" in ds:
                (
                    _ds[src + "_" + suf + "_box_wd_x"],
                    _ds[src + "_" + suf + "_box_wd_y"],
                ) = func(
                    ds[src + "_box_taue"],
                    ds[src + "_box_taun"],
                    ds["f"],
                    ds["box_theta_lon"],
                    ds["box_theta_lat"],
                    rot=True,
                )
                # attrs
                _ds[src + "_" + suf + "_box_wd_x"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_x)_{box}$ from "
                    + suf,
                    "description": "along track wind related term interpolated on the box from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }
                _ds[src + "_" + suf + "_box_wd_y"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_y)_{box}$ from "
                    + suf,
                    "description": "cross track wind related term interpolated on the box from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }

            if east_north:
                (
                    _ds[src + "_" + suf + "_alti_wd_e"],
                    _ds[src + "_" + suf + "_alti_wd_n"],
                ) = func(
                    ds[src + "_alti_matchup_taue"],
                    ds[src + "_alti_matchup_taun"],
                    ds["f"],
                    alti_matchup_theta_lon,
                    alti_matchup_theta_lat,
                    rot=False,
                )

                (
                    _ds[src + "_" + suf + "_drifter_wd_e"],
                    _ds[src + "_" + suf + "_drifter_wd_n"],
                ) = func(
                    ds[src + "_drifter_matchup_taue"],
                    ds[src + "_drifter_matchup_taun"],
                    ds["f"],
                    drifter_theta_lon,
                    drifter_theta_lat,
                    rot=False,
                )

                # attrs
                _ds[src + "_" + suf + "_alti_wd_e"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_e)_{altimatchup}$ from "
                    + suf,
                    "description": "eastward wind related term interpolated at the altimeter's matchup from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }
                _ds[src + "_" + suf + "_alti_wd_n"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_n)_{altimatchup}$ from "
                    + suf,
                    "description": "northward wind related term interpolated at the altimeter's matchup from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }
                _ds[src + "_" + suf + "_drifter_wd_e"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_e)_{driftermatchup}$ from "
                    + suf,
                    "description": "eastward wind related term interpolated at the drifter's matchup from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }
                _ds[src + "_" + suf + "_drifter_wd_n"].attrs = {
                    "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_n)_{driftermatchup}$ from "
                    + suf,
                    "description": "northward wind related term interpolated at the drifter's matchup from "
                    + source
                    + fonction,
                    "units": r"$m.s^{-2}$",
                }
                if src + "_traj_taue" in ds:
                    (
                        _ds[src + "_" + suf + "_traj_wd_e"],
                        _ds[src + "_" + suf + "_traj_wd_n"],
                    ) = func(
                        ds[src + "_traj_taue"],
                        ds[src + "_traj_taun"],
                        ds["f"],
                        ds["drifter_theta_lon"],
                        ds["drifter_theta_lat"],
                        rot=False,
                    )
                    # attrs
                    _ds[src + "_" + suf + "_traj_wd_e"].attrs = {
                        "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_e)_{traj}$ from "
                        + suf,
                        "description": "eastward wind related term interpolated on the drifter's trajectory from "
                        + source
                        + fonction,
                        "units": r"$m.s^{-2}$",
                    }
                    _ds[src + "_" + suf + "_traj_wd_n"].attrs = {
                        "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_n)_{traj}$ from "
                        + suf,
                        "description": "northward wind related term interpolated on the drifter's trajectory from "
                        + source
                        + fonction,
                        "units": r"$m.s^{-2}$",
                    }
                if src + "_box_taue" in ds:
                    (
                        _ds[src + "_" + suf + "_box_wd_e"],
                        _ds[src + "_" + suf + "_box_wd_n"],
                    ) = func(
                        ds[src + "_box_taue"],
                        ds[src + "_box_taun"],
                        ds["f"],
                        ds["box_theta_lon"],
                        ds["box_theta_lat"],
                        rot=False,
                    )
                    # attrs
                    _ds[src + "_" + suf + "_box_wd_e"].attrs = {
                        "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_e)_{box}$ from "
                        + suf,
                        "description": "eastward wind related term interpolated on the box from "
                        + source
                        + fonction,
                        "units": r"$m.s^{-2}$",
                    }
                    _ds[src + "_" + suf + "_box_wd_n"].attrs = {
                        "long_name": r"$(-\frac{1}{\rho}\partial_z\tau_n)_{box}$ from "
                        + suf,
                        "description": "northward wind related term interpolated on the box from "
                        + source
                        + fonction,
                        "units": r"$m.s^{-2}$",
                    }

            _ds = _ds.drop([v for v in list(_ds.coords) if v not in ["box_x", "box_y"]])

    return _ds
