import numpy as np
from numba import njit

@njit
def time_evolution_numba(temp, rad, prec, ndvi, c_s, alpha, beta, gamma, c_m, iota, theta, temp_w, ndvi_w):
    #Initialize 
    length = len(temp)
    runoff_out = np.full(length, np.nan)
    evapo_out = np.full(length, np.nan)
    soil_mois_out = np.full(length, np.nan)
    snow_out = np.full(length, np.nan)

    # Transformations / Calculations for Setup
    conv = 1 / 2260000  # from J/day/m**2 to mm/day
    rad = rad * conv  # convert radiation to mm/day
    prec = prec * 10 **3 # from m/day to mm/day
    w = 0.9 * c_s
    snow = 0

    # --- calc_et_weight function ---
    ndvi = np.nan_to_num(ndvi, nan=0.0)
    normalized_temp = (temp - temp.min()) / (temp.max() - temp.min())
    normalized_ndvi = (ndvi - ndvi.min()) / (ndvi.max() - ndvi.min())
    et_weight = temp_w * normalized_temp + ndvi_w * normalized_ndvi
    beta_weighted = beta * et_weight

    for t in range(1, length):
        prec_t = prec[t-1]
        temp_t = temp[t-1]
        rad_t = rad[t-1]
        beta_t = beta_weighted[t-1]

        # ---- snow_function ----
        is_melting = temp_t > 273.15
        has_snow = snow >= 0.001

        if not is_melting:
            snow = snow + prec_t
            water = 0.0
        elif is_melting and has_snow:
            melt = c_m * (temp_t - 273.15)
            melt = min(melt, snow)
            snow = snow - melt
            water = melt + prec_t
        else:
            water = prec_t

        baseflow = iota * (w / c_s) ** theta
        runoff = water * (w / c_s) ** alpha + baseflow
        evap = beta_t * (w / c_s) ** gamma * rad_t

        w = w + (water - runoff - evap)
        w = np.maximum(w, 0.0)

        # Store results
        runoff_out[t] = runoff
        evapo_out[t] = evap
        soil_mois_out[t] = w
        snow_out[t] = snow

    return runoff_out, evapo_out, soil_mois_out, snow_out

class WaterModel:
    def __init__(self, params: dict, data):
        self.params = params
        self.data = data

    def run_simulation_whole_catchment(self):
        runoff,_,_,_= time_evolution_numba(
            self.data['temperature'].values,
            self.data['radiation'].values,
            self.data['precipitation'].values,
            self.data['ndvi'].values,
            self.params['c_s'],
            self.params['alpha'],
            self.params['beta'],
            self.params['gamma'],
            self.params['c_m'],
            self.params['iota'],
            self.params['theta'],
            self.params['temp_w'],
            self.params['ndvi_w']
        )
        return runoff