import xarray as xr
import numpy as np
import os
import matplotlib.pyplot as plt

class WaterModel:
    def __init__(self, params, data):
        self.params = params
        self.data = data

    def calc_et_weight(self, temp, ndvi, w):
        """Calculate influence of LAI and temperature on ET.
            Input: temp: temperature data [K]
            lai: leaf area index data [m**2/m**2]
            w: weights for temperature and lai"""
        # Get coefficients for temperature and lai
        temp_w, lai_w = w
        ndvi = np.nan_to_num(ndvi, nan=0)
        temp_min = temp.min()
        temp_max = temp.max()
        lai_min = ndvi.min()
        lai_max = ndvi.max()

        # Perform normalization
        normalized_temp = (temp - temp_min) / (temp_max - temp_min)
        normalized_lai = (ndvi - lai_min) / (lai_max - lai_min)

        # Weight Temperature and LAI
        et_coef = temp_w * normalized_temp + lai_w * normalized_lai
        return et_coef

    def water_balance(self, w_t, prec_t, rad_t, snow_t, temp_t, cs, alpha, beta, gamma, c_m):
        """ Calculates the water balance for one time step as introduced in the lecture. Added features, such as snow"""
        snow_t, prec_t = self.snow_function(snow_t, prec_t, temp_t,
                                c_m)  # overwrites the precipitation (if snow melts or precipitation is accumulated as snow)
        runoff_t = self.runoff(w_t, prec_t, cs, alpha)
        evapo_t = self.evapotranspiration(w_t, rad_t, cs, beta, gamma)
        w_next = w_t + (prec_t - evapo_t - runoff_t)
        w_next = np.maximum(0, w_next)
        
        return runoff_t, evapo_t, w_next, snow_t

    def runoff(self, w_t, prec_t, cs, alpha):
        return prec_t * (w_t / cs) ** alpha

    def evapotranspiration(self, w_t, rad_t, cs, beta, gamma):
        return beta * (w_t / cs) ** gamma * rad_t

    def snow_function(self, snow_t, prec_t, temp_t, c_m):
        # Determine if temperature is above freezing (melting condition)
        is_melting = temp_t > 273.15
        
        # Determine if there is already snow on the ground
        has_snow = snow_t >= 0.001

        if not is_melting:
            # Temperature is below or at freezing → precipitation adds to snow
            snow_out = snow_t + prec_t
            water_out = 0.0
        elif is_melting and has_snow:
            # Snow is present and temperature is above freezing → melt snow
            SnowMelt = c_m * (temp_t - 273.15)
            snow_out = snow_t - SnowMelt
            if snow_out < 0:
                SnowMelt = snow_t  # Can't melt more than exists
                snow_out = 0.0
            water_out = SnowMelt + prec_t
        else:
            # No snow, and temperature above freezing → all precip is rain
            snow_out = snow_t
            water_out = prec_t

        return snow_out, water_out

    def time_evolution(self, temp, rad, prec, ndvi, params):
        runoff_out = np.full_like(temp, np.nan)
        evapo_out = np.full_like(temp, np.nan)
        soil_mois_out = np.full_like(temp, np.nan)
        snow_out = np.full_like(temp, np.nan)
        
        if np.all(np.isnan(ndvi)):
            #edge case for no vegetation
            return runoff_out, evapo_out, soil_mois_out, snow_out
        cs, alpha, gamma, beta, c_m, et_weight = params
        #conversion factor

        conv = 1 / 2260000  # from J/day/m**2 to mm/day
        rad = rad * conv  # convert radiation to mm/day
        prec = prec * 10 **3 # from m/day to mm/day
        
        w_0 = 0.9 * cs
        snow_0 = 0
        
        beta_weighted = beta * self.calc_et_weight(temp, ndvi, et_weight)
        
        for t in range(1, len(temp)):
            prec_t = prec[t-1]
            temp_t = temp[t-1]
            rad_t = rad[t-1]
            beta_weighted_t = beta_weighted[t-1]
            runoff_out[t], evapo_out[t], soil_mois_out[t], snow_out[t] = self.water_balance(
                w_0, prec_t, rad_t, snow_0, temp_t, cs, alpha, beta_weighted_t, gamma, c_m)
            w_0 = soil_mois_out[t]
            snow_0 = snow_out[t]
            
        return runoff_out, evapo_out, soil_mois_out, snow_out
    
    def run_simulation(self):
        runoff, evapo, soil_mois, snow = xr.apply_ufunc(
            self.time_evolution,
            self.data['temperature'],
            self.data['radiation'],
            self.data['precipitation'],
            self.data['ndvi'],
            kwargs={'params': self.params},
            input_core_dims=[['time'], ['time'], ['time'], ['time']],
            output_core_dims=[['time'], ['time'], ['time'], ['time']],
            vectorize=True,
            dask='allowed',
            output_dtypes=[np.float64, np.float64, np.float64, np.float64])

        results = xr.Dataset({
                'runoff': runoff,
                'evapotranspiration': evapo,
                'soil_moisture': soil_mois,
                'snow': snow
            })
        return results
        
    def run_simulation_whole_catchment(self):
        runoff,_,_,_= self.time_evolution(temp = self.data['temperature'],
                                          rad = self.data['radiation'],
                                          prec = self.data['precipitation'],
                                          ndvi = self.data['ndvi'],
                                          params = self.params)
        return xr.Dataset({'runoff': runoff})
        
        