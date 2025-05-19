import xarray as xr
from watermodelCLASS import WaterModel
from itertools import product
from scipy.stats import pearsonr
import numpy as np
import random

class Calibrator:
    def __init__(self, paramsChoice, rasterData, amountCombinations = 3):
        "Rasterdata needs the layers Precipitation, Radiation, Temperature, NDVI and observedRunoff"
        self.paramsChoice = paramsChoice
        self.rasterData = rasterData
        self.amountCombinations = amountCombinations

    def calibration(self):
        amount_combinations = 3
        all_combinations = list(product(*self.paramsChoice.values()))
        random_combinations = random.sample(all_combinations, amount_combinations)
        
        best_r = -np.inf
        best_params = None
        
        for params in random_combinations:
            wm = WaterModel(params=params, data=self.rasterData)
            results = wm.run_simulation()
            runoff = results["runoff"].values.flatten()
            observed = self.rasterData.values.flatten()

            mask = ~np.isnan(runoff) & ~np.isnan(observed)
            if np.sum(mask) < 2:
                continue
            
            r, _ = pearsonr(runoff[mask], observed[mask])
            
            if r > best_r:
                best_r = r
                best_params = params

        return best_r, best_params 
    
    def calibrate_pixel(self, observed, prec, rad, temp, ndvi, paramsChoice):
        all_combinations = list(product(*paramsChoice.values()))
        random_combinations = random.sample(all_combinations, self.amountCombinations)
        
        best_r = -np.inf
        best_params = [np.nan, np.nan, np.nan, np.nan, np.nan, (np.nan, np.nan)]
        
        for params in random_combinations:
            wm = WaterModel(params=params, data=self.rasterData)
            runoff, _, _, _ = wm.time_evolution(temp, rad, prec, ndvi, params)
            mask = ~np.isnan(runoff) & ~np.isnan(observed)
            if np.sum(mask) < 2:
                continue 
            
            r, _ = pearsonr(runoff[mask], observed[mask])
            
            if r > best_r:
                best_r = r
                best_params = params
        c_s, alpha, gamma, beta, c_m, (et1, et2) = best_params
        return float(c_s), float(alpha), float(gamma), float(beta), float(c_m), float(et1), float(et2)


    def calculate_best_params(self):
        param_names = list(self.paramsChoice.keys())
        c_s, alpha, gamma, beta, c_m, et1, et2 = xr.apply_ufunc(
                self.calibrate_pixel,
                self.rasterData["observedRunoff"],
                self.rasterData["precipitation"],
                self.rasterData["radiation"],
                self.rasterData["temperature"],
                self.rasterData["ndvi"],
                kwargs={"paramsChoice": self.paramsChoice},
                input_core_dims=[["time"]]*5,
                output_core_dims=[[], [], [], [], [], [], []],
                vectorize=True,
                dask="allowed",  # falls du mit Dask arbeitest
                output_dtypes=[float]*7,
                output_sizes={"param": len(param_names)}
        )

        res = xr.Dataset({
            'c_s': c_s,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'c_m': c_m,
            'et1': et1,
            'et2': et2
        })

        return res
    