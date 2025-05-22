import xarray as xr
from watermodelCLASS import WaterModel
from itertools import product
from scipy.stats import pearsonr
import numpy as np
import random
from joblib import Parallel, delayed

class Calibrator:
    def __init__(self, initParams, rasterData, areaSize, numCombi = 3, numIter = 3):
        self.params = initParams
        self.areaSize = areaSize
        self.rasterData = self.calculate_catchment_mean(rasterData)
        self.numCombi = numCombi
        self.numIter = numIter        

    def calculate_catchment_mean(self, data):
        dataMean = xr.Dataset()
        dataMean['temperature'] = data["temperature"].mean(dim=("x", "y"))
        dataMean['precipitation'] = data["precipitation"].mean(dim=("x", "y"))
        dataMean['radiation'] = data["radiation"].mean(dim=("x", "y"))
        dataMean['ndvi'] = data["ndvi"].mean(dim=("x", "y"))
        dataMean['observedRunoff'] = self.normalize_observedRunoff(data["observedRunoff"].sum(dim=("x", "y")), self.areaSize)
        return dataMean
    
    def normalize_observedRunoff(self, observedRunoff, areaSize):
        return observedRunoff*86400/(areaSize*1000)

    def create_paramsChoice(self, params, i):
        paramsChoice = {key: [
            value-value/(2**i),
            value, 
            value+value/(2**i)]
                for key, value in params.items()}
        return paramsChoice

    def split_data(self, data, splitPerc):
        leng = data.sizes["time"]
        maxYear = leng/365
        sample = np.random.randint(0, maxYear, int(maxYear*splitPerc))
        mask = np.zeros(leng, dtype=bool)
        for s in sample:
            mask[s*365:(s+1)*365] = True
        train = data.isel(time=np.where(~mask)[0])
        test = data.isel(time=np.where(mask)[0])

        return train, test

    def calibrate_pixel(self, paramsChoice, valTrain):
        allCombinations = list(product(*paramsChoice.values()))
        randomCombinationsSample = random.sample(allCombinations, self.numCombi)
        param_dicts = [
            dict(zip(paramsChoice.keys(), combo)) for combo in randomCombinationsSample
        ]
        train, val = self.split_data(valTrain, 0.9)

        observed = train["observedRunoff"].values
        nan_mask_obs = ~np.isnan(observed)        

        def evaluate(params):
            wm = WaterModel(params=params, data=train)
            runoff = wm.run_simulation_whole_catchment()
            mask = nan_mask_obs & ~np.isnan(runoff)

            if np.sum(mask) < 2:
                return params, -np.inf

            r, _ = pearsonr(runoff[mask], observed[mask])
            return params, r
        
        results = Parallel(n_jobs=-1)(delayed(evaluate)(params) for params in param_dicts)
        best_params, best_r = max(results, key=lambda x: x[1])

        # Calculate R2 for Validation Timeseries
        wm = WaterModel(params=best_params, data=val)
        runoff = wm.run_simulation_whole_catchment()
        observed = val["observedRunoff"].values

        mask = ~np.isnan(observed)  & ~np.isnan(runoff)
        if np.sum(mask) < 2:
            r_val = np.nan

        r_val, _ = pearsonr(runoff[mask], observed[mask])
        r_train = best_r
        return best_params, r_train, r_val
    
    def calculate_params_whole_catchment(self):
        valTrain, test = self.split_data(self.rasterData, 0.9)
        lParams = []
        lRVal = []
        lRTest = []
        lRTrain = []
        for i in range(1,self.numIter+1):
            paramsChoice = self.create_paramsChoice(self.params, i)
            newParams, r_train, r_val = self.calibrate_pixel(paramsChoice, valTrain)
            self.params = newParams
            lParams.append(newParams)
            lRVal.append(r_val)
    
            idxBest = np.argmax(lRVal)
            wm = WaterModel(params=lParams[i-1], data=test)
            runoff = wm.run_simulation_whole_catchment()
            observed = test["observedRunoff"].values
            mask = ~np.isnan(observed)  & ~np.isnan(runoff)

            if np.sum(mask) < 2:
                rTest = np.nan

            rTest, _ = pearsonr(runoff[mask], observed[mask])

            lRTest.append(rTest)
            lRTrain.append(r_train)
       
        newParams["R2"] = rTest
        return newParams
    