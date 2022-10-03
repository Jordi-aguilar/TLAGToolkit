import os
import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import pchip

import yaml

from BaselineRemoval import BaselineRemoval

from lmfit import models

"""
with open('./Y10444_ybco.yml', 'r') as file:
    prime_service = yaml.safe_load(file)

print(prime_service)
print(prime_service["paths"]["save_path"].format(period = "asdf", scanNo = "asdf"))

"""
def get_yml_content(yml_file):
    with open(yml_file, 'r') as file:
        config_yml = yaml.safe_load(file)

    return config_yml


class Peak_fitter:
    def __init__(self, config_dict):

        self.get_config_dict_variables(config_dict)

        self.define_parameters()

        self.read_logs()

        self.initialize_model()
        

    def get_config_dict_variables(self, config_dict):
        
        self.scan_folder = config_dict['sample']['scan_folder']
        self.scan_id = str(config_dict['sample']['scan_id']).zfill(3)
        self.facility = config_dict['sample']['facility']
        self.period = config_dict['sample']['period']

        self.peak_name = config_dict['peaks']['peak_name']
        self.index_start = config_dict['peaks']['index_start']
        self.index_end = config_dict['peaks']['index_end']
        self.step = config_dict['peaks']['step']
        self.models_defined = config_dict['peaks']['models']
        self.num_models = len(self.models_defined)

        self.path_scans = config_dict['paths']['path_scans'].format(
            period = self.period, 
            scan_folder = self.scan_folder,
            scan_id = self.scan_id
        )

        self.filepath_logs = config_dict['paths']['filepath_logs'].format(
            period = self.period, 
            scan_folder = self.scan_folder,
            scan_id = self.scan_id
        )

        self.save_path = config_dict['paths']['save_path'].format(
            period = self.period,
            scan_folder = self.scan_folder
        )

        
    def define_parameters(self):
        self.x_spacing = 100
        self.peak_interval = 0.15
        self.data_interval = 0.25

        self.logs_recorded = ["imgIndex", "temperature", "pressure", "time"]

    
    def read_logs(self):
        self.df_log = pd.read_csv(self.filepath_logs)

        self.log_columns = self.df_log.columns


    def fit_model(self, model, x_spaced, y_corrected, params, interval, plot_results = False):
        # TODO: End this function
        result = model.fit(data=y_corrected, x=x_spaced, params=params, max_nfev = 5000)
         
        return result


    def run_fitting(self):

        self.initialize_fitting_data()

        mse_integrations = []

        # Loop through all files in the folder path in numerical order
        for scan_index in range(self.index_start, self.index_end, self.step):
            scan_index_str = str(scan_index).zfill(4)

            # Read the file
            integration_file = self.find_integration_file(self.path_scans, scan_index)
            df_integration = self.read_integrations_file(os.path.join(self.path_scans, integration_file), self.facility)

            x = df_integration['2th_deg'].values
            y = df_integration['I'].values

            x_spaced = x * self.x_spacing
            # TODO: Add remove_background function
            # y_corrected = self.remove_background(y)

            y_corrected = y

            mse_models = [None for i in range(len(self.models))]
            result_params_models = None

            for i, (model, params, interval) in enumerate(zip(self.models, self.params, self.intervals)):

                fitted_model = self.fit_model(model, x_spaced, y_corrected, params, interval, plot_results = False)

                mse_models[i] = np.mean(np.power(fitted_model.residual, 2))

                # Save current parameters for next fitting
                # TODO: Check why are we using this
                current_params = fitted_model.model.make_params()
                self.params[i] = current_params

                # Save current parameters to create a dataframe later
                if result_params_models is None:
                    result_params_models = fitted_model.params
                else:
                    result_params_models += fitted_model.params

            
            # Add current mse to mse list
            mse_integrations.append(mse_models)

            # Add file number and logs to the dictionary
            for log in self.logs_recorded:
                # TODO: improve syntax?
                self.fitting_data[log].append(
                    self.df_log.loc[self.df_log['imgIndex'] == scan_index, log].iloc[0]
                    )

            # Given that we are multiplying 'x' axes by x_spacing, we have to correct the parameters.
            correction_factor = {
                'amplitude' : 1/self.x_spacing,
                'center' : 1/self.x_spacing,
                'sigma' : 1/self.x_spacing,
                'gamma' : 1/self.x_spacing,
                'fwhm' : 1/self.x_spacing,
                'height' : 1
            }

            # Add params of the model of this file to the dictionary
            for param_name, param in result_params_models.items():
                print(param_name)
                correction = correction_factor[param_name.split('_')[1]]
                self.fitting_data[param_name].append(param.value * correction)

            break

        self.save_results()

    def save_results(self):

        # creating the corresponding folder
        try:
            os.stat(self.save_path)
        except:
            os.makedirs(self.save_path)

        model_data_df = pd.DataFrame(self.fitting_data)
        print("hello", self.save_path)
        model_data_df.to_csv(
            os.path.join(
                self.save_path, 
                f"TEST_peak_fits_{self.scan_folder}_{self.peak_name}.csv"
            ),
            index = False
        )


    def initialize_fitting_data(self):
        self.fitting_data = {}

        # Add logs
        for log in self.logs_recorded:
            self.fitting_data[log] = []

        # Add all params
        for param_set in self.params:
            for param_name in param_set.keys():
                self.fitting_data[param_name] = []


    def initialize_model(self):

        # Define groups of models in case the peaks are close
        group2model = self.create_groups()

        self.models = []
        self.params = []
        self.intervals = []

        initial_integration = self.read_initial_integration(self.path_scans)

        # For each group, create the composite_model, the composite_params and the 2theta range
        for group, model_list in group2model.items():
            composite_model = None
            composite_params = None

            lowest_2theta = 10**10
            highest_2theta = 0
            for model_id in model_list:
                model_config = self.models_defined[model_id]
                model = getattr(models, model_config['model_type'])(prefix=model_config['prefix'] + '_')

                params = self.set_initial_params(model, initial_integration)

                if composite_model is None:
                    composite_model = model
                else:
                    composite_model += model

                if composite_params is None:
                    composite_params = params
                else:
                    composite_params.update(params)
                    
                # Define 2theta range
                lowest_2theta = min(lowest_2theta, model_config["2thlimits"]["min"])
                highest_2theta = max(highest_2theta, model_config["2thlimits"]["max"])


            lowest_index = self.findClosest(initial_integration["2th_deg"], lowest_2theta)
            highest_index = self.findClosest(initial_integration["2th_deg"], highest_2theta)
            index_interval = (lowest_index, highest_index)

            # Add everything into the vectors   
            self.models.append(composite_model)
            self.params.append(composite_params)
            self.intervals.append(index_interval)
    

    def create_groups(self):
        groups = [-1 for _ in range(len(self.models_defined))] # Value in position 'i' is the group of model 'i'
        next_group = 0

        # First we have to decide how are we going to group the models
        for i in range(self.num_models):
            if groups[i] == -1:
                groups[i] = next_group
                next_group += 1
            
            current_group_id = groups[i]

            for j in range(i+1, self.num_models):
                min_dist = min(
                    abs(self.models_defined[i]['2thlimits']['min'] - self.models_defined[j]['2thlimits']['min']),
                    abs(self.models_defined[i]['2thlimits']['min'] - self.models_defined[j]['2thlimits']['max']),
                    abs(self.models_defined[i]['2thlimits']['max'] - self.models_defined[j]['2thlimits']['min']),
                    abs(self.models_defined[i]['2thlimits']['max'] - self.models_defined[j]['2thlimits']['max'])
                )

                if min_dist < 0.2:
                    groups[j] = current_group_id

        group2model = {}

        for model, group in enumerate(groups):
            if group in group2model.keys():
                group2model[group].append(model)
            else:
                group2model[group] = [model]

        return group2model


    def find_integration_file(self, path_scans, target_index):
        list_files_integrations = os.listdir(path_scans)

        target_file = None

        for file in list_files_integrations:
            index = self.get_index_integration_file(file)
            if index == target_index:
                target_file = file
                break

        return target_file
    

    def read_initial_integration(self, path_scans):

        initial_integration_file = self.find_integration_file(path_scans, self.index_start)

        df_initial_integration = self.read_integrations_file(os.path.join(self.path_scans, initial_integration_file), self.facility)

        return df_initial_integration

    def read_integrations_file(self, filepath, facility = "ALBA"):
        if facility == "ALBA":
            return self.read_integrations_file_alba(filepath)
        else:
            return self.read_integrations_file_soleil(filepath)
            

    @staticmethod
    def read_integrations_file_alba(filepath):
        with open(filepath) as f:
            line = f.readline()
            cnt = 0
            while line.startswith('#'):
                prev_line = line
                line = f.readline()
                cnt += 1
                # print(prev_line)

        header = prev_line.strip().lstrip('# ').split()

        df = pd.read_csv(filepath, delimiter="\s+",
                        names=header,
                        skiprows=cnt
                    )

        return df


    @staticmethod
    def read_integrations_file_soleil(filepath):
        raise "Not implemented!"

    def get_index_integration_file(self, file):
        if self.facility == "ALBA":
            return int(file.split(".")[0].split("_")[-1])
            
        else:
            return int(file.split("_")[1])


    def set_initial_params(self, model, initial_integration):
        default_params = {

        }

        params = model.make_params(**default_params)

        return params

    @staticmethod
    # Returns element closest to target in arr[]
    def findClosest(arr, target):
        n = len(arr)

        # Corner cases
        if (target <= arr[0]):
            return 0
        if (target >= arr[n - 1]):
            return n-1

        # Doing binary search
        i = 0; j = n; mid = 0
        while (i < j): 
            mid = (i + j) // 2

            if (arr[mid] == target):
                return arr[mid]

            # If target is less than array 
            # element, then search in left
            if (target < arr[mid]) :

                # If target is greater than previous
                # to mid, return closest of two
                if (mid > 0 and target > arr[mid - 1]):
                    # return getClosest(arr[mid - 1], arr[mid], target)
                    return mid - 1

                # Repeat for left half 
                j = mid
            
            # If target is greater than mid
            else :
                if (mid < n - 1 and target < arr[mid + 1]):
                    # return getClosest(arr[mid], arr[mid + 1], target)
                    return mid + 1
                    
                # update i
                i = mid + 1
            
        # Only single element left after search
        # return arr[mid]
        return mid


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('yml_file', metavar='YML', type=str, nargs=1,
                    help='Path of the yaml file')

    args = parser.parse_args()
    yml_file = args.yml_file[0]

    config_dict = get_yml_content(yml_file)

    peak_fitter = Peak_fitter(config_dict)

    peak_fitter.run_fitting()
