# -*- coding: utf-8 -*-
"""
ANALYSING ALBA'S DATA IN-SITU TLAG GROWTH'

Created on Wed Mar 13 11:03:30 2024

@author: omola
"""


import tkinter as tk
from tkinter import filedialog
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.interpolate import pchip
import openpyxl
import colorcet
import os
import csv
#from simplified_peak_fitting import fit_peaks



# FUNCTION DEFINTIONS

# allows multiple selection of csv files
def upload_multiple_csv_files():
    root = tk.Tk()
    root.withdraw()
    csv_files = filedialog.askopenfilenames(                     # peak_fitting_files is a tuple containing the paths to the selected files
                    #initialdir = directory = 'C:/Users/onamo/Documents/05 Doctorat/ALBA python/ALBA March24/DATA PROCESSED/',
                    title = 'Select the fitting for all peaks that you want to include on the analysis',
                    filetypes = (('csv files', '*.csv'),),
                    multiple = True 
                    )
    
    if not csv_files:
        raise ValueError("No files selected. Please select at least one CSV file.")
    
    return csv_files


def upload_csv_files():
    peak_fitting_files = upload_multiple_csv_files()

    peak_files = []
    final_file = []
    mass_spectr_file = []
    for file in peak_fitting_files:
        basename = os.path.basename(file)
        if 'final' in basename:
            final_file = file
        elif 'S1_' in basename:
            mass_spectr_file = file
        else:
            peak_files.append(file)

    directory = os.path.dirname(peak_files[0])

    return peak_files, final_file, mass_spectr_file, directory



def get_name(header):
    ''' finds the name of the peak or peaks from the header as the string before '_2th_max' '''

    remove_string = "_2th_max"
    name = []
    for element in header:
        index_imax = element.find(remove_string)
        if index_imax != -1:
            name.append(element[:-len(remove_string)])

    if len(name) == 1:          # if only one peak, return it as a string instead of a list with one element
        name = name[0]

    elif not name:
        raise ValueError("The structure of the headers is not as expected, should have '2th_max' in it.")

    return name



def peak_dict(file_path):
    """
    Extracts all peak information from a file and keeps it on a dictionary. if multiple peaks, keep all info together
    """

    peak_parameters = {}
    with open(file_path, newline='') as csvfile:          # open csv file and keep the information of the peak in a dictionary

        header = csvfile.readline().strip().split(',')    # expected to be: imgIndex, temperature, pressure, time, timestamp, + peak parameters
        num_columns = len(header)
        csvdtype = [('imgIndex', 'U4')] + [('col{}'.format(i), float) for i in range(num_columns - 1)]

        data = np.genfromtxt(file_path, delimiter=',', skip_header=1, dtype=csvdtype, filling_values=np.nan, unpack=True)

        # entrances of the dictionary
        peak_parameters = dict(zip(header, data))
        peak_parameters['imgIndex'] = [str(index).zfill(4) for index in peak_parameters['imgIndex']]
        peak_parameters['name'] = get_name(header)

    return peak_parameters



def extract_peak_info(peak_fitting_files):
    """
    If csv file contains more than one peak (multiple peaks on peak_dict), here we separate them for the dictionary 'peaks_data'
    """

    exp_data_desired = ['temperature', 'pressure', 'resistance', 'time', 'timestamp', 'omega', 'att1', 'att2']
    headers = list(peak_dict(peak_fitting_files[0]).keys())
    experiment_data = {header: [] for header in exp_data_desired if header in headers}

    peaks_data = {}
    peak_count = 0
    for i, file in enumerate(peak_fitting_files):

        new_peak_info = peak_dict(file)

        # keep only the experiment_data if we do not have this time range of the experiment
        t0 = new_peak_info['time'][0]
        if t0 not in experiment_data['time']:
            for key in experiment_data:
                if key in new_peak_info:
                    experiment_data[key].extend(new_peak_info[key])

        # check if there is more than one peak information in the csv file
        if isinstance(new_peak_info['name'], list):
            num_peaks_in_dict = len(new_peak_info['name'])
        else:
            num_peaks_in_dict = 1

        # if a file has more than one peak information, here we split this in order to have all peaks in different entrancies
        if num_peaks_in_dict > 1:
            headers_desired = ['imgIndex', 'temperature', 'pressure', 'resistance', 'time', 'timestamp', 'omega', 'att1', 'att2']

            for j, peak in enumerate(new_peak_info['name']):

                headers = [name for name in list(new_peak_info.keys())]
                shared_info_headers = [header for header in headers_desired if header in headers]

                peak_info = [name for name in list(new_peak_info.keys()) if peak in name]   # extract headers with the name of the peak in it
                header_j = shared_info_headers + peak_info
                peak_dict_j = {key: new_peak_info[key] for key in header_j}
                peak_dict_j['name'] = peak

                peaks_data[f'peak{peak_count}'] = peak_dict_j
                peak_count = peak_count + 1

        # only one peak on peak_dict
        else:
            peaks_data[f'peak{peak_count}'] = new_peak_info
            peak_count = peak_count + 1

    return peaks_data, experiment_data


def time_to_seconds(time_str):
    hours, minutes, seconds = time_str.split(':')
    total_seconds = int(hours)*3600 + int(minutes)*60 + float(seconds)
    return round(total_seconds,3)


def read_mass_spectrometer(mass_spectr_file):
    if mass_spectr_file:
        columns_to_keep = [1, 11]     # keep time and CO2 signal only

        with open(mass_spectr_file, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for _ in range(37):    # # Skip the first 37 lines
                next(csvfile)

            all_headers = csvfile.readline().strip().split(',')
            header = [all_headers[i] for i in columns_to_keep]
            mass_spectr_data = {'time': [], 'co2_values':[]}
            for row in reader:
                mass_spectr_data['time'].append(time_to_seconds(row[columns_to_keep[0]]))
                mass_spectr_data['co2_values'].append(float(row[columns_to_keep[1]]))
    else:
        mass_spectr_data = None

    return mass_spectr_data

# finds the repeated elements on a list and returns the repeated element and its positions
def find_duplicates_with_positions(my_list):
    duplicates = defaultdict(list)
    for i, item in enumerate(my_list):
        duplicates[item].append(i)
    return {item: positions for item, positions in duplicates.items() if len(positions) > 1}



# sort a list and returns both sorted elements and their original indices
def sort_list_keep_index(my_list):
    sorted_elements_with_indices = sorted(enumerate(my_list), key=lambda x: x[1])

    # Extract sorted elements and their original indices
    sorted_elements = [element for index, element in sorted_elements_with_indices]
    original_indices = [index for index, element in sorted_elements_with_indices]

    return sorted_elements, original_indices


def flatten(nested_list):
    flattened_list = [value for sublist in nested_list for value in sublist]
    return flattened_list



def concatenated_dict_data(peaks_data, repeated_peaks_sorted_index):
    keys_to_skip = ['name', 'model']

    first_peak_index = repeated_peaks_sorted_index[0]
    concatenated_peak = peaks_data[f'peak{first_peak_index}']

    for i in repeated_peaks_sorted_index[1:]:  
        peak_name = f'peak{i}'
        for key in peaks_data[peak_name].keys():
            if key not in keys_to_skip:
                concatenated_peak[key] = np.concatenate((concatenated_peak[key], peaks_data[peak_name][key]))

    return concatenated_peak


def unify_key_names(peaks):
    for peak in peaks.keys():
        replace_text = peaks[peak]['name'] + '_'
        peaks[peak] = {key.replace(replace_text, ''): value for key, value in peaks[peak].items()}
    return peaks


def join_same_peaks(peaks_data):
    
    list_peaks_name = []
    for i, peak in enumerate(peaks_data):
        list_peaks_name.append(peaks_data[peak]['name'])

    duplicate_peaks_with_positions = find_duplicates_with_positions(list_peaks_name)

    peaks = {}
    if duplicate_peaks_with_positions:

        # for each repeated peak, we order them by time and concatenate all information
        for peak in duplicate_peaks_with_positions.keys():  
            initial_time_list = []
            positions_peak_in_peaks_data = duplicate_peaks_with_positions[peak]
            for position in positions_peak_in_peaks_data:
                initial_time_list.append(peaks_data[f'peak{position}']['time'][0])
            
            initial_time_list_sorted, repeated_peaks_sorted_index = sort_list_keep_index(initial_time_list)
            index_peak = [positions_peak_in_peaks_data[index] for index in repeated_peaks_sorted_index]

            concatenated_peak = concatenated_dict_data(peaks_data, index_peak)
            peak_num = len(peaks)
            peaks[f'peak{peak_num}'] = concatenated_peak

        # copy all not repeated peaks information to the new dictionary such that the new dictionary has the information for all peaks without being repeated
        index_all_repeated_peaks = sorted(flatten(list(duplicate_peaks_with_positions.values())))
        missing_numbers = [num for num in range(len(peaks_data)) if num not in index_all_repeated_peaks]
        for i, peak in enumerate(missing_numbers):
            peak_num = len(peaks)
            peaks[f'peak{peak_num}'] = peaks_data[f'peak{peak}']

    else:
        peaks = peaks_data

    # change name of keys, so it will be easy to call them during plots (change dome_AUC to AUC)
    peaks = unify_key_names(peaks)

    return peaks


def concatenate_time_values(peaks_data):
    time_values = []
    for peak_key, peak_data in peaks_data.items():
        time_values.extend(peak_data.get('time', []))
    return time_values


def eliminate_data_before_jump(peaks, index_jump):
    for peak in peaks.keys():
        if peaks[peak]['name'] == 'YBCO005' or peaks[peak]['name'] == 'YBCO103':
            peaks[peak]['I_max'][0:index_jump] = 0

    return peaks



def add_normalised_intensity(peaks, time_cooling):
    """
    Add normalised intensity for each peak considering the heating and jump, not cooling
    """
    for peak in peaks.keys():
        index_cooling = find_index_of_closest_num(peaks[peak]['time'], time_cooling)
        max_intensity = max(peaks[peak]['I_max'][:index_cooling])
        peaks[peak]['I_max_norm'] = peaks[peak]['I_max'] / max_intensity
        peaks[peak]['I_max_norm_err'] = peaks[peak]['I_max_err'] / max_intensity

        max_AUC = max(peaks[peak]['AUC'][:index_cooling])
        peaks[peak]['AUC_norm'] = peaks[peak]['AUC'] / max_AUC
        peaks[peak]['AUC_norm_err'] = peaks[peak]['AUC_err'] / max_AUC

    return peaks



def reinicialise_time(peaks, time):
    t0 = min(time)
    time = time- t0
    for peak in peaks.keys():
        peaks[peak]['time'] = peaks[peak]['time'] - t0
    
    return peaks, time



def acquisition_time_compensation(peaks, additional_info):
    ''' 
    look at max intensity of the dome of the image when jump occurs and apply this value to all other images to compensate
    for different acqquisition times. We take the jump because at heating and cooling we may take a higher time step, but
    in this area plot we are interested in the jump
    '''

    time_jump = additional_info['time_jump']

    index_dome = find_index_peak(peaks, 'dome')
    index_jump = find_index_of_closest_num(peaks[index_dome]['time'], time_jump)
    intensity_dome_reference = peaks[index_dome]['I_max'][index_jump]
    time_acquisition_at_jump = peaks[index_dome]['time'][index_jump]- peaks[index_dome]['time'][index_jump-1]

    print(' ')
    print(f"Impose to all xrd images that I_max(dome) is {intensity_dome_reference:.4f}, which has a time acquisition of {time_acquisition_at_jump:.4f} seconds.")

    additional_info['I_max dome reference'] = intensity_dome_reference
    additional_info['time acquisition reference'] = time_acquisition_at_jump

    # create a dictionary that have at each time (key) the value of the correction to apply (value)
    correction_time_acquisition = []
    for i,intensity_dome in enumerate(peaks[index_dome]['I_max']):
        if intensity_dome == 0:
            imgIndex_value = peaks[index_dome]['imgIndex'][i]
            raise ValueError(f'Intensity of dome is 0 for imgIndex {imgIndex_value:.4f}')
        correction_time_acquisition.append((intensity_dome_reference/intensity_dome, peaks[index_dome]['time'][i]))
    value_dict = {time: value for value, time in correction_time_acquisition}

    for peak in peaks:
        peak_time = peaks[peak]['time']

        for j, time in enumerate(peak_time):
            if time in value_dict:
                peaks[peak]['I_max'][j] = peaks[peak]['I_max'][j] * value_dict[time]
                peaks[peak]['I_max_err'][j] = peaks[peak]['I_max_err'][j] * value_dict[time]
                peaks[peak]['AUC'][j] = peaks[peak]['AUC'][j] * value_dict[time]
                peaks[peak]['AUC_err'][j] = peaks[peak]['AUC_err'][j] * value_dict[time]

    return peaks, additional_info


def calculate_derivative(x_data, y_data):
    dx = np.diff(x_data)
    dy = np.diff(y_data)
    derivative = dy / dx
    return derivative


def find_pressure_jump(experiment_data, additional_info):
    time = experiment_data['time']
    pressure = experiment_data['pressure']

    pressure_derivative = calculate_derivative(time, pressure)
    index_jump = np.nanargmax(pressure_derivative)  
    time_jump = time[index_jump]
    pressure_jump = pressure[index_jump]

    additional_info['time_jump'] = time_jump
    additional_info['pressure_jump'] = pressure_jump

    return additional_info


def change_mass_spectr_data(mass_spectr_data, experiment_data, time_jump):
    ''' it adapts the data form mass spectrometer to look nice on the graph. it matches the time with the jump and it changes the y_data to match pressure y axis'''

    ms_min = min(mass_spectr_data['co2_values'][5:])
    mass_spectr_data['co2_values_plot'] = [(ms_value-ms_min) *9e10 for ms_value in mass_spectr_data['co2_values']]

    interpolation = pchip(mass_spectr_data['time'],mass_spectr_data['co2_values'])
    x_interpolation = np.arange(mass_spectr_data['time'][0], mass_spectr_data['time'][-1], 0.1)
    y_interpolation = interpolation(x_interpolation)

    ms_derivative = calculate_derivative(x_interpolation, y_interpolation)
    jump_ms_time = x_interpolation[np.argmin(ms_derivative)]
    index_jump_ms = find_index_of_closest_num(mass_spectr_data['time'], jump_ms_time)
    
    shift_time = mass_spectr_data['time'][index_jump_ms] - time_jump
    mass_spectr_data['time_exp'] = mass_spectr_data['time'] - shift_time

    plt.plot(mass_spectr_data['time'], mass_spectr_data['co2_values'], '-', color='blue', markersize=2, label='mass spectrometer')
    plt.plot(mass_spectr_data['time'][index_jump_ms], mass_spectr_data['co2_values'][index_jump_ms], 'o', color='red', label='start jump')
    plt.legend(loc='upper left')

    return mass_spectr_data


# given a target number, it finds the index of the closest number on a certain list. this list is meant to be ordered, since the algorithm is optimized for this case
def find_index_of_closest_num(list, target_num):
    low = 0
    high = len(list) - 1
    closest_index = None

    while low <= high:
        mid = (low + high) // 2
        if list[mid] == target_num:
            return mid
        elif list[mid] < target_num:
            low = mid + 1
        else:
            high = mid - 1

        if closest_index is None or abs(list[mid] - target_num) < abs(list[closest_index] - target_num):
            closest_index = mid

    return closest_index

# given a target number, it finds the index of the closest number on a certain list. this list is meant to be ordered and the numbers equidistat, since the 
# algorithm is optimized for this case
def find_index_of_closest_num_equidistant(list, target_num):
    diff = target_num - list[0]
    closest_index = int(diff // (list[1] - list[0]))

    if abs(list[closest_index] - target_num) > abs(list[closest_index + 1] - target_num):
        closest_index += 1

    return closest_index



def find_start_of_increase(data):
    for i in range(len(data) - 1, 0, -1):
        if data[i]  > data[i - 1]:
            return i
    return None  # If no increase is found


def find_start_cooling(experiment_data, additional_info):
    time = experiment_data['time']
    temperature = experiment_data['temperature']

    index_cooling = find_start_of_increase(temperature)
    time_cooling = time[index_cooling]
    temp_cooling = temperature[index_cooling]

    additional_info['time_cooling'] = time_cooling
    additional_info['temp_cooling'] = temp_cooling

    return additional_info



def save_peaks_information(peaks, additional_info, directory):
    path_save_file = directory + '\peaks_data.xlsx'
    with pd.ExcelWriter(path_save_file, engine='openpyxl') as writer:
        for peak, values in peaks.items():
            data = {key: values[key] for key in values if key != 'name'}
            df = pd.DataFrame(data)
            df.to_excel(writer, sheet_name=values['name'], index=False)

        data = {key: additional_info[key] for key in additional_info}
        df = pd.DataFrame([data])
        df.to_excel(writer, sheet_name='additional info', index=False)



def plot_cooling_and_pressure_jump(experiment_data, additional_info):
    time = experiment_data['time']
    temperature = experiment_data['temperature']
    pressure = experiment_data['pressure']
    time_cooling = additional_info['time_cooling']
    temp_cooling = additional_info['temp_cooling']
    time_jump = additional_info['time_jump']
    pressure_jump = additional_info['pressure_jump']

    plt.figure(figsize=(7, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time, temperature, color='blue')
    plt.plot(time_cooling, temp_cooling, 'o', color='red', label='start cooling')
    plt.xlabel('Time')
    plt.ylabel('Temperature')
    plt.legend(loc='upper right')

    plt.subplot(2, 1, 2)
    plt.plot(time, pressure, color='orange')
    plt.plot(time_jump, pressure_jump, 'o', color='red', label='start jump')
    plt.xlabel('Time')
    plt.ylabel('Pressure')
    plt.legend(loc='lower right')

    plt.tight_layout()



def plot_2d_figure(peaks, experiment_data, mass_spectr_data, color_plots, yaxis, resistance_measurement):
    time = experiment_data['time']
    temperature = experiment_data['temperature']
    pressure = experiment_data['pressure']

    # specify y axis
    if yaxis == 'I_max_norm':
        yaxis_label = 'intensity/Imax'
    elif yaxis == 'I_max':
        yaxis_label = 'intensity'
    elif yaxis == 'AUC':
        yaxis_label = 'integrated intensity'
    elif yaxis == 'AUC_norm':
        yaxis_label = 'normalized AUC'

    # plot figure
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, hspace=0, height_ratios=[3, 1])  # Adjust height_ratios to make first plot bigger
    axs = gs.subplots(sharex=True, sharey=False)
    for i,peak in enumerate(peaks.keys()):
        if peaks[peak]['name'] != 'dome':

            x_peak = peaks[peak]['time']
            y_peak = np.array(peaks[peak][yaxis])
            y_err = np.array(peaks[peak][yaxis+'_err'])

            if len(y_peak) != 0:
                # to have shaded errro bars
                axs[0].plot(x_peak, list(y_peak), label=peaks[peak]['name'], color=color_plots[i])
                axs[0].fill_between(x_peak, list(y_peak - y_err), list(y_peak + y_err), color=color_plots[i], alpha=0.5)
                axs[0].fill_between(x_peak, list(y_peak-y_err), color=color_plots[i], alpha=0.3)

    axs[0].legend()
    axs[0].set_ylabel(yaxis_label)

    if resistance_measurement:
        experiment_data['inv_resistance'] = [1/r for r in experiment_data['resistance']]
        ax2r = axs[0].twinx()
        ax2r.plot(experiment_data['time'], experiment_data['inv_resistance'], label='inv resistance', color='black')
        ax2r.set_ylabel('conductance (Omega^{-1}$)')

    # second subplot: temperature and pressure
    axs[1].plot(time, temperature, label='temperature', color='red')
    ax2 = axs[1].twinx()
    ax2.plot(time, pressure, label='pressure', color='blue')
    if mass_spectr_data:
        ax2.plot(mass_spectr_data['time_exp'], mass_spectr_data['co2_values_plot'], label='CO2', color='orange')
    axs[1].set_ylabel('temperature (ºC)')
    axs[1].set_xlabel('time (s)')
    ax2.set_ylabel('pressure (mbar)')

    # Combine legends for both temperature and pressure
    lines, labels = axs[1].get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    axs[1].legend(lines + lines2, labels + labels2, loc='lower right')

    # Hide x labels and tick labels for all but bottom plot.
    for ax in axs:
        ax.label_outer()


# it find the indices when the list decreases
def find_decreasing_indices(lst):
    decreasing_indices = []
    last_num = lst[0]
    for i in range(len(lst)):
        if lst[i] < last_num:
            decreasing_indices.append(i)
        else:
            last_num = lst[i]
    return decreasing_indices


def remove_elements_at_indices(lst, indices):
    return [value for i, value in enumerate(lst) if i not in indices]


def find_index_peak(peaks_dict, target_peak):
    for peak in peaks_dict:
        name = peaks_dict[peak]['name']
        if name == target_peak:
            index_peak = peak
    return index_peak


def find_omega_correction(final_file_path, YBCO_wrong_omega, dome_wrong_omega, additional_info):
    ''' It returns the ratio between the intensity of YBCO at correct omega and wrong omega after the cooling. We normalized
    both values by its dome in case the time acquisition was different.  '''

    with open(final_file_path, newline='') as csvfile:

        header = csvfile.readline().strip().split(',')
        filtered_header = [col for col in header if 'dome' in col or 'YBCO005' in col]
        filtered_header.append('omega')
        indices = [header.index(col) for col in filtered_header]

        data = np.genfromtxt(final_file_path, delimiter=',', skip_header=1, dtype=float, usecols=indices, filling_values=np.nan, unpack=True)
        parameters = dict(zip(filtered_header, data))

        ratio_YBCO_dome_correct_omega = parameters['YBCO005_I_max'] / parameters['dome_I_max']
        ratio_YBCO_dome_wrong_omega = YBCO_wrong_omega['I_max'][-1] / dome_wrong_omega['I_max'][-1]
        correction = ratio_YBCO_dome_correct_omega / ratio_YBCO_dome_wrong_omega

        if  not isinstance(ratio_YBCO_dome_correct_omega, float):
            raise ValueError('final csv file has more than one raw, only one expected')
        
        print(f"Growth at omega {YBCO_wrong_omega['omega'][-1]}.")
        additional_info['growth_omega'] = YBCO_wrong_omega['omega'][-1]
        if parameters['omega'] == 0:
            print('Omega not provided for the final csv file')
        else:
            print(f"Optimal omega for YBCO is {parameters['omega']}.")
            additional_info['optimal_omega'] = parameters['omega']

        print(f"The correction factor is {correction:.4f}.")
        additional_info['omega_corr'] = correction
        additional_info['ratio YBCO005/dome final'] = ratio_YBCO_dome_correct_omega

        return correction, additional_info




def correction_wrong_omega(peaks, final_file, additional_info):
    """
    It corrects the intensity and AUC of the YBCO during growth, caused because we were not at the optimal omega during.
    It corrects the YBCO intensity value by multiplying those values by the returns the ratio between the intensity of 
    YBCO at correct omega and wrong omega after the cooling. Correction applied only if the final file exists
    """
    if final_file:
        index_dome = find_index_peak(peaks, 'dome')
        index_YBCO = find_index_peak(peaks, 'YBCO005')
        omega_corr, additional_info = find_omega_correction(final_file, peaks[index_YBCO], peaks[index_dome], additional_info)

        # correct to I_max and AUC of the YBCO and their errors
        peaks[index_YBCO]['I_max'] = [value * omega_corr for value in peaks[index_YBCO]['I_max']]
        peaks[index_YBCO]['AUC'] = [value * omega_corr for value in peaks[index_YBCO]['AUC']]
        peaks[index_YBCO]['I_max_err'] = [value * omega_corr for value in peaks[index_YBCO]['I_max_err']]
        peaks[index_YBCO]['AUC_err'] = [value * omega_corr for value in peaks[index_YBCO]['AUC_err']]

        return peaks, additional_info

    else:
        print(' ')
        print('WARNING: not final file provided to calculate the ratio between the YBCO005 and the dome')
        return peaks, additional_info


def growth_YBCO005(peaks, additional_info):
    index_YBCO = find_index_peak(peaks, 'YBCO005')

    start_growth_YBCO_index = next((index for index, value in enumerate(peaks[index_YBCO]['I_max']) if value != 0), None)

    additional_info['start_growthYBCO_temp'] = peaks[index_YBCO]['temperature'][start_growth_YBCO_index]

    additional_info['start_growthYBCO_pressure'] = peaks[index_YBCO]['pressure'][start_growth_YBCO_index]

    return additional_info




def main():
    
    files_for_plotting, final_file, mass_spectr_file, directory = upload_csv_files()           # select csv files and keep all peak data in dictionary 'peaks_data'. 

    peaks_data, experiment_data = extract_peak_info(files_for_plotting)                        # each entrance is a peak named 'peak0','peak1',.. with its parameters as specified in 'peak_dict'

    peaks = join_same_peaks(peaks_data)                                                        # if two peaks have the same name (is same peak at different time), we  merge them and order by time

    mass_spectr_data = read_mass_spectrometer(mass_spectr_file)

    peaks, experiment_data['time'] = reinicialise_time(peaks, experiment_data['time'])

    additional_info = {}
    additional_info = find_pressure_jump(experiment_data, additional_info)

    if mass_spectr_data:
        mass_spectr_data = change_mass_spectr_data(mass_spectr_data, experiment_data, additional_info['time_jump'])

    peaks, additional_info = acquisition_time_compensation(peaks, additional_info)

    additional_info = find_start_cooling(experiment_data, additional_info)

    peaks, additional_info = correction_wrong_omega(peaks, final_file, additional_info)

    peaks = add_normalised_intensity(peaks, additional_info['time_cooling'])

    additional_info = growth_YBCO005(peaks, additional_info)

    save_peaks_information(peaks, additional_info, directory)


    # MAKE GRAPHS
    colors = colorcet.glasbey

    plot_cooling_and_pressure_jump(experiment_data, additional_info)
    plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'I_max_norm', False)
    plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'I_max', False)
    plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'AUC', False)
    plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'AUC_norm', False)

    if 'resistance' in experiment_data:
        plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'I_max_norm', True)
        plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'I_max', True)
        plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'AUC', True)
        plot_2d_figure(peaks, experiment_data, mass_spectr_data, colors, 'AUC_norm', True)

    plt.show()



if __name__ == "__main__":
    main()




""" # Open file dialog to select a .txt file
log_file = filedialog.askopenfilename(initialdir = 'Z:/PhD/Alba_Feb23/', 
                                      title = 'Select .log file',
                                      filetypes = (('log files', '*.log'),))

# Open file dialog to select a folder containing images
folder_path = filedialog.askdirectory(initialdir = 'Z:/PhD/Alba_Feb23/',
                                      title = 'Select the folder containing the rayonix images')

if not (log_file and folder_path):
    # Raise an exception if not all required data is selected
    raise Exception('You need to select both a .log file and a folder containing the XRD images') """