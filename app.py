from viktor import ViktorController, File
from viktor.api_v1 import FileResource
from viktor.parametrization import (
    ViktorParametrization,
    TextField,
    FileField,
    ActionButton,
    Table,
    Section,
    OptionField,
    AutocompleteField,
    MultiSelectField,
    IntegerField,
    NumberField,
    DynamicArray,
    BooleanField,
    Step,
    SetParamsButton,
    DownloadButton,
    Lookup,
    RowLookup,
    UserError
)
from viktor.views import (
    PlotlyResult,
    PlotlyView,
    ImageResult, 
    ImageView
)

from viktor.result import SetParamsResult, DownloadResult
from viktor.core import progress_message, Storage, File
from viktor.errors import UserError

import plotly.graph_objects as go
import numpy as np
import pandas as pd
import hashlib
import calplot
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

from itertools import zip_longest
from io import BytesIO
from geopy.geocoders import Nominatim
from geopy import distance
from pathlib import Path

# Suppress DeprecationWarnings
import warnings
warnings.filterwarnings('ignore', category=pd.errors.SettingWithCopyWarning )
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Parametrization option functions
def options_of_branch_array(params, **kwargs):
    # Return list of all audit branches from dynamic array
    return ['No store name specified' if row.name == '' else row.name for row in params.step_1.section_1.audit_branches]

def data_options_of_branch_array(params, **kwargs):
    # Return year range list of given audit_years in step 1 section 1
    audit_years = params.step_1.section_1.audit_years
    start_year, end_year = map(int, audit_years.split('-'))
    year_range = list(range(start_year, end_year + 1))
    return year_range

def graph_options_of_visualization_array(params, **kwargs):
    # Return combined list of all graph names based on visualization arrays (section 2, 3 and 4)
    name_list = []

    arrays_to_check = [
        params.step_2.section_2.visualization_array,
        params.step_2.section_3.impact_array,
        params.step_2.section_4.custom_array
    ]

    for array in arrays_to_check:
        for row in array:
            if row.name:
                name_list.append(row.name)

    if not name_list:
        name_list.append('No graph specified')

    return name_list

def options_nl_cities(params, **kwargs):
    # Load woonplaatsen.csv from weather_data folder
    script_folder = Path(__file__).parent
    file_path = script_folder / "weather_data" / "woonplaatsen.csv"
    
    data = pd.read_csv(file_path, sep=",")
    city_list = data.iloc[:, 0].unique().tolist()
    
    return city_list

# VIKTOR specific functions
def validate_step_1(params, **kwargs):
    # Check if all attributes have values, skip file uploads that are not necessary
    for row in params.step_1.section_1.audit_branches:
        for attribute, value in row.items():
            if attribute in ('g_upload', 'pv_upload'):
                continue  # Skip these attributes
            if not value:
                raise UserError("Not all array elements contain information.")
                
    if params.step_1.section_2.analyzed == False:
        raise UserError("Data not yet analysed, please use the Analyse data button.")

def select_branch_from_array(array, branch_name):
    for row in array:
        if row.name == branch_name: 
            return row

# EED Data Processing
def calculate_energy_cost(e_df, g_df, energytable):
    # Convert DateTime columns to datetime objects
    e_df['DateYear'] = (e_df['DateTime']).dt.year
    g_df['DateYear'] = (g_df['DateTime']).dt.year
    energytable['year'] = pd.to_datetime(energytable['year']).dt.year

    # Find the earliest available year in the energy price table
    earliest_year = energytable['year'].min()

    # Merge energy consumption dataframes with energy price dataframe based on year
    e_df_merged = pd.merge_asof(e_df, energytable, left_on='DateYear', right_on='year')
    g_df_merged = pd.merge_asof(g_df, energytable, left_on='DateYear', right_on='year')
    
    # Fill missing prices with the earliest year's prices
    e_df_merged = e_df_merged.fillna(energytable[energytable['year'] == earliest_year].iloc[0])
    g_df_merged = g_df_merged.fillna(energytable[energytable['year'] == earliest_year].iloc[0])
    
    
    # Calculate energy costs
    e_df_merged['Energy Cost (EUR)'] = (
        ( e_df_merged.iloc[:,0] - e_df_merged.iloc[:,1] ) * e_df_merged.apply(
            lambda row: row['peak'] if row['Classification'] == 'peak' else row['offpeak'],
            axis=1
        )
    )
    
    g_df_merged['Energy Cost (EUR)'] = g_df_merged.iloc[:,0] * g_df_merged['gas']
    
    # Drop columns based on column names
    columns_to_drop = ['peak', 'offpeak', 'gas', 'year', 'DateYear']
    e_df_merged = e_df_merged.drop(columns=columns_to_drop, axis=1)    
    g_df_merged = g_df_merged.drop(columns=columns_to_drop, axis=1)  

    return e_df_merged, g_df_merged

def calculate_emissions(e_df, g_df, emissionstable):
    # Convert DateTime columns to datetime objects
    e_df['DateYear'] = (e_df['DateTime']).dt.year
    g_df['DateYear'] = (g_df['DateTime']).dt.year
    emissionstable['year'] = pd.to_datetime(emissionstable['year']).dt.year

    # Find the earliest available year in the energy price table
    earliest_year = emissionstable['year'].min()
    last_year = emissionstable['year'].max()
    

    # Merge energy consumption dataframes with energy price dataframe based on year
    e_df_merged = pd.merge_asof(e_df, emissionstable, left_on='DateYear', right_on='year')
    g_df_merged = pd.merge_asof(g_df, emissionstable, left_on='DateYear', right_on='year')

   # Handle missing year matching in emissionstable
    def choose_emission_row(row):
        if pd.isnull(row['year']):
            if row['DateYear'] < earliest_year:
                return emissionstable[emissionstable['year'] == earliest_year].iloc[0]
            else:
                return emissionstable[emissionstable['year'] == last_year].iloc[0]
        return row

    e_df_merged = e_df_merged.apply(choose_emission_row, axis=1)
    g_df_merged = g_df_merged.apply(choose_emission_row, axis=1)

    e_df_merged['Energy Emissions (kg CO2-eq.)'] = ( e_df_merged.iloc[:,0] - e_df_merged.iloc[:,1] ) * e_df_merged['electricity_emissions']
    g_df_merged['Energy Emissions (kg CO2-eq.)'] = g_df_merged.iloc[:,0] * g_df_merged['gas_emissions']

    # Drop columns based on column names
    columns_to_drop = ['electricity_emissions', 'gas_emissions', 'grid_heat_emissions', 'year', 'DateYear']
    e_df_merged = e_df_merged.drop(columns=columns_to_drop, axis=1)    
    g_df_merged = g_df_merged.drop(columns=columns_to_drop, axis=1)    

    return e_df_merged, g_df_merged

def find_nearest_city(target_city):
    geolocator = Nominatim(user_agent="nearest_city_app")
    city_list = ['IJmond', 'Voorschoten', 'IJmuiden', 'Texelhors', 'De Kooy', 'Schiphol', 'Vlieland', 'Wijdenes', 'Berkhout', 'Hoorn', 'Wijk aan Zee', 'Houtribdijk', 'De Bilt', 'Stavoren', 'Lelystad', 'Leeuwarden', 'Marknesse', 'Deelen', 'Lauwersoog', 'Heino', 'Hoogeveen', 'Eelde', 'Hupsel', 'Huibertgat', 'Nieuw Beerta', 'Twenthe', 'Cadzand', 'Vlissingen', 'Oosterschelde', 'Vlakte v.d. Raan', 'Hansweert', 'Schaar', 'Westdorpe', 'Wilhelminadorp', 'Stavenisse', 'Hoek van Holland', 'Tholen', 'Woensdrecht', 'Rotterdam-Geulhaven', 'Rotterdam', 'Cabauw', 'Gilze-Rijen', 'Herwijnen', 'Eindhoven', 'Volkel', 'Ell', 'Maastricht', 'Arcen']
    target_location = geolocator.geocode(target_city + ", Netherlands", timeout=None)
    
    if target_location is None:
        return None
    
    target_coords = (target_location.latitude, target_location.longitude)
    nearest_city = None
    min_distance = float('inf')
    

    for city in city_list:
        city_location = geolocator.geocode(city + ", Netherlands", timeout=None)
        if city_location is None:
            continue
        
        city_coords = (city_location.latitude, city_location.longitude)
        distance_to_city = distance.distance(target_coords, city_coords).kilometers
        
        if distance_to_city < min_distance:
            min_distance = distance_to_city
            nearest_city = city
    
    return nearest_city

def inside_temperature(openinghours, df):
    # Convert Date column to datetime type
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Get the day of the week as an integer (Monday = 0, Tuesday = 1, etc.)
    df['Day'] = df['Date'].dt.dayofweek
    
    # Initialize the Binnen temperatuur column with default value of 15
    df.loc[:, 'Binnen temperatuur'] = 15
    
    # Iterate over the rows of the DataFrame
    for index, row in df.iterrows():
        day_index = row['Day']
        opening_hours = openinghours[day_index].split('-')
        start_hour, end_hour = int(opening_hours[0]), int(opening_hours[1])
        current_hour = row['Time'].hour
        
        # Check if the current hour is within the opening hours range
        if start_hour <= current_hour <= end_hour:
            df.at[index, 'Binnen temperatuur'] = 20
    
    # Drop the intermediate 'Day' column
    df = df.drop('Day', axis=1)
    return df

def outside_temperature(weather_location, df):
    entity_folder_path = Path(__file__).parent
    weather_data_directory = entity_folder_path / 'weather_data'
    if weather_location:
        # Find the text file with city_name in its name
        matching_files = [file for file in weather_data_directory.glob('*') if weather_location in file.name and file.name.endswith(".txt")]
        if matching_files:
            # Create an empty DataFrame to store the combined weather data
            combined_weather_df = pd.DataFrame()

            # Iterate over the selected files
            for selected_file in matching_files:
                # Construct the full path to the selected file
                selected_file_path = weather_data_directory / selected_file.name

                # Read the selected file as a DataFrame
                weather_df = pd.read_csv(selected_file_path, skiprows=31, low_memory=False)
                weather_df = weather_df.iloc[:,[1, 2, 7]]
                weather_df.iloc[:,2] /= 10

                # Convert the date and time column to datetime format
                weather_df.iloc[:, 0] = pd.to_datetime(weather_df.iloc[:, 0], format="%Y%m%d")

                # Convert the second column to time format and handle midnight (24) value
                weather_df.iloc[:, 1] = pd.to_numeric(weather_df.iloc[:, 1].astype(str).replace("24", "0")) % 24

                # Add 1 day to the first column where the second column is 0 (midnight)
                weather_df[weather_df.columns[0]] = weather_df[weather_df.columns[0]] + pd.to_timedelta((weather_df[weather_df.columns[1]] == 0).astype(int), unit="D")

                weather_df.iloc[:, 1] = pd.to_datetime(weather_df.iloc[:, 1].astype(str), format="%H", errors='coerce').dt.time

                # Change column names
                weather_df.columns = ["Date", "Time", "Temperature"]

                # Append the current weather data to the combined DataFrame
                combined_weather_df = pd.concat([combined_weather_df, weather_df], ignore_index=True)
        else:
            raise("No matching file found.")
    else:
        raise("City not found in the list.")

    # Merge the dataframes based on Date and Time columns
    merged_df = df.merge(combined_weather_df, on=["Date", "Time"], how="left")

    # Add the values from the 'Buiten temperatuur' column to 'df'
    df.loc[:, "Buiten temperatuur"] = merged_df["Temperature"]
    df['Temperatuur delta'] = df['Binnen temperatuur'] - df['Buiten temperatuur']
    df.loc[(df['Binnen temperatuur'] == 15) & (df['Binnen temperatuur'] < df['Buiten temperatuur']) & (df['Buiten temperatuur'] >= 25), 'Temperatuur delta'] += 5

    return df

def climate_data(openinghours, weather_location, df):
    climate_df = df[["Date", "Time"]].copy()

    # Binnen- en buiten temperatuur vestiging toevoegen
    climate_df = inside_temperature(openinghours, climate_df)
    climate_df = outside_temperature(weather_location, climate_df)
    return climate_df

def process_dataframe(df):
    df['DateTime'] = pd.to_datetime(df.iloc[:, 0], format='%d-%m-%Y %H:%M', errors='coerce')
    df['Date'] = df['DateTime'].dt.date
    df['Time'] = df['DateTime'].dt.time
    df.drop(df.columns[0], axis=1, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def classify_peak(row, peakhours):
    if peakhours[0] <= row['DateTime'].hour < peakhours[1] and row['DateTime'].weekday() < 5:
        return 'Peak'
    else:
        return 'Offpeak'

def joulz_reader(params, branch):
    # Read the CSV files from Joulz and process to standarised dataframe for EED analysis
    def find_skip_index(file, target):
        for idx, line in enumerate(file):
            if target in line.decode('utf-8'):
                return idx
        return None
    
    def process_dataframe_with_classification(df, peakhours):
        df = process_dataframe(df)
        df["Classification"] = df.apply(classify_peak, args=(peakhours,), axis=1)
        return df

    def read_and_process_csv(file, skiprows, header, peakhours=None):
        df = pd.read_csv(file, skiprows=skiprows, engine='python', skipfooter=1, header=header, encoding='utf-8')
        return process_dataframe_with_classification(df, peakhours) if peakhours else df

    def read_and_process_pv(file, peakhours):
        pv_df = pd.read_csv(file, skipfooter=1, engine='python', header=None, encoding='utf-8')
        header_row1 = pv_df.iloc[0].astype(str)
        header_row5 = pv_df.iloc[4].astype(str)
        header = header_row1 + ' ' + header_row5
        pv_df.columns = header
        pv_df = pv_df.iloc[5:,:]
        return process_dataframe_with_classification(pv_df, peakhours)


    peakhours = list(map(int, params.step_1.section_2.peakhours.split("-")))
    energytable = pd.DataFrame(params.step_1.section_2.energytable)
    emissionstable = pd.DataFrame(params.step_1.section_2.emissionstable)
    energytable.loc[:, "peak"] /= 1000
    energytable.loc[:, "offpeak"] /= 1000       
    
    # Open file from VIKTOR server
    e_upload = branch.e_upload.file.open_binary()
    if hasattr(branch, 'g_upload') and branch.g_upload is not None:
        g_upload = branch.g_upload.file.open_binary()
    else:
        g_upload = None

    if hasattr(branch, 'pv_upload') and branch.g_upload is not None:
        pv_upload = branch.pv_upload.file.open_binary()
    else:
        pv_upload = None
    
    # Find the row index with the value "Datum-tijd tot" in the first column
    datum_tijd_tot_index = find_skip_index(e_upload, "Datum-tijd tot")
    
    e_upload.seek(0)  # Reset the file pointer
    
    # Skip rows until the row with "Datum-tijd tot" using the found index
    e_df = read_and_process_csv(e_upload, skiprows=datum_tijd_tot_index, header=0, peakhours=peakhours)
    
    e_upload.close()

    g_df = e_df.drop(['Elektriciteit Productie (kWh)', 'Blindverbruik Consumptie (kVARh)', 'Blindverbruik Productie (kVARh)', 'Classification'], axis=1)
    g_df = g_df[g_df['DateTime'].dt.minute == 0].reset_index(drop=True)
    g_df.iloc[:,0] = 0
    
    if g_upload:
        g_df = read_and_process_csv(g_upload, skiprows=3, header=0, peakhours=peakhours)
        g_upload.close()

    pv_df = pd.DataFrame()
    if pv_upload:
        pv_df = read_and_process_pv(pv_upload, peakhours)
        pv_upload.close()
    
    e_df, g_df = calculate_energy_cost(e_df, g_df, energytable)
    e_df, g_df = calculate_emissions(e_df, g_df, emissionstable)
    return e_df, g_df, pv_df


# Diagrams and graph functions
def energy_profile_diagram(params, diagram, **kwargs):
    # Initial assignments
    section_1 = params.step_1.section_1
    language = section_1.language
    client = section_1.client
    branch_name = params.step_2.section_1.output_reference
    year_list = diagram.timerange if diagram.comparison else [section_1.baseyear]

    # Language-specific settings
    language_settings = {
        "English": {
            "x_ticks_season": ['Spring', 'Summer', 'Autumn', 'Winter'],
            "x_ticks_month": ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
            "x_ticks_day": ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
            "sources": {
                "Electricity": {
                    "unit": "kWh",
                    "label_consumption": "Electricity Consumption",
                    "label_production": "Electricity Production",
                    "y_label": "Electricity [{}]",
                    "name": "Electricity",
                    "bar_label": "Consumption {}"
                },
                "Natural gas": {
                    "unit": "Nm3",
                    "label_consumption": "Natural gas Consumption",
                    "y_label": "Natural Gas [{}]",
                    "name": "Natural gas",
                    "bar_label": "Consumption {}"
                }
            },
            "title": "Energy profile - {} [{}]",
            "timeframes": {
                "Day profile per hour": "Hourly {} day profile in {} - {} - {}",
                "Week profile per day": "Daily {} week profile in {} - {} - {}",
                "Year profile per month": "Monthly {} year profile in {} - {} - {}",
                "Year profile per season": "Seasonly {} year profile in {} - {} - {}"
            }
        },
        "Dutch": {
            "x_ticks_season": ['Lente', 'Zomer', 'Herfst', 'Winter'],
            "x_ticks_month": ['Jan', 'Feb', 'Mrt', 'Apr', 'Mei', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dec'],
            "x_ticks_day": ['Ma', 'Di', 'Wo', 'Do', 'Vr', 'Za', 'Zo'],
            "sources": {
                "Electricity": {
                    "unit": "kWh",
                    "label_consumption": "Electriciteitsverbruik",
                    "label_production": "Electriciteitsproductie",
                    "y_label": "Electriciteit [{}]",
                    "name": "Electriciteit",
                    "bar_label": "Consumptie {}"
                },
                "Natural gas": {
                    "unit": "Nm3",
                    "label_consumption": "Aardgasverbruik",
                    "y_label": "Aardgas [{}]",
                    "name": "Aardgas",
                    "bar_label": "Consumptie {}"
                }
            },
            "title": "Energiegebruiksprofiel - {} [{}]",
            "timeframes": {
                "Day profile per hour": "Uurlijks {} verbruik per dag in {} - {} - {} ",
                "Week profile per day": "Dagelijks {} verbruik per week in {} - {} - {}",
                "Year profile per month": "Maandelijks {} verbruik per jaar profile in {} - {} - {}",
                "Year profile per season": "Seizoensverbruik {} per jaar in {} - {} - {}"
            }
        }
    }

    # Language and source-specific assignments
    settings = language_settings[language]
    source_settings = settings["sources"][diagram.source]

    # Destructuring parameters from settings and source_settings
    x_ticks_season = settings["x_ticks_season"]
    x_ticks_month = settings["x_ticks_month"]
    x_ticks_day = settings["x_ticks_day"]

    unit = source_settings["unit"]
    label_production = source_settings.get("label_production")  # using .get() in case some sources don't have this key
    label_consumption = source_settings["label_consumption"]
    y_label = source_settings["y_label"].format(unit)
    bar_label = [source_settings["bar_label"].format(year) for year in year_list]

    title = settings["title"].format(source_settings["name"], unit)
    subtitle = settings["timeframes"][diagram.timeframe].format(source_settings["name"].lower(), year_list, client, branch_name)

    if isinstance(year_list, list) and len(year_list) == 1:
        year_list = year_list[0]



    # Retrieve correct dataset
    branch = select_branch_from_array(params.step_1.section_1.audit_branches, params.step_2.section_1.output_reference)
    if diagram.source == "Electricity":
        key_name = branch.name.replace(" ", "_") + "_e_df"
    elif diagram.source == "Natural gas":
        key_name = branch.name.replace(" ", "_") + "_g_df"
    data_temp = Storage().get(key_name, scope='entity')
    with data_temp.open_binary() as file:
        data = pd.read_csv(file)
        data['DateTime'] = pd.to_datetime(data['DateTime'])

    # Grouping data based on x_range and aggregating gas consumption and power production
    timeframe_settings = {
        'Year profile per season': {
            'x_value_function': lambda dt: (
                'Spring' if dt.month in range(3, 6) else
                'Summer' if dt.month in range(6, 9) else
                'Autumn' if dt.month in range(9, 12) else
                'Winter'
            ),
            'x_label': 'Season',
            'x_ticks': x_ticks_season,
            'mapping': {
                'Spring': 0,
                'Summer': 1,
                'Autumn': 2,
                'Winter': 3
            },
            'average': 1
        },
        'Year profile per month': {
            'x_value_function': lambda dt: dt.strftime('%B'),
            'x_label': 'Month',
            'x_ticks': x_ticks_month,
            'mapping': {
                'January': 0,
                'February': 1,
                'March': 2,
                'April': 3,
                'May': 4,
                'June': 5,
                'July': 6,
                'August': 7,
                'September': 8,
                'October': 9,
                'November': 10,
                'December': 11
            },
            'average': 1
        },
        'Week profile per day': {
            'x_value_function': lambda dt: dt.strftime('%A'),
            'x_label': 'Day',
            'x_ticks': x_ticks_day,
            'mapping': {
                'Monday': 0,
                'Tuesday': 1,
                'Wednesday': 2,
                'Thursday': 3,
                'Friday': 4,
                'Saturday': 5,
                'Sunday': 6
            },
            'average': 52
        },
        'Day profile per hour': {
            'x_value_function': lambda dt: dt.hour,
            'x_label': 'Hour',
            'x_ticks': [str(i) for i in range(1, 25)],
            'mapping': {0: 24, **{i: i for i in range(1, 24)}},
            'average': 365
        }
    }

    # Retrieve settings based on diagram.timeframe
    current_setting = timeframe_settings.get(diagram.timeframe)

    if current_setting:
        data['x_value'] = data["DateTime"].map(current_setting['x_value_function'])
        x_label = current_setting['x_label']
        x_ticks = current_setting['x_ticks']
        mapping = current_setting.get('mapping', {})
        average = int(current_setting['average'])

    # Average hourly or daily values to their corresponding use (not sum of all values)
    if diagram.repetition == True:
        data[data.columns[0]] /= average
        if diagram.source == "Electricity":
            data[data.columns[1]] /= average    

    # Separate columns into numeric and non-numeric groups
    numeric_columns = []
    non_numeric_columns = []

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            numeric_columns.append(column)
        else:
            non_numeric_columns.append(column)

    base_data = data[data["DateTime"].dt.year == section_1.baseyear]
    if diagram.comparison == True:
        range_data = data[data["DateTime"].dt.year.isin(diagram.timerange)]


    # Grouping by x_value and calculating the sum of consumption and production for the base year
    grouped_base_data = base_data.groupby('x_value')[numeric_columns].agg('sum')

    grouped_base_data['x_index'] = grouped_base_data.index.map(mapping)
    grouped_base_data = grouped_base_data.sort_values(by=["x_index"])


    # Compute net and average consumption
    if diagram.production and diagram.source == "Electricity":
        grouped_base_data['net'] = grouped_base_data.iloc[:, 0] - grouped_base_data.iloc[:, 1]
        column_of_interest = grouped_base_data['net']
    else:
        column_of_interest = grouped_base_data.iloc[:, 0]

    avg_consumption = column_of_interest.mean()
    baseload_value = column_of_interest.min()

    # Setting up the figure and axes
    fig, ax = plt.subplots()

    if diagram.comparison == True:
        # Define color palettes
        red_gradient = mcolors.LinearSegmentedColormap.from_list('red_gradient', ['#D63232', '#F9C5C5'])
        blue_gradient = mcolors.LinearSegmentedColormap.from_list('blue_gradient', ['#2B5C8A','#96D3FA'])
        colour_count = len(diagram.timerange)
        # Create color lists based on the number of colors required
        red_colors = [red_gradient(1.0 - i / colour_count) for i in range(colour_count)]
        blue_colors = [blue_gradient(1.0 - i / colour_count) for i in range(colour_count)]

        # Iterate over each year and create a separate grouped bar chart
        counter = 0
        num_years = len(diagram.timerange)
        
        spacing = 0.25
        group_width = 1 - spacing
        bar_width =  group_width / num_years
        x_range = len(mapping)

        for i, year in enumerate(diagram.timerange):
            year_data = range_data[range_data["DateTime"].dt.year == year]

            year_data = year_data.groupby('x_value')[numeric_columns].agg('sum')

            year_data['x_index'] = year_data.index.map(mapping)

            year_data = year_data.sort_values(by=["x_index"])

            # Calculate the x position for the bars to create gaps
            x_positions = year_data['x_index'] + (counter - (num_years - 1) / 2) * bar_width 

            # Plotting the consumption bars with red gradient
            consumption_bars = ax.bar(x_positions, year_data[year_data.columns[0]], width=bar_width*(1-0.015*x_range), label=bar_label[i], align='center', edgecolor='none')
            for idx, bar in enumerate(consumption_bars):
                color = red_colors[counter]
                bar.set_color(color)

            # Plotting the production bars with blue gradient
            if diagram.production == True and diagram.source == "Electricity":
                production_bars = ax.bar(x_positions, -year_data[year_data.columns[1]], width=bar_width*(1-0.015*x_range), align='center', edgecolor='none')
                for idx, bar in enumerate(production_bars):
                    color = blue_colors[counter]
                    bar.set_color(color)
                    
            counter += 1
        
        # Set the x-tick labels and positions
        ax.set_xticks(year_data['x_index'])  # Use the common x_index values for all years
        ax.set_xticklabels(x_ticks, fontsize=8)
                    
    else: 
        consumption_bars = ax.bar(grouped_base_data['x_index'], grouped_base_data[grouped_base_data.columns[0]], edgecolor='black')
        for idx, bar in enumerate(consumption_bars):
            color = '#D63232'
            bar.set_color(color)

        # Plotting the production bars with blue gradient
        if diagram.production == True and diagram.source == "Electricity":
            production_bars = ax.bar(grouped_base_data['x_index'], -grouped_base_data[data.columns[1]], edgecolor='black')
            for idx, bar in enumerate(production_bars):
                color = '#2B5C8A'
                bar.set_color(color)
        
        # Set the x-tick labels and positions
        ax.set_xticks(grouped_base_data['x_index'])
        ax.set_xticklabels(x_ticks, fontsize=8)

    # Common chart settings
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter('{:,.0f}'.format)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.text(0.5, -0.3, subtitle, ha='center', fontsize=8, fontstyle='italic', color='gray', transform=ax.transAxes)

    # Legend settings
    ax.bar(0, 0, color='#D63232', label=label_consumption)
    if diagram.production and diagram.source == 'Electricity':
        ax.bar(0, 0, color='#2B5C8A', label=label_production)
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3, fontsize=8)
    plt.setp(legend.get_texts(), color='black')

    # Lines settings based on flags
    ref_year = section_1.baseyear
    line_label = lambda label_text: f"{label_text} ({ref_year})"
    if not diagram.comparison or diagram.averageline:
        ax.axhline(y=avg_consumption, color='black', linestyle=':', linewidth=1, label=line_label('Average net consumption'))
    if not diagram.comparison or diagram.baseline:
        ax.axhline(y=baseload_value, color='black', linestyle='--', linewidth=1, label=line_label('Baseload net consumption'))
    if (not diagram.comparison or diagram.netline) and diagram.production and diagram.source == "Electricity":
        ax.plot(grouped_base_data['x_index'], grouped_base_data['net'], color='green', linestyle='-', linewidth=1, label=line_label('Net consumption'))
    if diagram.production and diagram.source == "Electricity":
        ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # Layout and saving
    plt.tight_layout()
    graph_data = BytesIO()
    data_format = params.step_2.section_1.output_format.replace(".", "")
    fig.savefig(graph_data, format=data_format, dpi=300)
    plt.close()

    return graph_data

def energy_impact_graph(params, impact_graph, **kwargs):
    # Initial assignments
    section_1 = params.step_1.section_1
    language = section_1.language
    client = section_1.client
    branch_name = params.step_2.section_1.output_reference
    years_str = params.step_1.section_1.audit_years

    start_year, end_year = map(int, years_str.split('-'))
    year_list = list(range(start_year, end_year + 1))

    # Language-specific settings
    language_settings = {
        "English": {
            "x_ticks_season": ['Spring', 'Summer', 'Autumn', 'Winter'],
            "impact": {
                "Cost": {
                    "title": "Energy cost profile [{}]",
                    "subtitle": "Evolution of the energy cost profile in {} - {} - {}",
                    "unit": "EUR",
                    "label_consumption": "Energy Cost",
                    "x_label": 'Month',
                    "y_label": "Cost Energy Usage [{}]",
                    "name": "Cost",
                    "bar_label": "Costs {}"
                },
                "Emission": {
                    "title": "Energy emission profile [{}]",
                    "subtitle": "Evolution of the energy related emission profile in {} - {} - {}",
                    "unit": "kg CO2-eq.",
                    "label_consumption": "GHG Emissions",
                    "x_label": 'Month',
                    "y_label": "GHG Emissions Energy Usage[{}]",
                    "name": "Emission",
                    "bar_label": "Emissions {}"
                }
            }
        },
        "Dutch": {
            "x_ticks_season": ['Lente', 'Zomer', 'Herfst', 'Winter'],
            "impact": {
                "Cost": {
                    "title": "Energiekostenprofiel [{}]",
                    "subtitle": "Evolutie van het Energiekostenprofiel in {} - {} - {}",
                    "unit": "EUR",
                    "label_consumption": "Energie kosten",
                    "x_label": 'Maand',
                    "y_label": "Kosten Energieverbruik [{}]",
                    "name": "Kosten"
                },
                "Emissions": {
                    "title": "Emissieprofiel [{}]",
                    "subtitle": "Evolutie van het Energie emissieprofiel in {} - {} - {}",
                    "unit": "kg CO2-eq.",
                    "label_consumption": "GHG Emissies",
                    "x_label": 'Maand',
                    "y_label": "GHG Emissies Energieverbruik [{}]",
                    "name": "Emissies"
                }
            }
        }
    }

    # Language and source-specific assignments
    settings = language_settings[language]
    source_settings = settings["impact"][impact_graph.impact]

    # Destructuring parameters from settings and source_settings
    unit = source_settings["unit"]
    label_consumption = source_settings["label_consumption"]
    y_label = source_settings["y_label"].format(unit)
    x_label = source_settings["x_label"]

    title = source_settings["title"].format(unit)
    subtitle = source_settings["subtitle"].format(year_list, client, branch_name)

    

    # Retrieve correct dataset
    branch = select_branch_from_array(params.step_1.section_1.audit_branches, params.step_2.section_1.output_reference)
    retrieve_settings = {'electricity': {
                                'name': 'Electricity',
                                'storage': '_e_df'
                            },
                        'gas': {
                                'name': 'Natural Gas',
                                'storage': '_g_df'
                            }
                         }

    merged_dataframe = pd.DataFrame()
    for source in ['electricity', 'gas']:
        name = retrieve_settings[source]['name']
        storage = retrieve_settings[source]['storage']
        key_name = branch.name.replace(" ", "_") + storage
        data_temp = Storage().get(key_name, scope='entity')
        with data_temp.open_binary() as file:
            data = pd.read_csv(file)
            data['DateTime'] = pd.to_datetime(data['DateTime'])

        # Separate columns into numeric and non-numeric groups
        numeric_columns = []
        non_numeric_columns = []

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                numeric_columns.append(column)
            else:
                non_numeric_columns.append(column)

        data = data[data["DateTime"].dt.year.isin(year_list)]
        data["Year"] = data['DateTime'].dt.year
        data["Month"] = data['DateTime'].dt.month

        if source == 'electricity':
            grouped_data = data.groupby(['Year', 'Month', 'Classification'])[numeric_columns].agg('sum').reset_index()
        else:
            grouped_data = data.groupby(['Year', 'Month'])[numeric_columns].agg('sum').reset_index()

        # Dropping columns based on substrings in column names
        columns_to_drop = ['(kWh)', '(kVARh)', 'gas)']
        filtered_columns = [col for col in grouped_data.columns if not any(sub in col for sub in columns_to_drop)]
        filtered_data = grouped_data[filtered_columns]
        
        # Add a new column
        if source == 'electricity':
            filtered_data['Source'] = filtered_data['Classification'] + " " + name
            filtered_data = filtered_data.drop("Classification", axis=1)
        else:
            filtered_data['Source'] = name

        merged_dataframe = pd.concat([merged_dataframe, filtered_data], axis=0).reset_index(drop=True)

    merged_dataframe['DateTime'] = pd.to_datetime(merged_dataframe[['Year', 'Month']].assign(day=1))
    merged_dataframe = merged_dataframe.drop(["Year", "Month"], axis=1)
    
    # Define colors for each source
    colors = {
        'Offpeak Electricity': '#96D3FA',
        'Peak Electricity': '#2B5C8A',
        'Natural Gas': '#D63232'
    }

    fig, ax = plt.subplots()
    
    # Pivot the DataFrame to create the stacked bar chart
    if impact_graph.impact == "Cost":
        pivot_df = merged_dataframe.pivot(index='DateTime', columns='Source', values='Energy Cost (EUR)')
    elif impact_graph.impact == "Emissions":
        pivot_df = merged_dataframe.pivot(index='DateTime', columns='Source', values='Energy Emissions (kg CO2-eq.)')

    # Convert the DateTime index to an array of Python datetime objects
    datetime_objects = pivot_df.index.to_pydatetime()

    # Plotting stacked areas for each column
    bottom = None
    for col in pivot_df.columns:
        if bottom is None:
            ax.fill_between(datetime_objects, pivot_df[col], label=col, color=colors[col])
            bottom = pivot_df[col]
        else:
            ax.fill_between(datetime_objects, bottom, bottom + pivot_df[col], label=col, color=colors[col])
            bottom += pivot_df[col]


    # Advanced chart settings
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.set_xlim(datetime_objects[0], datetime_objects[-1])
    ax.set_ylim(0, pivot_df.max().max())
    plt.gcf().autofmt_xdate()

    # Common chart settings
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    # ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter('{:,.0f}'.format)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.text(0.5, -0.3, subtitle, ha='center', fontsize=8, fontstyle='italic', color='gray', transform=ax.transAxes)

    # Legend settings
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3, fontsize=8)
    plt.setp(legend.get_texts(), color='black')

    # Layout and saving
    plt.tight_layout()
    graph_data = BytesIO()
    data_format = params.step_2.section_1.output_format.replace(".", "")
    fig.savefig(graph_data, format=data_format, dpi=300)
    plt.close()

    return graph_data

def cost_projection_graph(params, impact_graph, **kwargs):
    # Initial assignments
    section_1 = params.step_1.section_1
    language = section_1.language
    client = section_1.client
    branch_name = params.step_2.section_1.output_reference
    year = params.step_1.section_1.baseyear

    # Language-specific settings
    language_settings = {
        "English": {
            "title": "Energy cost projections [{}]",
            "subtitle": "Yearly energy cost for {} - {} (reference: {})",
            "unit": "EUR",
            "y_label": "Energy Usage Cost[{}]",
            "name": "Cost"
        },
        "Dutch": {
            "title": "Energiekostenprojectie [{}]",
            "subtitle": "Jaarlijkse energiekosten {} - {} (referentie: {})",
            "unit": "EUR",
            "y_label": "Kosten Energieverbruik [{}]",
            "name": "Kosten"
            }
        }

    # Language and source-specific assignments
    settings = language_settings[language]

    # Destructuring parameters from settings and source_settings
    unit = settings["unit"]
    y_label = settings["y_label"].format(unit)
    title = settings["title"].format(unit)
    subtitle = settings["subtitle"].format(client, branch_name, year)

    # Retrieve correct dataset
    branch = select_branch_from_array(params.step_1.section_1.audit_branches, params.step_2.section_1.output_reference)
    retrieve_settings = {'electricity': {
                            'name': 'Electricity',
                            'storage': '_e_df'
                        },
                    'gas': {
                            'name': 'Natural Gas',
                            'storage': '_g_df'
                        }
                        }
    
    merged_dataframe = pd.DataFrame()
    for source in ['electricity', 'gas']:
        name = retrieve_settings[source]['name']
        storage = retrieve_settings[source]['storage']
        key_name = branch.name.replace(" ","_") + storage
        data_temp = Storage().get(key_name, scope='entity')
        with data_temp.open_binary() as file:
            data = pd.read_csv(file)
            data['DateTime'] = pd.to_datetime(data['DateTime'])

        # Separate columns into numeric and non-numeric groups
        numeric_columns = []
        non_numeric_columns = []

        for column in data.columns:
            if pd.api.types.is_numeric_dtype(data[column]):
                numeric_columns.append(column)
            else:
                non_numeric_columns.append(column)

        data = data[data["DateTime"].dt.year.isin([year])]
        data["Year"] = data['DateTime'].dt.year

        if source == 'electricity':
            grouped_data = data.groupby(['Year', 'Classification'])[numeric_columns].agg('sum').reset_index()
        else:
            grouped_data = data.groupby(['Year'])[numeric_columns].agg('sum').reset_index()

        # Dropping columns based on substrings in column names
        columns_to_drop = ['(EUR)', '(kVARh)', 'CO2-eq.)']
        filtered_columns = [col for col in grouped_data.columns if not any(sub in col for sub in columns_to_drop)]
        filtered_data = grouped_data[filtered_columns]
        
        # Add a new column
        if source == 'electricity':
            filtered_data['Source'] = filtered_data['Classification'] + " " + name
            filtered_data = filtered_data.drop("Classification", axis=1)
        else:
            filtered_data['Source'] = name

        merged_dataframe = pd.concat([merged_dataframe, filtered_data], axis=0).reset_index(drop=True)


    # Pivot the DataFrame
    pivot_df = merged_dataframe.pivot_table(index='Year', columns='Source', values=['Elektriciteit Consumptie (kWh)', 'Elektriciteit Productie (kWh)', 'Gas Consumptie (m³ gas)'])

    # Flatten the multi-level columns
    pivot_df.columns = [f'{col[1]} ({col[0]})' for col in pivot_df.columns]
    
    pivot_df['Offpeak Electricity (kWh)'] = pivot_df.iloc[:,0] - pivot_df.iloc[:,2]
    pivot_df['Peak Electricity (kWh)'] = pivot_df.iloc[:,1] - pivot_df.iloc[:,3]
    pivot_df['Natural gas (Nm3)'] = pivot_df.iloc[:,4]
    pivot_df = pivot_df.iloc[:, 5:]
    pivot_df['year'] = year

    #load energytable
    energytable = pd.DataFrame(params.step_1.section_2.energytable)

    # Duplicate the row and incrementally create rows up to max known projection year
    dfs_to_concat = [pivot_df]
    for year in range(year+1, int(energytable["year"].max())+1):
        new_row = pivot_df.iloc[0].copy()
        new_row['year'] = year
        new_df = pd.DataFrame([new_row])
        dfs_to_concat.append(new_df)

    # Concatenate the DataFrames
    projected_df = pd.concat(dfs_to_concat, ignore_index=True)
    # Convert 'year' column values to strings
    projected_df['year'] = projected_df['year'].astype(int).astype(str)
    
    projected_df = pd.merge(projected_df, energytable, on='year', how='outer')
    # Set 'year' column as the index and set to float
    projected_df.set_index('year', inplace=True)
    projected_df = projected_df.astype(float)


    projected_df['Offpeak Electricity (EUR)'] = projected_df['Offpeak Electricity (kWh)'] * projected_df['offpeak'] /1000
    projected_df['Peak Electricity (EUR)'] = projected_df['Peak Electricity (kWh)'] * projected_df['peak'] /1000
    projected_df['Natural Gas (EUR)'] =projected_df['Natural gas (Nm3)'] * projected_df['gas']

    # Remove all other columns except for cost
    projected_df = projected_df.iloc[:, -3:]

        
    # Define colors for each source
    colors = {
        'Offpeak Electricity (EUR)': '#96D3FA',
        'Peak Electricity (EUR)': '#2B5C8A',
        'Natural Gas (EUR)': '#D63232'
    }

    # Plot
    fig, ax = plt.subplots()

    # Convert the DateTime index to an array of Python datetime objects
    datetime_objects = pd.to_datetime(projected_df.index).to_pydatetime()

    # Plotting stacked areas for each column
    bottom = None
    for col in projected_df.columns:
        if bottom is None:
            ax.fill_between(datetime_objects, projected_df[col], label=col, color=colors[col])
            bottom = projected_df[col]
        else:
            ax.fill_between(datetime_objects, bottom, bottom + projected_df[col], label=col, color=colors[col])
            bottom += projected_df[col]


    # Advanced chart settings
    ax.xaxis.set_major_locator(mdates.YearLocator(5,month=1,day=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1, month=1,day=1))
    ax.set_xlim(datetime_objects[0], datetime_objects[-1])
    ax.set_ylim(0, projected_df.max().max())
    plt.gcf().autofmt_xdate()

    # Common chart settings
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    # ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.yaxis.set_major_formatter('{:,.0f}'.format)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.text(0.5, -0.3, subtitle, ha='center', fontsize=8, fontstyle='italic', color='gray', transform=ax.transAxes)

    # Legend settings
    legend = ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.4), ncol=3, fontsize=8)
    plt.setp(legend.get_texts(), color='black')

    # Layout and saving
    plt.tight_layout()
    graph_data = BytesIO()
    data_format = params.step_2.section_1.output_format.replace(".", "")
    fig.savefig(graph_data, format=data_format, dpi=300)
    plt.close()

    return graph_data

def energy_table(params, **kwargs):
    # Retrieve correct dataset
    branch = select_branch_from_array(params.step_1.section_1.audit_branches, params.step_2.section_1.output_reference)
    key_name_E = branch.name.replace(" ", "_") + "_e_df"
    key_name_G = branch.name.replace(" ", "_") + "_g_df"
    data_temp_E = Storage().get(key_name_E, scope='entity')
    data_temp_G = Storage().get(key_name_G, scope='entity')
    with data_temp_E.open_binary() as file:
        data_E = pd.read_csv(file)
        data_E['DateTime'] = pd.to_datetime(data_E['DateTime'])
    with data_temp_G.open_binary() as file:
        data_G = pd.read_csv(file)
        data_G['DateTime'] = pd.to_datetime(data_G['DateTime'])

    # Separate columns into numeric and non-numeric groups
    numeric_columns_E = []
    numeric_columns_G = []
    
    for column in data_E.columns:
        if pd.api.types.is_numeric_dtype(data_E[column]):
            numeric_columns_E.append(column)
    for column in data_G.columns:
        if pd.api.types.is_numeric_dtype(data_G[column]):
            numeric_columns_G.append(column)

    data_E = data_E[data_E["DateTime"].dt.year == params.step_1.section_1.baseyear]
    data_E["year"] = data_E["DateTime"].dt.year

    data_G = data_G[data_G["DateTime"].dt.year == params.step_1.section_1.baseyear]
    data_G["year"] = data_G["DateTime"].dt.year


    # Grouping by x_value and calculating the sum of consumption and production for the base year
    data_E = data_E.groupby(['year', "Classification"])[numeric_columns_E].agg('sum').reset_index()
    data_G = data_G.groupby('year')[numeric_columns_G].agg('sum')
    
    return data_E, data_G

def overview_table(df_E, df_G):
    df_merged = pd.merge(df_E, df_G, how='outer')
    df_merged = df_merged.drop('year', axis=1)

    # Get the column names
    column_names = df_merged.columns.tolist()

    # Find the index of the last column
    last_column_index = len(column_names) - 1
    # Find the index of the "Energy Cost (EUR)" column
    target_column_index = column_names.index("Energy Cost (EUR)")

    # Pop the last column and insert it before the target column
    last_column = column_names.pop(last_column_index)
    column_names.insert(target_column_index, last_column)

    # Reorder the columns
    df = df_merged[column_names]

    # Add a new row for the sum
    sum_row = df.drop(columns='Classification').apply(pd.to_numeric, errors='coerce').sum()
    sum_row['Classification'] = 'Total'
    df = df.append(sum_row, ignore_index=True)

    # Reorder columns
    df = df[['Classification'] + [col for col in df.columns if col != 'Classification']]

    def format_column_value(column_name, value):
        if isinstance(value, str):  # Check if the value is a string
            return value, column_name

        if ' (EUR)' in column_name:
            return '€ {:,.2f}'.format(value), column_name
        elif ' (kg CO2-eq.)' in column_name:
            if all(df[column_name] > 1000):
                value /= 1000
                column_name = column_name.replace('kg', 'tonne')
            return '{:,.1f}'.format(value), column_name
        elif ' (kWh)' in column_name:
            if all(df[column_name] > 1000):
                value /= 1000
                column_name = column_name.replace('kWh', 'MWh')
            return '{:,.1f}'.format(value), column_name
        else:
            return '{:,.1f}'.format(value), column_name

    formatted_values = []
    formatted_column_names = []
    for column in df.columns:
        formatted_vals, formatted_col = zip(*df[column].apply(lambda x: format_column_value(column, x)))
        formatted_values.append(formatted_vals)
        formatted_column_names.append(formatted_col[0])

    formatted_df = pd.DataFrame(formatted_values).T
    formatted_df.columns = formatted_column_names

    formatted_df = formatted_df.replace("nan", "-")
    return formatted_df, df

def overview_graphs(params, impact_graph, **kwargs):
    # Initial assignments
    section_1 = params.step_1.section_1
    language = section_1.language
    client = section_1.client
    branch_name = params.step_2.section_1.output_reference
    years_str = params.step_1.section_1.audit_years

    start_year, end_year = map(int, years_str.split('-'))
    year_list = list(range(start_year, end_year + 1))

    # Language-specific settings
    language_settings = {
        "English": {
            'Temperature difference': {
                "title": "Temperature difference inside:outside",
                "subtitle": "Calculated temperature difference for {} - {} [{}]",
                "unit": "*C",
                "y_label": "Temperature difference [{}]"
            },
            'Electricity demand overview': {
                "title": "Electricity demand overview",
                "subtitle": "Net electricity demand of {} - {}",
                "unit": "kWh",
                "y_label": "Electricity consumption [{}]"
            },
            'Natural gas demand overview': {
                "title": "Natural gas demand overview",
                "subtitle": "Natural gas demand of {} - {} [{}]",
                "unit": "Nm3",
                "y_label": "Natural gas consumption [{}]"
            }
        },
        "Dutch": {
        'Temperature difference': {
                "title": "Temperatuurverschil binnen:buiten",
                "subtitle": "Berekend temperatuurverschil {} - {} [{}]",
                "unit": "*C",
                "y_label": "Temperatuurverschil [{}]"
            },
            'Electricity demand overview': {
                "title": "Electriciteitsverbruik overzicht",
                "subtitle": "Netto electriciteitsverbruik {} - {} [{}]",
                "unit": "kWh",
                "y_label": "Electriciteitsconsumptie [{}]"
            },
            'Natural gas demand overview': {
                "title": "Gasverbruik overzicht",
                "subtitle": "Gasverbruik {} - {} [{}]",
                "unit": "Nm3",
                "y_label": "Gasconsumptie [{}]"
            }
        }
    }

    # Language and source-specific assignments
    settings = language_settings[language][impact_graph.selection]

    # Destructuring parameters from settings and source_settings
    unit = settings["unit"]
    y_label = settings["y_label"].format(unit)
    title = settings["title"]
    subtitle = settings["subtitle"].format(client, branch_name, unit)

    # Retrieve correct dataset
    branch = select_branch_from_array(params.step_1.section_1.audit_branches, params.step_2.section_1.output_reference)
    retrieve_settings = {'electricity': {
                            'name': 'Electricity',
                            'storage': '_e_df'
                        },
                    'gas': {
                            'name': 'Natural Gas',
                            'storage': '_g_df'
                        },
                    'weather': {
                            'name': 'Climate data',
                            'storage': '_climate_df'
                        }
                        }
    
    if impact_graph.selection == "Temperature difference":
        name = retrieve_settings['weather']['name']
        storage = retrieve_settings['weather']['storage']
        
    elif impact_graph.selection == "Electricity demand overview":
        name = retrieve_settings['electricity']['name']
        storage = retrieve_settings['electricity']['storage']
    
    elif impact_graph.selection == "Natural gas demand overview":
        name = retrieve_settings['gas']['name']
        storage = retrieve_settings['gas']['storage']

    key_name = branch.name.replace(" ", "_") + storage

    data_temp = Storage().get(key_name, scope='entity')
    with data_temp.open_binary() as file:
        data = pd.read_csv(file)
        if impact_graph.selection == "Temperature difference":
            data['DateTime'] = pd.to_datetime(data['Date'])
        else: 
            data['DateTime'] = pd.to_datetime(data['DateTime'])

    # # Groupby day
    # Separate columns into numeric and non-numeric groups
    numeric_columns = []
    non_numeric_columns = []

    for column in data.columns:
        if pd.api.types.is_numeric_dtype(data[column]):
            numeric_columns.append(column)
        else:
            non_numeric_columns.append(column)

    data = data[data["DateTime"].dt.year.isin(year_list)]
    data["Year"] = data['DateTime'].dt.year
    data["Month"] = data['DateTime'].dt.month
    data["Day"] = data['DateTime'].dt.day

    if impact_graph.selection == "Temperature difference":
        grouped_data = data.groupby(['Year', 'Month', 'Day'])[numeric_columns].median().reset_index()
        cmap_data = matplotlib.colors.LinearSegmentedColormap.from_list("custom_cmap", [(0.0, "darkblue"), (10/35, "lightyellow"), (22.5/35, '#FFCC80') ,(1.0, "red")])

    else:
        grouped_data = data.groupby(['Year', 'Month', 'Day'])[numeric_columns].agg('sum').reset_index()
        cmap_data = "YlOrRd"

    # Set 'datetime' as the index
    grouped_data['DateTime'] = pd.to_datetime(grouped_data[['Year', 'Month', 'Day']])
    grouped_data.set_index('DateTime', inplace=True)

    # Drop the year, month, and day columns
    grouped_data.drop(['Year', 'Month', 'Day'], axis=1, inplace=True)

    if impact_graph.selection == "Temperature difference":
        data_series = pd.Series(grouped_data.iloc[:,-1])
        vmax_input = 25
        vmin_input = -10

    elif impact_graph.selection == "Electricity demand overview":
        grouped_data.iloc[:,0] = grouped_data.iloc[:,0] - grouped_data.iloc[:,1]
        data_series = pd.Series(grouped_data.iloc[:,0])
        vmax_input = np.ceil(data_series.max() / 250) * 250
        vmin_input = 0
    else:
        data_series = pd.Series(grouped_data.iloc[:,0])
        vmax_input = np.ceil(data_series.max() / 250) * 250
        vmin_input = 0

    
    fig, ax = calplot.calplot(data_series,
                figsize=(20,9),
                yearascending=False,
                cmap= cmap_data,
                suptitle_kws = {'x': 0.0, 'y': 0.0},
                vmax= vmax_input,
                vmin=vmin_input,
                tight_layout=True)
    fig.suptitle(subtitle, fontsize=24)
    # Layout and saving
    plt.subplots_adjust(top=0.85)
    graph_data = BytesIO()
    data_format = params.step_2.section_1.output_format.replace(".", "")
    fig.savefig(graph_data, format=data_format, dpi=300)
    plt.close()

    return graph_data


## VIKTOR CODE ###
class Parametrization(ViktorParametrization):
    # Step 1
    step_1 = Step('Step 1 - Generic Information', on_next=validate_step_1)
    # Projectinformatie
    step_1.section_1 = Section('Generic EED Audit information')
    step_1.section_1.client = TextField("Client name")
    step_1.section_1.dataprovider = OptionField(
        "Energy Data Provider",
        options=[
            'Joulz'
        ],
        default='Joulz',
        description="Select the correct Energy Data provider. When a provider is not known, please contact the tool administrator."
    )
    step_1.section_1.language = OptionField(
        "Output language",
        options=[
            'English',
            'Dutch'
        ],
        default='English'
    )
    step_1.section_1.audit_years = TextField(
        "EED Audit Period",
        default='2019-2022',
        description="Please write down the timerange of the current EED Audit range. Use the format [year-year]."
    )
    step_1.section_1.baseyear = IntegerField(
        "EED Audit Base Year",
        default=2022,
        description="Please write down the base year of the current EED Audit."
    )

    step_1.section_1.audit_branches = DynamicArray("Audit branches")
    step_1.section_1.audit_branches.name = TextField("Reference name")
    step_1.section_1.audit_branches.location = AutocompleteField("City", options=options_nl_cities)
    step_1.section_1.audit_branches.openinghour_mon = TextField(
        ui_name = "Opening hours - Monday",
        default = "6-23",
        description = "Please write the opening hour range in format [h-h]. Also include staff hours."
        )
    step_1.section_1.audit_branches.openinghour_tue = TextField(
        ui_name = "Opening hours - Tuesday",
        default = "6-23",
        description = "Please write the opening hour range in format [h-h]. Also include staff hours."
        )
    step_1.section_1.audit_branches.openinghour_wed = TextField(
        ui_name = "Opening hours - Wednesday",
        default = "6-23",
        description = "Please write the opening hour range in format [h-h]. Also include staff hours."
        )
    step_1.section_1.audit_branches.openinghour_thu = TextField(
        ui_name = "Opening hours - Thursday",
        default = "6-23",
        description = "Please write the opening hour range in format [h-h]. Also include staff hours."
        )
    step_1.section_1.audit_branches.openinghour_fri = TextField(
        ui_name = "Opening hours - Friday",
        default = "6-23",
        description = "Please write the opening hour range in format [h-h]. Also include staff hours."
        )
    step_1.section_1.audit_branches.openinghour_sat = TextField(
        ui_name = "Opening hours - Saturday",
        default = "6-22",
        description = "Please write the opening hour range in format [h-h]. Also include staff hours."
        )
    step_1.section_1.audit_branches.openinghour_sun = TextField(
        ui_name = "Opening hours - Sunday",
        default = "6-21",
        description = "Please write the opening hour range in format [h-h]. Also include staff hours."
        )
    step_1.section_1.audit_branches.e_upload = FileField("Electricity Data Export", 
                                                         file_types=['.csv'], 
                                                         max_size=50_000_000, 
                                                         description="Please upload the exported Excel file from your energy data provider. Include electrity- production, consumption and reactive power- production and consumption (blindverbruik in Dutch).")
    step_1.section_1.audit_branches.g_upload = FileField("Gas Data Export", 
                                                         file_types=['.csv'], 
                                                         max_size=50_000_000, 
                                                         description="Please upload the exported Excel file from your energy data provider. Include only the gas consumption data.")
    step_1.section_1.audit_branches.pv_upload = FileField("PV Export", 
                                                         file_types=['.csv'], 
                                                         max_size=50_000_000, 
                                                         description="Please upload the exported Excel file fPom your energy data provider. Include only electricity production data, most likely named as PV datapoints.")
    step_1.section_1.audit_branches.data_years = TextField("Year range",
                                                           default="None",
                                                           visible=False)
    
    # Section 2 - Energy prices
    step_1.section_2 = Section('Energy Price & Emissions Information')
    step_1.section_2.peakhours = TextField(
        ui_name = "Electricity peak rate hours",
        default = "7-23",
        description = "Please write the peakrate hour range in format [h-h]."
        )
    step_1.section_2.standard_prices = BooleanField("Do you want to use custom estimated energy prices [no/yes]?",
                                                    default=False,
                                                    description = "The standard energy prices are based on the IKEA forecast figures for 2023-2027.")
    step_1.section_2.show_emissions = BooleanField("Do you want to use custom CO2-eq emission rates [no/yes]?",
                                                    default=False,
                                                    description = "The standard GHG emission rates (CO2-eq) are extracted from the Klimaatmonitor.")
    
    step_1.section_2.energytable = Table('Energy price table')
    
    _default_content = [
            {'year': '2022', 'peak': 409.71, 'offpeak': 339.97, "gas": 0.677},
            {'year': '2023', 'peak': 396.75, 'offpeak': 313.13, "gas": 0.689},
            {'year': '2024', 'peak': 284.27, 'offpeak': 234.27, "gas": 0.706},
            {'year': '2025', 'peak': 251.71, 'offpeak': 216.71, "gas": 0.749},
            {'year': '2026', 'peak': 267.54, 'offpeak': 230.79, "gas": 1.111},
            {'year': '2027', 'peak': 284.46, 'offpeak': 245.87, "gas": 1.196},
            {'year': '2028', 'peak': 302.55, 'offpeak': 262.04, "gas": 1.287},
            {'year': '2029', 'peak': 318.82, 'offpeak': 276.28, "gas": 1.386},
            {'year': '2030', 'peak': 335.98, 'offpeak': 291.31, "gas": 1.492},
            {'year': '2031', 'peak': 354.08, 'offpeak': 307.18, "gas": 1.567},
            {'year': '2032', 'peak': 373.18, 'offpeak': 323.93, "gas": 1.645},
            {'year': '2033', 'peak': 393.33, 'offpeak': 341.62, "gas": 1.727},
            {'year': '2034', 'peak': 414.60, 'offpeak': 360.30, "gas": 1.814},
            {'year': '2035', 'peak': 437.04, 'offpeak': 380.03, "gas": 1.904},
            {'year': '2036', 'peak': 460.72, 'offpeak': 400.86, "gas": 2.000},
            {'year': '2037', 'peak': 485.71, 'offpeak': 422.86, "gas": 2.100}
        ]
    step_1.section_2.energytable = Table('Energy price table', default=_default_content, visible=Lookup('step_1.section_2.standard_prices'))
    step_1.section_2.energytable.year = TextField("Year")
    step_1.section_2.energytable.peak = NumberField("Electricity Peak rate", suffix='EUR/MWh', num_decimals=2)
    step_1.section_2.energytable.offpeak = NumberField("Electricity Off-Peak rate", suffix='EUR/MWh', num_decimals=2)
    step_1.section_2.energytable.gas = NumberField("Gas rate", suffix='EUR/Nm3', num_decimals=3)

    _default_content = [
        {'year': '2012', 'electricity_emissions': 0.472, 'gas_emissions': 1.788, 'grid_heat_emissions': 35.970},
        {'year': '2013', 'electricity_emissions': 0.481, 'gas_emissions': 1.788, 'grid_heat_emissions': 35.970},
        {'year': '2014', 'electricity_emissions': 0.503, 'gas_emissions': 1.785, 'grid_heat_emissions': 35.970},
        {'year': '2015', 'electricity_emissions': 0.529, 'gas_emissions': 1.788, 'grid_heat_emissions': 35.970},
        {'year': '2016', 'electricity_emissions': 0.495, 'gas_emissions': 1.788, 'grid_heat_emissions': 35.970},
        {'year': '2017', 'electricity_emissions': 0.453, 'gas_emissions': 1.791, 'grid_heat_emissions': 35.970},
        {'year': '2018', 'electricity_emissions': 0.428, 'gas_emissions': 1.791, 'grid_heat_emissions': 35.970},
        {'year': '2019', 'electricity_emissions': 0.369, 'gas_emissions': 1.791, 'grid_heat_emissions': 35.970},
        {'year': '2020', 'electricity_emissions': 0.292, 'gas_emissions': 1.785, 'grid_heat_emissions': 35.970},
        {'year': '2021', 'electricity_emissions': 0.300, 'gas_emissions': 1.785, 'grid_heat_emissions': 35.970},
        {'year': '2022', 'electricity_emissions': 0.272, 'gas_emissions': 1.788, 'grid_heat_emissions': 35.970}
        #source": https://klimaatmonitor.databank.nl/content/co2-uitstoot
    ]

    step_1.section_2.emissionstable = Table('Energy price table', default=_default_content, visible=Lookup('step_1.section_2.show_emissions'))
    step_1.section_2.emissionstable.year = TextField("Year")
    step_1.section_2.emissionstable.electricity_emissions = NumberField("Electricity CO2-eq emission rate", suffix='kg/kWh', num_decimals=3)
    step_1.section_2.emissionstable.gas_emissions = NumberField("Natural CO2-eq emission rate", suffix='kg/kWh', num_decimals=3)
    step_1.section_2.emissionstable.grid_heat_emissions = NumberField("Grid heating CO2-eq emission rate", suffix='kg/kWh', num_decimals=3)

    step_1.section_2.analyzed = BooleanField("Analyzed data?", default=False, visible=False)

    step_1.section_2.perform_analysis_button = SetParamsButton('Analyse Data', method='analyse_data')

    # STEP 2 - Required output graphs
    step_2 = Step('Step 2 - Data Visualizations')
    # Section 2 - Energy profiles
    step_2.section_1 = Section("Output information")
    step_2.section_1.output_reference = OptionField("Select the branch for visualization.",
                                        options=options_of_branch_array)
    step_2.section_1.output_format = OptionField("Output File Format",
                                       options=[
                                           ".png", ".jpg", ".svg", ".pdf"
                                       ],
                                       default=".png")
    step_2.section_1.custom = BooleanField("Use custom export names?",
                                                  default=False)
    
    # Section 2 - Energy profiles
    step_2.section_2 = Section("Energy Profiles")
    _default_content = [
        {'name': None, 'source': "Electricity", 'production': True, 'timeframe': "Day profile per hour", 'repetition': True, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True},
        {'name': None, 'source': "Electricity", 'production': True, 'timeframe': "Week profile per day", 'repetition': True, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True},
        {'name': None, 'source': "Electricity", 'production': True, 'timeframe': "Year profile per month", 'repetition': False, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True},
        {'name': None, 'source': "Electricity", 'production': True, 'timeframe': "Year profile per season", 'repetition': False, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True},
        {'name': None, 'source': "Natural gas", 'production': False, 'timeframe': "Day profile per hour", 'repetition': True, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True},
        {'name': None, 'source': "Natural gas", 'production': False, 'timeframe': "Week profile per day", 'repetition': True, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True},
        {'name': None, 'source': "Natural gas", 'production': False, 'timeframe': "Year profile per month", 'repetition': False, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True},
        {'name': None, 'source': "Natural gas", 'production': False, 'timeframe': "Year profile per season", 'repetition': False, 'comparison': True, 'timerange': None, 'baseline': True, 'averageline': True, 'netline': True}
    ]

    step_2.section_2.visualization_array = DynamicArray("Consumption profiles", default=_default_content)
    step_2.section_2.visualization_array.name = TextField("Graph export name", visible=Lookup("step_2.section_1.custom"))
    step_2.section_2.visualization_array.source = OptionField("Energy type",
                                                  options=["Electricity", "Natural gas"])
    step_2.section_2.visualization_array.production = BooleanField("Include energy production?",
                                                  default=False)
    step_2.section_2.visualization_array.timeframe = OptionField("Timeframe",
                                                  options=["Day profile per hour", "Week profile per day", "Year profile per month", "Year profile per season"])
    step_2.section_2.visualization_array.repetition = BooleanField("Retrieve true values representing average daily or hourly use?",
                                                  default=False)
    step_2.section_2.visualization_array.comparison = BooleanField("EED Audit year comparison",
                                                  default=False,
                                                  description="Do you want to add a comparison of the EED Audit years or do you just want to analyse the latest year?")
    step_2.section_2.visualization_array.timerange = MultiSelectField("Choose comparison years", 
                                                                    options=data_options_of_branch_array,
                                                                    visible=RowLookup('comparison'))
    step_2.section_2.visualization_array.baseline = BooleanField("Reference year baseload line",
                                                  default=False,
                                                  description="Do you want to add a horizontal baseload line of the reference year to the graph?",
                                                  visible=RowLookup('comparison'))
    step_2.section_2.visualization_array.averageline = BooleanField("Reference year average net consumption line",
                                                  default=False,
                                                  description="Do you want to add a horizontal average net consumption line of the reference year to the graph?",
                                                  visible=RowLookup('comparison'))
    step_2.section_2.visualization_array.netline = BooleanField("Reference net consumption line",
                                                  default=False,
                                                  description="Do you want to add the reference net consumption line of the reference year to the graph?",
                                                  visible=RowLookup('comparison'))
    
    _default_content = [
        {'name': None, 'impact': "Cost"},
        {'name': None, 'impact': "Emissions"}
    ]
    
    step_2.section_3 = Section("Consumption Impact Graphs")    
    step_2.section_3.impact_array = DynamicArray("Energy Impact Graphs", default=_default_content)
    step_2.section_3.impact_array.name = TextField("Graph export name", visible=Lookup("step_2.section_1.custom"))
    step_2.section_3.impact_array.impact = OptionField("Impact type",
                                                  options=["Cost", "Emissions"])

    _default_content = [
        {'name': None, 'selection': "Cost projection"},
        {'name': None, 'selection': "Temperature difference"},
        {'name': None, 'selection': "Electricity demand overview"},
        {'name': None, 'selection': "Natural gas demand overview"}
    ]
    
    step_2.section_4 = Section("Custom Graphs")    
    step_2.section_4.custom_array = DynamicArray("Custom Graphs", default=_default_content)
    step_2.section_4.custom_array.name = TextField("Graph export name", visible=Lookup("step_2.section_1.custom"))
    step_2.section_4.custom_array.selection = OptionField("Select custom graph",
                                                  options=["Cost projection", "Temperature difference", "Electricity demand overview", "Natural gas demand overview"])
    
    step_2.section_5 = Section("Process data to graphs")
    step_2.section_5.set_graph_names = SetParamsButton('Save Visualization Input', method='set_graph_name')
    step_2.section_5.create_graphs_button = ActionButton('Create Graphs', method='create_graphs')

    # STEP 3 - Graph output
    step_3 = Step('Step 3 - Output', views=["get_plot_view", "get_table_view"])
    
    step_3.graph_selection = OptionField("Choose graph", 
                                        options=graph_options_of_visualization_array,
                                        default='No graph specified')

    step_3.download_graph = DownloadButton('Download Selected Graph', method='download_graph')
    step_3.download_all_graphs = DownloadButton('Download All Graphs', method='download_all_graphs')

    step_3.download_table = DownloadButton('Download Processed Data', method='download_table')
    step_3.download_overview = DownloadButton('Download Overview Table', method='download_overview')



class Controller(ViktorController):
    label = 'My Entity Type'
    parametrization = Parametrization
    
    def analyse_data(self, params, **kwargs):
        # Load CSV data and process to standarised dataframes
        progress_message("Uploading and analysing files...")
        client = params.step_1.section_1.client
        dataprovider = params.step_1.section_1.dataprovider
        audit_branches = params.step_1.section_1.audit_branches


        # Find correct weather data
        weather_array = []
        branch_counter = 0
        for branch in audit_branches:
            weather_location = find_nearest_city(branch.location)
            if dataprovider == "Joulz":
                e_df, g_df, pv_df = joulz_reader(params, branch)
                e_key = branch.name.replace(" ","_") + "_" + "e_df"
                g_key = branch.name.replace(" ","_") + "_" + "g_df"
                pv_key = branch.name.replace(" ","_") + "_" + "pv_df"
                Storage().set(e_key, data=File.from_data(e_df.to_csv(index=False)), scope='entity')
                Storage().set(g_key, data=File.from_data(g_df.to_csv(index=False)), scope='entity')
                Storage().set(pv_key, data=File.from_data(pv_df.to_csv(index=False)), scope='entity')

                # Find the earliest maximum start date and latest minimum end date
                earliest_max_start = max(df['DateTime'].min() for df in [e_df, g_df])
                latest_min_end = min(df['DateTime'].max() for df in [e_df, g_df])
                # Extract years from the DateTime values
                start_year = earliest_max_start.year
                end_year = latest_min_end.year
                data_years = [year for year in range(start_year + 1, end_year)
                            if pd.Timestamp(year, 12, 31) <= latest_min_end and pd.Timestamp(year, 1, 1) >= earliest_max_start]
                delimiter = ", "
                data_years = delimiter.join(map(str, data_years))

                time_range = pd.date_range(start=earliest_max_start, end=latest_min_end, freq='H')
                df_template = pd.DataFrame({'DateTime': time_range})
                df_template['Date'] = df_template['DateTime'].dt.date
                df_template['Time'] = df_template['DateTime'].dt.time

                branch_counter += 1
            else:
                raise ("Data provider is currently not supported in the EED Audit Data Visualization tool. Please contact the tool administrator.")
            
            openinghours = [getattr(branch, attr) for attr in dir(branch) if attr.startswith('openinghour')]

            climate_df = climate_data(openinghours, weather_location, df_template)
            climate_key = branch.name.replace(" ","_") + "_" + "climate_df"
            Storage().set(climate_key, data=File.from_data(climate_df.to_csv(index=False)), scope='entity')
            
            # weather_array.append({'Reference name': branch.name, 'Weather location': weather_location})
            weather_array.append(weather_location)

        return SetParamsResult({
            "step_1": {
                "section_2": { 
                    "analyzed": True,
                    "weather_array": weather_array
                }
            }
        })
            
    def set_graph_name(self, params, **kwargs):
        # Create graph names based on given input
        client = params.step_1.section_1.client
        years = params.step_1.section_1.audit_years
        language = params.step_1.section_1.language
        branch = params.step_2.section_1.output_reference
        if language == "English":
            lan = "EN_"
        elif language == "Dutch":
            lan = "NL_"
        else:
            lan = ""

        for row in params.step_2.section_2.visualization_array:
            if row.production == True:
                net = "Net"
            else:
                net = ""
            if row.comparison == True:
                time_range = years
            else:
                time_range = str(params.step_1.section_1.baseyear)
                
            info = row.timeframe.replace(" ", "-")
            
            row.name = lan + client + "_" + branch + "_" + "EnergyProfile_" + net + "_" + row.source + "_" + time_range + "_" + info

        for row in params.step_2.section_3.impact_array:
            row.name = lan + client + "_" + branch + "_" + "ImpactGraph_" + row.impact + "_" + years + "_"

        for row in params.step_2.section_4.custom_array:
            if row.selection == "Cost projection":
                graph_name = "CostProjection"
            elif row.selection == "Temperature difference":
                graph_name = "dTemperature-Overview"
            elif row.selection == "Electricity demand overview":
                graph_name = "ElectricityDemand-Overview"
            elif row.selection == "Natural gas demand overview":
                graph_name = "GasDemand-Overview"
            row.name = lan + client + "_" + branch + "_" + graph_name + "_" + years + "_"
            
        return SetParamsResult(params)

    @staticmethod
    def create_graphs(params, **kwargs):
        for row in params.step_2.section_2.visualization_array:
            graph_data = energy_profile_diagram(params, row)
            graph_name = row.name
            # Code to create a static hash
            hash_obj = hashlib.sha256(graph_name.encode())
            key_name = hash_obj.hexdigest()[:64]
            Storage().set(key_name, data=File.from_data(graph_data.getvalue()), scope='entity')

        for row in params.step_2.section_3.impact_array:
            graph_data = energy_impact_graph(params, row)
            graph_name = row.name
            # Code to create a static hash
            hash_obj = hashlib.sha256(graph_name.encode())
            key_name = hash_obj.hexdigest()[:64]
            Storage().set(key_name, data=File.from_data(graph_data.getvalue()), scope='entity')

        for row in params.step_2.section_4.custom_array:
            if row.selection == "Cost projection":
                graph_data = cost_projection_graph(params, row)
            else:
                graph_data = overview_graphs(params, row)
            graph_name = row.name
            # Code to create a static hash
            hash_obj = hashlib.sha256(graph_name.encode())
            key_name = hash_obj.hexdigest()[:64]
            Storage().set(key_name, data=File.from_data(graph_data.getvalue()), scope='entity')

        return 
    
    @ImageView("Visualization Results", duration_guess=1)
    def get_plot_view(self, params, **kwargs):
        if params.step_3.graph_selection == "No graph specified":
            raise UserError('Select a graph to visualize')
        graphs = zip_longest(
            params.step_2.section_2.visualization_array,
            params.step_2.section_3.impact_array,
            params.step_2.section_4.custom_array,
            fillvalue=None
        )
        for diagram_1, diagram_2, diagram_3 in graphs:
            if diagram_1 and diagram_1.name == params.step_3.graph_selection:
                # Code to create a static hash
                graph_name = diagram_1.name
                hash_obj = hashlib.sha256(graph_name.encode())
                key_name = hash_obj.hexdigest()[:64]
                graph_file = Storage().get(key_name, scope='entity')
                with graph_file.open_binary() as file:
                    graph_data = file.read()
                image_data = BytesIO(graph_data)  
                return ImageResult(image_data)
            
            elif diagram_2 and diagram_2.name == params.step_3.graph_selection:
                # Code to create a static hash
                graph_name = diagram_2.name
                hash_obj = hashlib.sha256(graph_name.encode())
                key_name = hash_obj.hexdigest()[:64]
                graph_file = Storage().get(key_name, scope='entity')
                with graph_file.open_binary() as file:
                    graph_data = file.read()
                image_data = BytesIO(graph_data)  
                return ImageResult(image_data)
            
            elif diagram_3 and diagram_3.name == params.step_3.graph_selection:
                # Code to create a static hash
                graph_name = diagram_3.name
                hash_obj = hashlib.sha256(graph_name.encode())
                key_name = hash_obj.hexdigest()[:64]
                graph_file = Storage().get(key_name, scope='entity')
                with graph_file.open_binary() as file:
                    graph_data = file.read()
                image_data = BytesIO(graph_data)  
                return ImageResult(image_data)
    
    
    @PlotlyView("Result Table", duration_guess=1)
    def get_table_view(self, params, **kwargs):           
        # Initial assignments
        section_1 = params.step_1.section_1
        client = section_1.client
        branch_name = params.step_2.section_1.output_reference
        year = section_1.baseyear

        df_E, df_G = energy_table(params)

        df, temp = overview_table(df_E, df_G)

        # Create the Plotly table
        table = go.Figure(data=[go.Table(
            header=dict(values=list(df.columns),
                        fill_color='paleturquoise',
                        align='left'),
            cells=dict(values=[df[col] for col in df.columns],
                    fill_color='lavender',
                    align='left'))
        ])

        table.update_layout(title="EED Energy Overview - {} {} ({})".format(client, branch_name, year))
        # table.update_layout({'margin':{'t':50}})

        return PlotlyResult(table.to_json())

    def download_all_graphs(self, params, **kwargs):
        zipped_files = {}
        data_format = params.step_2.section_1.output_format
        for row in params.step_2.section_2.visualization_array:
            graph_name = row.name
            hash_obj = hashlib.sha256(graph_name.encode())
            key_name = hash_obj.hexdigest()[:64]

            graph_file = Storage().get(key_name, scope='entity')
            with graph_file.open_binary() as file:
                graph_data = file.read()
            image_data = BytesIO(graph_data)            
            download_name = graph_name + data_format
            zipped_files[download_name] = image_data
        for row in params.step_2.section_3.impact_array:
            graph_name = row.name
            hash_obj = hashlib.sha256(graph_name.encode())
            key_name = hash_obj.hexdigest()[:64]

            graph_file = Storage().get(key_name, scope='entity')
            with graph_file.open_binary() as file:
                graph_data = file.read()
            image_data = BytesIO(graph_data)            
            download_name = graph_name + data_format
            zipped_files[download_name] = image_data
        
        for row in params.step_2.section_4.custom_array:
            graph_name = row.name
            hash_obj = hashlib.sha256(graph_name.encode())
            key_name = hash_obj.hexdigest()[:64]

            graph_file = Storage().get(key_name, scope='entity')
            with graph_file.open_binary() as file:
                graph_data = file.read()
            image_data = BytesIO(graph_data)            
            download_name = graph_name + data_format
            zipped_files[download_name] = image_data

        return DownloadResult(zipped_files=zipped_files, file_name='EED_audit_graphs.zip')
    
    def download_graph(self, params, **kwargs):
        data_format = params.step_2.section_1.output_format
        for row in params.step_2.section_2.visualization_array:
            if row.name == params.step_3.graph_selection:
                graph_name = row.name
                hash_obj = hashlib.sha256(graph_name.encode())
                key_name = hash_obj.hexdigest()[:64]

                graph_file = Storage().get(key_name, scope='entity')
                with graph_file.open_binary() as file:
                    graph_data = file.read()
                image_data = BytesIO(graph_data)            
                download_name = graph_name + data_format

        for row in params.step_2.section_3.impact_array:
            if row.name == params.step_3.graph_selection:
                graph_name = row.name
                hash_obj = hashlib.sha256(graph_name.encode())
                key_name = hash_obj.hexdigest()[:64]

                graph_file = Storage().get(key_name, scope='entity')
                with graph_file.open_binary() as file:
                    graph_data = file.read()
                image_data = BytesIO(graph_data)            
                download_name = graph_name + data_format

        for row in params.step_2.section_4.custom_array:
            if row.name == params.step_3.graph_selection:
                graph_name = row.name
                hash_obj = hashlib.sha256(graph_name.encode())
                key_name = hash_obj.hexdigest()[:64]

                graph_file = Storage().get(key_name, scope='entity')
                with graph_file.open_binary() as file:
                    graph_data = file.read()
                image_data = BytesIO(graph_data)            
                download_name = graph_name + data_format
                
                return DownloadResult(image_data, file_name=download_name)
        
    def download_table(self, params, **kwargs):
        branch_name = params.step_2.section_1.output_reference
        zipped_files = {}
        data_format = ".csv"
        dataframe_list = ["e_df", "g_df", "pv_df"]
        download_name = 'EED_audit_data_{}.zip'.format(branch_name)

        for df_name in dataframe_list:
            storage_key = branch_name.replace(" ","_") + "_" + df_name
            data = Storage().get(storage_key, scope='entity')
            
            with data.open_binary() as file:
                file_contents = file.read()
            
                # Check if the file is empty or contains only whitespace
                if not file_contents.strip():
                    continue  # Skip this iteration and move to the next dataframe
            
                df = pd.read_csv(BytesIO(file_contents))

            
            # Create a BytesIO object to store the CSV data
            csv_buffer = BytesIO()

            # Write the DataFrame to the BytesIO object as CSV
            df.to_csv(csv_buffer, index=False)

            csv_filename = branch_name.replace(" ", "_") + "_" + df_name + data_format
            zipped_files[csv_filename] = csv_buffer

        return DownloadResult(zipped_files=zipped_files, file_name=download_name)
    
    def download_overview(self, params, **kwargs):           
        # Initial assignments
        section_1 = params.step_1.section_1
        client = section_1.client
        branch_name = params.step_2.section_1.output_reference
        year = section_1.baseyear

        df_E, df_G = energy_table(params)
        temp, df = overview_table(df_E, df_G)

        # Create a BytesIO object to store the CSV data
        csv_buffer = BytesIO()

        # Write the DataFrame to the BytesIO object as CSV
        df.to_csv(csv_buffer, index=False)

        download_name = "EED_Energy_Overview-{}_{}_{}.csv".format(client, branch_name, year)
        return DownloadResult(csv_buffer, download_name)
        