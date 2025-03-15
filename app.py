from math import e
import webbrowser
import json
import requests
from darts.models import RNNModel
from darts.timeseries import TimeSeries
from darts.dataprocessing.transformers import Scaler
import io
import base64
import folium
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from flask import Flask, request, render_template, redirect, url_for,jsonify
from polyline import decode
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from geopy.distance import geodesic
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from apyori import apriori  # This is the correct import for the Apriori algorithm

CSV_PATH = "fedex-main\combined_inventory_dataset.csv"
def load_data():
    """Load and validate dataset"""
    try:
        df = pd.read_csv(CSV_PATH)
        df['ExpiryDate'] = pd.to_datetime(df['ExpiryDate'], errors='coerce')
        
        required_cols = ['ItemName', 'Category', 'Quantity', 
                        'StorageCapacity(kg)', 'Destination']
        if not all(col in df.columns for col in required_cols):
            missing = [col for col in required_cols if col not in df.columns]
            raise ValueError(f"Missing columns: {', '.join(missing)}")
            
        return df
    except Exception as e:
        raise RuntimeError(f"Data loading failed: {str(e)}")

def calculate_weight(quantity, capacity):
    return quantity * capacity

def allocate_vehicles(df):
    """Vehicle allocation logic"""
    results = []
    
    # Process perishables
    perishables = df[
        (df['Category'].isin(['Food', 'Dairy'])) & 
        (df['ExpiryDate'].notna())
    ].copy().sort_values('ExpiryDate')
    
    for _, row in perishables.iterrows():
        weight = calculate_weight(row['Quantity'], row['StorageCapacity(kg)'])
        vehicle = 'refrigerated_18_wheeler' if weight > 500 else 'refrigerated_truck'
        results.append({
            'item': row['ItemName'],
            'weight': weight,
            'destination': row['Destination'],
            'vehicle': vehicle,
            'type': 'Special'
        })
    
    # Process non-perishables
    non_perish = df[~df.index.isin(perishables.index)]
    for _, row in non_perish.iterrows():
        weight = calculate_weight(row['Quantity'], row['StorageCapacity(kg)'])
        vehicle = allocate_regular_vehicle(weight)
        results.append({
            'item': row['ItemName'],
            'weight': weight,
            'destination': row['Destination'],
            'vehicle': vehicle,
            'type': 'Regular'
        })
    
    return results

# Load your trained LSTM model
model = RNNModel.load(r'C:\Users\aniru\Downloads\fedex-main (6)\fedex-main (5)\fedex-main (4)\fedex-main\fedex-main\lstm_model_weights.pth')

# Load scalers (assuming you saved them separately)
scaler1 = Scaler()  # For target variable
scaler2 = Scaler()  # For covariates

# --- Helper functions ---
def create_future_covariates(future_df, series_start_time, series_freq_str):
    """Creates future covariates with datetime attributes."""
    X = future_df.iloc[:, 1:]  # Assuming 'Date' is the first column
    
    year_series = datetime_attribute_timeseries(
        pd.date_range(start=series_start_time,
                      freq=series_freq_str,
                      periods=X.shape[0]),
        attribute="year",
        one_hot=False)
    month_series = datetime_attribute_timeseries(year_series, attribute="month", one_hot=True)
    weekday_series = datetime_attribute_timeseries(year_series, attribute="weekday", one_hot=True)

    covariates = TimeSeries.from_dataframe(X)
    covariates = covariates.stack(year_series)
    covariates_transformed = scaler2.transform(covariates)  # Use the loaded scaler
    covariates_transformed = covariates_transformed.stack(month_series)
    covariates_transformed = covariates_transformed.stack(weekday_series)

    return covariates_transformed

def plot_forecast(train_data, forecast_data):
    """Generates and encodes the train vs. forecast plot."""
    plt.figure(figsize=(10, 4))
    plt.plot(train_data, label='Train')
    plt.plot(forecast_data, label='Forecast')
    plt.title("Train and Forecast")
    plt.legend()

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    plt.close()
    
    return plot_url

def allocate_regular_vehicle(weight):
    if weight <= 10: return 'motorcycle'
    elif weight <= 50: return 'car'
    elif weight <= 100: return 'minivan'
    elif weight <= 500: return 'truck'
    else: return '18_wheeler'

def process_report_data(results):
    """Structure data for template rendering"""
    report = {
        'special': {},
        'regular': {},
        'routes': {},
        'stats': {
            'total_items': 0,
            'total_weight': 0,
            'special_count': 0,
            'regular_count': 0
        }
    }
    
    for item in results:
        dest = item['destination']
        veh_type = item['vehicle']
        report['stats']['total_items'] += 1
        report['stats']['total_weight'] += item['weight']
        
        if item['type'] == 'Special':
            report['special'].setdefault(dest, {})
            report['special'][dest][veh_type] = report['special'][dest].get(veh_type, 0) + 1
            report['stats']['special_count'] += 1
        else:
            report['regular'].setdefault(dest, {})
            report['regular'][dest][veh_type] = report['regular'][dest].get(veh_type, 0) + 1
            report['stats']['regular_count'] += 1
        
        report['routes'].setdefault(dest, []).append(item)
    
    return report

# Replace with your API keys
TOMTOM_API_KEY = "kkmlX7oWaAfNqs7Iyz6RTLBJTfNXWpKH"
WEATHERBIT_API_KEY = "86e22791c10d480eaf5ab94337fde24d"

# Function to get coordinates using geopy
def get_coordinates(city_name):
    geolocator = Nominatim(user_agent="city_route_locator")
    geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1)
    location = geocode(city_name)
    if location:
        return location.latitude, location.longitude
    else:
        print(f"City '{city_name}' not found.")
        return None

# Function to get weather forecast for coordinates using Weatherbit API
import time


def predict_co2_emissions(num_gears, transmission_type, engine_size, fuel_type, cylinders, fuel_consumption_comb):

    loaded_model = joblib.load('C:/Users/aniru/Downloads/fedex-main (6)/fedex-main (5)/fedex-main (4)/fedex-main/fedex-main/model_final.pkl')


    transmission_type_mapping = {'A': 0, 'AM': 1, 'AS': 2, 'AV': 3, 'M': 4}
    fuel_type_mapping = {'D': 0, 'E': 1, 'N': 2, 'X': 3, 'Z': 4}

    # Get encoded values from user input
    transmission_type_encoded = transmission_type_mapping.get(transmission_type)
    fuel_type_encoded = fuel_type_mapping.get(fuel_type)
    # Create input array for the model
    input_data = [[num_gears, transmission_type_encoded, engine_size, fuel_type_encoded, cylinders, fuel_consumption_comb]]
    input_data = pd.DataFrame(input_data, columns=['Number of Gears', 'Transmission Type', 'Engine Size', 'Fuel Type', 'Cylinders', 'Fuel Consumption Comb'])

    # Make prediction
    prediction = loaded_model.predict(input_data)

    # Print prediction
    print("Predicted CO2 Emissions:", prediction[0])
    return prediction[0]


# Modified version of get_weather_data with rate-limiting
def get_weather_data(coords, retries=5, delay=1):
    weather_url = "https://api.weatherbit.io/v2.0/forecast/daily"
    params = {
        "lat": coords[0],
        "lon": coords[1],
        "key": WEATHERBIT_API_KEY,
        "days": 1  # Fetch the next 1 day of forecast
    }

    for attempt in range(retries):
        try:
            response = requests.get(weather_url, params=params)
            response.raise_for_status()  # Raise exception for 4xx or 5xx responses

            data = response.json()
            if "data" in data:
                weather_info = []
                for forecast in data["data"]:
                    timestamp = forecast["datetime"]
                    temperature = forecast["temp"]
                    rain = forecast.get("precip", 0)  # Precipitation in mm
                    rain_percentage = (rain / 10) * 100  # Approximation for rain percentage
                    weather_info.append({
                        "timestamp": timestamp,
                        "temperature": temperature,
                        "rain_percentage": rain_percentage  # Convert rain to percentage
                    })
                return weather_info
            else:
                print("No weather data found.")
                return None
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 429:  # Too many requests
                print(f"Rate limited. Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponentially increase delay
            else:
                print(f"Error fetching weather data: {e}")
                return None

    print("Exceeded retry attempts.")
    return None

"""def get_weather_data(coords):
    weather_url = "https://api.weatherbit.io/v2.0/forecast/daily"
    params = {
        "lat": coords[0],
        "lon": coords[1],
        "key": WEATHERBIT_API_KEY,
        "days": 1  # Fetch the next 2 days of forecast
    }
    try:
        response = requests.get(weather_url, params=params)
        response.raise_for_status()  # Raise exception for 4xx or 5xx responses
        data = response.json()
        if "data" in data:
            weather_info = []
            for forecast in data["data"]:
                timestamp = forecast["datetime"]
                temperature = forecast["temp"]
                rain = forecast.get("precip", 0)  # Precipitation in mm
                rain_percentage = (rain / 10) * 100  # Approximation for rain percentage
                weather_info.append({
                    "timestamp": timestamp,
                    "temperature": temperature,
                    "rain_percentage": rain_percentage  # Convert rain to percentage
                })
            return weather_info
        else:
            print("No weather data found.")
            return None
    except Exception as e:
        print(f"Error fetching weather data: {e}")
        return None"""

weather_cache = {}

def get_weather_data_cached(coords):
    if coords in weather_cache:
        return weather_cache[coords]

    weather_data = get_weather_data(coords)  # Original API call
    weather_cache[coords] = weather_data
    return weather_data


# Function to calculate rain delay based on probability
def calculate_rain_delay(rain_probability, base_delay=8):
    """
    Calculate delay based on rain probability.
    - base_delay: Base delay in minutes for rain > 50%.
    - Returns additional delay in minutes.
    """
    if rain_probability > 0.5:
        if rain_probability > 0.9:
            print("Plane stalled in sky due to heavy bad weather")
            return base_delay * 5  # Severe rain delay
        elif rain_probability > 0.8:
            print("Plane stalled in sky due to heavy bad weather")
            return base_delay * 4
        elif rain_probability > 0.7:
            print("Plane slowing down due to bad weather")
            return base_delay * 3
        elif rain_probability > 0.6:
            print("Plane slowing down due to bad weather")
            return base_delay * 2
        else:  # 0.5 < probability <= 0.6
            print("Plane slightly taking rounds due to bad weather")
            return base_delay
    return 0  # No delay if rain probability <= 0.5


# Function to calculate route using OSRM
def get_osrm_route(start_coords, end_coords):
    osrm_url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&alternatives=true"
    try:
        response = requests.get(osrm_url)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            routes = []
            for route in data["routes"]:
                geometry = route["geometry"]
                decoded_geometry = decode(geometry)  # Decode polyline to lat-lon pairs
                distance = route["distance"] / 1000  # Convert meters to kilometers
                duration = route["duration"] / 60  # Convert seconds to minutes
                routes.append((decoded_geometry, distance, duration))
            return routes
        else:
            print("No routes found from OSRM.")
            return []
    except Exception as e:
        print(f"Error fetching route from OSRM: {e}")
        return []
def get_osrm_walking_route(start_coords, end_coords):
    osrm_url = f"https://routing.openstreetmap.de/routed-foot/route/v1/foot/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&alternatives=true&continue_straight=false&annotations=true&steps=true"
    try:
        response = requests.get(osrm_url)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            routes = []
            for route in data["routes"]:
                geometry = route["geometry"]
                decoded_geometry = decode(geometry)  # Decode polyline to lat-lon pairs
                distance = route["distance"] / 1000  # Convert meters to kilometers
                duration = route["duration"] / 60  # Convert seconds to minutes
                routes.append((decoded_geometry, distance, duration))
            return routes
        else:
            print("No walking routes found from OSRM.")
            return []
    except Exception as e:
        print(f"Error fetching walking route from OSRM: {e}")
        return []
def get_osrm_bike_route(start_coords, end_coords):
    osrm_url = f"https://routing.openstreetmap.de/routed-bike/route/v1/bike/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&alternatives=true&continue_straight=false&annotations=true&steps=true"
    try:
        response = requests.get(osrm_url)
        response.raise_for_status()
        data = response.json()
        if data.get("routes"):
            routes = []
            for route in data["routes"]:
                geometry = route["geometry"]
                decoded_geometry = decode(geometry)  # Decode polyline to lat-lon pairs
                distance = route["distance"] / 1000  # Convert meters to kilometers
                duration = route["duration"] / 60  # Convert seconds to minutes
                routes.append((decoded_geometry, distance, duration))
            return routes
        else:
            print("No walking routes found from OSRM.")
            return []
    except Exception as e:
        print(f"Error fetching walking route from OSRM: {e}")
        return []
# Function to fetch traffic flow data from TomTom
def get_traffic_color(coords):
    traffic_url = "https://api.tomtom.com/traffic/services/4/flowSegmentData/absolute/10/json"
    params = {
        "key": TOMTOM_API_KEY,
        "point": f"{coords[0]},{coords[1]}"
    }
    try:
        response = requests.get(traffic_url, params=params)
        response.raise_for_status()
        data = response.json()
        if "flowSegmentData" in data:
            free_flow_speed = data["flowSegmentData"]["freeFlowSpeed"]
            current_speed = data["flowSegmentData"]["currentSpeed"]
            congestion = free_flow_speed - current_speed
            # Determine color based on congestion
            if congestion > 15:
                return "red"
            elif 5 < congestion <= 15:
                return "orange"
            else:
                return "blue"
        return "blue"  # Default color if no data
    except Exception as e:
        print(f"Error fetching traffic data: {e}")
        return "blue"  # Default color if error occurs

# Function to find the nearest airport to given coordinates
def find_nearest_airport(coords, airport_data):
    airport_data["distance"] = airport_data.apply(
        lambda row: geodesic((row["latitude_deg"], row["longitude_deg"]), coords).km, axis=1
    )
    nearest_airport = airport_data.loc[airport_data["distance"].idxmin()]
    return nearest_airport["name"], (nearest_airport["latitude_deg"], nearest_airport["longitude_deg"]), nearest_airport["distance"]

def get_vehicle_type():
    """
    Prompt the user for a vehicle type and return it.
    The user can choose between different vehicle types.
    """
    valid_vehicle_types = [
        "bike", "cargo_van", "minivan", "small_truck", "heavy_duty_truck", "18_wheeler", "car", "plane"
    ]

    # Prompt the user to select a vehicle type
    vehicle_type = input("Enter vehicle type (bike, cargo_van, minivan, small_truck, heavy_duty_truck, 18_wheeler, car, plane): ").lower()

    # Validate the user input
    while vehicle_type not in valid_vehicle_types:
        print("Invalid vehicle type. Please select from the following options:")
        print(", ".join(valid_vehicle_types))
        vehicle_type = input("Enter vehicle type: ").lower()

    return vehicle_type
# Function to display the route with traffic data on a map

def display_route_with_traffic(route_map, routes, traffic_colors, air_segments, location_airport_segments, vehicle_emission,tf,weather_info=None):
    # Add road segments
    total_co2_emission11 = 0.0
    for route, colors in zip(routes, traffic_colors):
        for i in range(len(route) - 1):
            segment_coords = [route[i], route[i + 1]]
            segment_distance = geodesic(route[i], route[i + 1]).km
            #vehicle_type=vehicle
            # Check weather and adjust color (green for good, red for heavy rain)
            if weather_info:
                rain_percentage = weather_info["rain_percentage"]
                line_color = 'green' if rain_percentage < 0.4 else 'red'
            else:
                line_color = colors[i]  # Default to traffic color if no weather data
            
            co2_emissions = calculate_co2_emissions1(segment_distance, tf, vehicle_emission)
            total_co2_emission11 +=  co2_emissions
            print(f"CO2 emissions for this route segment: {co2_emissions:.2f} kg")
            


            folium.PolyLine(
                locations=segment_coords, color=line_color, weight=5, opacity=0.7
            ).add_to(route_map)
          

    # Add air travel segments
    for air_segment in air_segments:
        folium.PolyLine(
            locations=air_segment, color="red", weight=5, opacity=0.7, tooltip="Air Travel"
        ).add_to(route_map)

    # Add location-to-airport connections
    for segment in location_airport_segments:
        folium.PolyLine(
            locations=segment["coords"], color="green", weight=2, opacity=0.7, tooltip=f"{segment['label']}"
        ).add_to(route_map)
    return total_co2_emission11

def traffic_factor(traffic_colors):
    total_r,total_o,total_b=0,0,0
    length = len(traffic_colors)
    for color in traffic_colors:
        if color=="red":
            total_r+=1
        elif color=="orange":
            total_o+=1
        elif color=="blue":
            total_b+=1
    return (total_r*2.5 + total_o*1.5+total_b*1)/length





def calculate_co2_emissions(distance_km,traffic_factor,vehicle_type="car"):
    print(vehicle_type)
    """
    Calculate CO2 emissions based on the distance and vehicle type.

    Args:
    - distance_km (float): The distance in kilometers.
    - vehicle_type (str): The type of vehicle.

    Returns:
    - float: The CO2 emissions in kilograms.
    """
    # CO2 emission factors for different vehicle types (in kg per km)
    vehicle_emission_factors = {
        "bike": 0.005,  # Minimal emissions for a bike
        "cargo_van": 0.295,  # Approx 295g CO2 per km for cargo van
        "minivan": 0.198,  # Approx 198g CO2 per km for minivan
        "truck": 0.311,  # Approx 311g CO2 per km for small truck
        "heavy_duty_truck": 0.768,  # Approx 768g CO2 per km for heavy-duty truck
        "18_wheeler": 1.0,  # Approx 1000g CO2 per km for an 18-wheeler
        "car": 0.120,  # Approx 120g CO2 per km for a car
        "plane": 2.5,  # Approx 2.5 kg CO2 per km for the entire plane
    }


    #if vehicle_type not in vehicle_emission_factors:
    #   raise ValueError(f"Unsupported vehicle type: {vehicle_type}")

    co2_per_km = vehicle_emission_factors[vehicle_type]
    co2_emissions = distance_km * co2_per_km * traffic_factor # CO2 in kilograms
    return co2_emissions
def calculate_co2_emissions1(distance,traffic_factor,vehicle_emission):
    co2_emissions = 0.0
    co2_emissions = distance * vehicle_emission * traffic_factor*0.001
    return co2_emissions

# def get_vehicle_type(cargo_weight):
#     if cargo_weight < 10:
#         v = input("Motocycle/car/minivan/truck/18-wheeler")
#     elif cargo_weight < 50:
#         v = input("car/minivan/truck/18-wheeler")
#     elif cargo_weight < 100:
#         v = input("minivan/truck/18-wheeler")
#     elif cargo_weight < 500:
#         v = input("truck/18-wheeler")
#     else:
#         v = "18_wheeler"
#     return v

def get_vehicle_airports(cargo_weight):
    if cargo_weight < 10:
        return "motorcycle"
    elif cargo_weight < 50:
        return "car"
    elif cargo_weight < 100:
        return "minivan"
    elif cargo_weight < 500:
        return "truck"
    else:

        return "18_Wheeler"
def calculate_co2_emissions_air(distance):
    emissions = distance*2.5
    return emissions


# Haversine formula to calculate the great-circle distance between two points
def haversine(coords1,coords2):
    R = 6371  # Earth radius in kilometers
    lat1, lon1, lat2, lon2 = map(np.radians, [coords1[0], coords1[1], coords2[0], coords2[1]])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0)*2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)*2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c

# Function to find the nearest airport
va=[]
def find_next_airport(coords, df,va,tolerance=0.01):
    """
    Finds the nearest airport to the specified coordinates.

    Parameters:
        lat (float): Latitude of the input location.
        lon (float): Longitude of the input location.
        df (DataFrame): The dataset containing airport information.
        tolerance (float): A small value to exclude the input airport based on coordinates.

    Returns:
        dict: Information about the nearest airport.
    """
    # Calculate distances to all airports
    distances = haversine(coords, (df["latitude_deg"], df["longitude_deg"]))

    # Exclude the input airport by ensuring a tolerance for coordinates
    same_airport_mask = (
        (np.abs(df["latitude_deg"] - coords[0]) < tolerance) &
        (np.abs(df["longitude_deg"] - coords[1]) < tolerance)
    )
    distances[same_airport_mask] = np.inf  # Set distance to self as infinity

    for tup in va:
      visited_airport_mask = (
        (np.abs(df["latitude_deg"] - coords[0]) < tolerance) &
        (np.abs(df["longitude_deg"] - coords[1]) < tolerance)
    )
    distances[visited_airport_mask] = np.inf


    # Find the index of the nearest airport
    nearest_index = distances.idxmin()
    nearest_airport = df.loc[nearest_index]

    return nearest_airport["name"],  (nearest_airport["latitude_deg"], nearest_airport["longitude_deg"])

def generate_points_between_coordinates(coord1, coord2, num_points=8):
    """
    Generate equally spaced points between two coordinates.

    :param coord1: Tuple (x1, y1) representing the starting coordinate.
    :param coord2: Tuple (x2, y2) representing the ending coordinate.
    :param num_points: Number of points to generate between the coordinates.
    :return: List of tuples representing the generated points.
    """
    x1, y1 = coord1
    x2, y2 = coord2
    return [
        (x1 + i * (x2 - x1) / (num_points + 1), y1 + i * (y2 - y1) / (num_points + 1))
        for i in range(1, num_points + 1)
    ]
def get_aqi_for_location(tuple1, api_key):
    print(tuple1)
    url = f"http://api.waqi.info/feed/geo:{tuple1[0]};{tuple1[1]}/?token={api_key}"
    response = requests.get(url)
    data = response.json()
    if data['status'] == 'ok':
        return data['data']['aqi']  # Return the AQI value
    else:
        return None
def smooth_reroute_path(start, end, affected_points, reroute_offset=0.3):
    waypoints = generate_waypoints(start, end, num_points=20)

    # Separate latitudes and longitudes
    lats, lons = zip(*waypoints)

    # Apply offset to affected waypoints
    for i, point in enumerate(waypoints):
        if point in affected_points:
            lats = list(lats)
            lons = list(lons)
            lats[i] += reroute_offset  # Adjust latitude smoothly
            lons[i] += reroute_offset / 2  # Adjust longitude smoothly

    # Use interpolation to create a smooth reroute
    interp_lat = interp1d(range(len(lats)), lats, kind='cubic')
    interp_lon = interp1d(range(len(lons)), lons, kind='cubic')

    smooth_waypoints = list(zip(interp_lat(np.linspace(0, len(lats) - 1, len(lats))),
                                interp_lon(np.linspace(0, len(lons) - 1, len(lons)))))

    return smooth_waypoints

# Function to reroute air path based on weather conditions
def reroute_air_path(start, end, api_key):
    waypoints = generate_points_between_coordinates(start, end, num_points=8)
    safe_waypoints = []
    affected_points = []

    for wp in waypoints:
        weather_inf = get_weather_data(wp)
        if weather_inf:
          for forecast in weather_inf:
            rain_probability = forecast["rain_percentage"] / 100

        if rain_probability is not None and rain_probability >  0.1:
            print(f"High rain detected at {wp} ({rain_probability}%), rerouting...")
            affected_points.append(wp)
        else:
            safe_waypoints.append(wp)

    # Smooth the rerouted path if bad weather is found
    if affected_points:
        return smooth_reroute_path(start, end, affected_points)
    else:
        return waypoints

# Function to plot the route on a Folium map


# Function to get coordinates of a city using Geopy
def generate_waypoints(start, end, num_points=10):
    lat_points = np.linspace(start[0], end[0], num_points)
    lon_points = np.linspace(start[1], end[1], num_points)
    return list(zip(lat_points, lon_points))
from flask import Flask, request, render_template
from markupsafe import Markup
import pandas as pd
from geopy.distance import geodesic
import folium
app = Flask(__name__)

@app.route('/plan_route', methods=['POST'])
def plan_route():
    with app.app_context():
        start_city = request.form['startLocation']
        end_city = request.form['endLocation']
        cargo_weight = int(request.form['cargoWeight'])  # Convert to integer

        stop_loc=request.form['stops']

        # Vehicle details at start
        vehicle_input = request.form['vehicle']
        num_gears_start = request.form['numGears']
        transmission_type_start = request.form['transmissionType']
        engine_size_start = request.form['engineSize']
        fuel_type_start = request.form['fuelType']
        cylinders_start = request.form['cylinders']
        fuel_consumption_start = request.form['fuelConsumption']
        air_travel_start = request.form['airTravel']


        # Collect stops
        stops = []
        stop_index = 0
        while f'stopLocation{stop_index}' in request.form:
            stop_location = request.form[f'stopLocation{stop_index}']
            ret_drop = request.form[f'ret_drop{stop_index}']
            drop_off_weight = request.form[f'dropOffWeight{stop_index}']
            vehic_input = request.form[f'VehicleInput{stop_index}']
            num_gears = request.form[f'numGears{stop_index}']
            transmission_type = request.form[f'transmissionType{stop_index}']
            engine_size = request.form[f'engineSize{stop_index}']
            fuel_type = request.form[f'fuelType{stop_index}']
            cylinders = request.form[f'cylinders{stop_index}']
            fuel_consumption = request.form[f'fuelConsumption{stop_index}']

            stops.append({
                'location': stop_location,
                'ret_drop': ret_drop,
                'drop_off_weight': drop_off_weight,
                'vehicle_input': vehic_input,
                'num_gears': num_gears,
                'transmission_type': transmission_type,
                'engine_size': engine_size,
                'fuel_type': fuel_type,
                'cylinders': cylinders,
                'fuel_consumption': fuel_consumption
            })
            stop_index += 1

        # Call the main function with the provided parameters
        route_map = main(start_city, end_city,stop_loc,vehicle_input, stops, cargo_weight, 
                        num_gears_start, transmission_type_start, 
                        engine_size_start, fuel_type_start, 
                        cylinders_start, fuel_consumption_start, 
                        air_travel_start,ret_drop)

        if route_map:
            # Render the map as HTML
            return render_template('final_page.html', map=Markup(route_map.repr_html()))
        else:
            return "Error generating route", 500
def final_page():
    return render_template('final_page.html')
@app.route('/')
def index():
    return render_template('about.html')
@app.route("/route")
def route_page():
    return render_template("routes.html")
# Main function

def main(start_city, end_city, stop_loc, vehicle_input,stops_data, cargo_weight, num_gears, transmission_type, engine_size, fuel_type, cylinders, fuel_consumption_comb, air_travel_start,ret_drop):
    # Input start, destination, and optional stops
    html_content="""<html>
    <head><title>Outputs</title></head>
    <body>
    <h1>Output for routes:</h1>"""
    print(start_city)
    stops = stop_loc.split(",")
    vehicle_type_road = vehicle_input
    co2_for_vehicle = predict_co2_emissions(num_gears, transmission_type, engine_size, fuel_type, cylinders, fuel_consumption_comb)


    stops = [stop.strip() for stop in stops if stop.strip()]

    start_coords = get_coordinates(start_city)
    end_coords = get_coordinates(end_city)
    stop_coords = [get_coordinates(stop) for stop in stops]


    # Load airport data for air travel logic
    airport_data = pd.read_csv(r"C:\Users\aniru\Downloads\fedex-main (6)\fedex-main (5)\fedex-main (4)\fedex-main\fedex-main\airports.csv")
    airport_data = airport_data[airport_data['type'].str.contains("airport", case=False)]


    if start_coords and end_coords and all(stop_coords):
        # Combine all points into a full route (start -> stops -> end)
        full_route = [start_coords] + stop_coords + [end_coords]
        air_segments = []
        location_airport_segments = []

        # Initialize map
        route_map = folium.Map(location=start_coords, zoom_start=7)
        
        # Iterate through each segment in the route
        N=1
        U=1
        if vehicle_type_road == "car" or vehicle_type_road == "motorcycle":
            U=1
        elif vehicle_type_road == "minivan":
            U=2
        else:
            U=3

        for i in range(len(full_route) - 1):
            segment_start = full_route[i]
            segment_end = full_route[i + 1]
            if U==3:
                all_routes = get_osrm_route(segment_start, segment_end)
            elif U==2:
                all_routes = get_osrm_bike_route(segment_start, segment_end)
            else:
                all_routes = get_osrm_walking_route(segment_start, segment_end)

            
            if all_routes:
                
                x=1
                print(N)
                if N > 1:
                    if N-1 < len(stops_data):  # Check if N-1 is within the valid range
                        print("Getting inputs for stop", N)
                        stop_name = stops_data[N-1].get("location", "Unknown")
                        drop_off = stops_data[N-1].get('drop_off_weight', 0)

                        try:
                            ret_drop = ret_drop
                            if(ret_drop=="return"):
                                idropoff1 = int(drop_off)
                                icargo_weight += idropoff1
                            else:
                                idropoff = int(drop_off)
                                icargo_weight -= idropoff
                        except ValueError:
                            idropoff = 0

                        vehicle_type_road = stops_data['vehicle_input']

                        num_gears = stops_data[N-1].get('num_gears', num_gears)
                        transmission_type = stops_data[N-1].get('transmission_type', transmission_type)
                        engine_size = stops_data[N-1].get('engine_size', engine_size)
                        fuel_type = stops_data[N-1].get('fuel_type', fuel_type)
                        cylinders = stops_data[N-1].get('cylinders', cylinders)
                        fuel_consumption_comb = stops_data[N-1].get('fuel_consumption_comb', fuel_consumption_comb)
                        co2_for_vehicle = predict_co2_emissions(
                            num_gears, transmission_type, engine_size, fuel_type, cylinders, fuel_consumption_comb
                        )
                        N += 1
                        print("Updated N to", N)
                    else:
                        print(f"Warning: No data for stop {N}, skipping.")
                        N += 1  # Increment N to continue the loop
                else:
                    print("Incrementing N")
                    N += 1
                    print("Incremented N to", N)
                for idx, (decoded_geometry, _, _) in enumerate(all_routes):

                    """print(n)
                    if n>1:
                      drop_off = input("Enter drop off at segment 1")
                      idropoff = int(drop_off)
                      icargo_weight -= idropoff
                      vehicle_type_road = get_vehicle_type(icargo_weight)
                      n+=1
                    if n==1:
                      n+=1"""
                    road_time=0.0


                  

                    print(f"Route {idx + 1} for Segment {i + 1}:")
                    segment_distance = geodesic(segment_start, segment_end).km
                    #print(f"Route Distance{segment_distance:.2f}")
                    if U==3:
                      if vehicle_type_road == "18_wheeler":

                        road_time = segment_distance/35
                      else:
                        road_time = segment_distance/45

                    elif U==2:
                        road_time = segment_distance/60
                    else:
                        road_time = segment_distance/70
                    #print(f"Road Time without delay: {road_time:.2f} hrs")



                    if segment_distance > 300 and air_travel_start=="yes":
                        air_emssions = calculate_co2_emissions_air(segment_distance)
                        print(f"CO2 emissions for this route segment: {air_emssions:.2f} kg")
                        total_co2_emissions += air_emssions
                        # Nearest airports for air travel
                        start_airport_name, start_airport_coords, _ = find_nearest_airport(segment_start, airport_data)
                        end_airport_name, end_airport_coords, _ = find_nearest_airport(segment_end, airport_data)
                        c1=get_weather_data(start_airport_coords)
                        c2=get_weather_data(end_airport_coords)
                        va.append(start_airport_coords)
                        va.append(end_airport_coords)



                        while c1[0]["rain_percentage"] > 0.5 or c2[0]["rain_percentage"] > 0.5:

                            print(c1)
                            print(c2)

                            if c1[0]["rain_percentage"] > 0.1:
                              na1,c11=find_next_airport(start_airport_coords,airport_data,va)
                              start_airport_name, start_airport_coords, _ = find_nearest_airport(c11, airport_data)
                              va.append(start_airport_coords)
                            if c2[0]["rain_percentage"] > 0.1:
                              na2,c22=find_next_airport(end_airport_coords,airport_data,va)
                              end_airport_name, end_airport_coords, _ = find_nearest_airport(c22, airport_data,)
                              va.append(end_airport_coords)

                            c1=get_weather_data(start_airport_coords)
                            c2=get_weather_data(end_airport_coords)
                        print(f"Air Travel: {start_airport_name} -> {end_airport_name}")

                        # Flight segment
                        air_segments.append([start_airport_coords, end_airport_coords])
                        flight_distance = geodesic(start_airport_coords, end_airport_coords).km

                        # Calculate flight time
                        flight_time = (flight_distance / 400) * 60  # Assuming average flight speed is 800 km/h

                        # Road segments to/from airports
                        road_to_airport, road_to_airport_distance, road_to_airport_duration = get_osrm_route(segment_start, start_airport_coords)[0]
                        road_from_airport, road_from_airport_distance, road_from_airport_duration = get_osrm_route(end_airport_coords, segment_end)[0]
                        print("USED AIR TRAVEL CALCULATING UR VEHICLE FOR ROAD SEGMENT BASED ON CARGO WEIGHT")
                        vehiclee = get_vehicle_airports(icargo_weight)
                        print(vehiclee)
                        print("USED")

                        # Total times
                        total_road_time = road_to_airport_duration + road_from_airport_duration
                        total_air_time = flight_time

                        # Rain delay for air route
                        #sampled_indices = np.linspace(0, len(air_segments[-1]) - 1, 2, dtype=int)
                        #sampled_air_points = [air_segments[-1][i] for i in sampled_indices]
                        #air_weather_data = [get_weather_data_cached(point) for point in sampled_air_points]
                        tt=generate_points_between_coordinates(start_airport_coords, end_airport_coords, num_points=8)

                        tt1=reroute_air_path(start_airport_coords, end_airport_coords, WEATHERBIT_API_KEY)
                        tt=tt1


                        air_weather_data = [get_weather_data_cached(point) for point in tt]
                        air_rain_delay = 0

                        for point, weather in zip(tt, air_weather_data):
                            if weather:
                                for forecast in weather:
                                    rain_probability = forecast["rain_percentage"] / 100  # Convert to decimal
                                    air_rain_delay += calculate_rain_delay(rain_probability)
                                    print(f"Air Route Weather - Timestamp: {forecast['timestamp']}, "
                                          f"Coords: {point}, Temp: {forecast['temperature']}°C, "
                                          f"Rain Probability: {forecast['rain_percentage']}%, "
                                          f"Added Delay: {calculate_rain_delay(rain_probability)} minutes")


                        # Rain delay for road routes to/from airports
                        road_rain_delay = 0

                        for road_point in road_to_airport + road_from_airport:
                            weather = get_weather_data(road_point)
                            if weather:
                                for forecast in weather:
                                    rain_probability = forecast["rain_percentage"] / 100  # Convert to decimal
                                    road_rain_delay += calculate_rain_delay(rain_probability)

                        # Total travel time
                        total_travel_time = total_road_time + total_air_time + air_rain_delay + road_rain_delay

                        print(f"Flight Distance: {flight_distance:.2f} km, Flight Time: {flight_time:.2f} minutes")
                        print(f"Road Distance to/from Airports: {road_to_airport_distance + road_from_airport_distance:.2f} km")
                        print(f"Total Ground Travel Time: {total_road_time:.2f} minutes")
                        #print(f"Total Rain Delay (Air + Ground): {air_rain_delay + road_rain_delay:.2f} minutes")
                        print(f"Total Travel Time (Including Delays): {total_travel_time:.2f} minutes")
                        
                        html_content+=f"<p>Flight Distance: {flight_distance:.2f} km, Flight Time: {flight_time:.2f} minutes<br>Road Distance to/from Airports: {road_to_airport_distance + road_from_airport_distance:.2f} km<br>Total Ground Travel Time: {total_road_time:.2f} minutes<br>Total Travel Time (Including Delays): {total_travel_time:.2f} minuutes<br></p><br><br>"

                        # Display on map
                        traffic_colors = [get_traffic_color(point) for point in road_to_airport + road_from_airport]
                        tf = traffic_factor(traffic_colors)
                        display_route_with_traffic(
                            route_map,
                            [road_to_airport + road_from_airport],
                            [traffic_colors],
                            air_segments,
                            location_airport_segments,
                            vehiclee,
                            tf
                        )
                    else:

                        # Continue with normal driving route
                        for idx, (decoded_geometry, _, _) in enumerate(all_routes):
                            print(f"Route {idx + 1} for Segment {i + 1}:")
                            sampled_indices = np.linspace(0, len(decoded_geometry) - 1, 10, dtype=int)
                            sampled_points = [decoded_geometry[i] for i in sampled_indices]


                            # Fetch weather data for sampled points
                            road_time_rain = 0
                            for point in sampled_points:

                                weather_info = get_weather_data(point)

                                if weather_info:
                                    for weather in weather_info:
                                        rain_percentage = weather["rain_percentage"]
                                        temperature = weather["temperature"]
                                        timestamp = weather["timestamp"]

                                        print(f"Timestamp: {timestamp}")
                                        print(f"Coordinates: {point}")
                                        print(f"Temperature: {temperature}°C")
                                        print(f"Rain Probability: {rain_percentage}%")
                                        road_time_rain += calculate_rain_delay(rain_percentage)
                                        if rain_percentage > 40:
                                            print("WARNING: Heavy rain expected! Possible delays.")
                                            x+=1

                                    print("-" * 50)
                                else:
                                    print(f"No weather data available for point {point}.")
                                    print("-" * 50)
                            road_time = road_time + road_time_rain

                            sampled_indices = np.linspace(0, len(decoded_geometry) - 1, 100, dtype=int)
                            road_routes_1 = [decoded_geometry[i] for i in sampled_indices]
                            avg_air=0
                            total_air=0
                            total_air = int(total_air)
                            """
                            for point in road_routes_1:
                                air_info = get_aqi_for_location(point, "68575db04030b46ff3e43bec9339813ee7e1b1ff")
                                
                                total_air+=air_info
                            avg_air=total_air/len(sampled_points)"""
                            air_1 = get_aqi_for_location(road_routes_1[0], "68575db04030b46ff3e43bec9339813ee7e1b1ff")
                            print(air_1)
                            
                            #road_routes, road_routes_distance, road_routes_duration = get_osrm_route(segment_start, segment_end)[0]
                            sampled_indices = np.linspace(0, len(decoded_geometry) - 1, 60, dtype=int)
                            road_routes = [decoded_geometry[i] for i in sampled_indices]
                            # Get traffic data for sampled points along the route
                            traffic_colors = [get_traffic_color(point) for point in road_routes]
                            tf = traffic_factor(traffic_colors)
                            

                            # Add route to map
                            road_emissions = display_route_with_traffic(route_map, [road_routes], [traffic_colors], air_segments, location_airport_segments,co2_for_vehicle,tf)
                            road_time = road_time*tf

                            print(f"Total road time: {road_time:.2f} hrs")
                            print(f"CO2 emissions for this route segment: {road_emissions:.2f} kg")
                            
                            html_content +=f"""<p>Route {idx + 1} for Segment {i + 1}:<br>
                                    CO2 Emissions: {road_emissions:.2f} kg<br>
                                    Road_time: {road_time:.2f} hrs<br>
                                    Segment_distance: {segment_distance:.2f}<br></p><br><br>"""
                            
                                    





        # Save and display the complete map
        route_map.save("route_with_traffic_and_weather.html")
        print("Route map with traffic and weather saved as route_with_traffic_and_weather.html")

    else:
        print("Error in getting coordinates for cities or stops.")
    html_content+="</body></html>"
    with open("Outputs.html", "w") as f:
      f.write(html_content)

    print("HTML file has been generated.")

@app.route('/forecast', methods=['POST'])
def forecast():
    try:
        # Get historical data from CSV file
        file = request.files['historical_data']  
        historical_df = pd.read_csv(file, index_col="Date", parse_dates=True) 
        historical_df = historical_df.rename(columns={'Demand': 'y'})
        historical_df = historical_df.asfreq("D")
        
        # Get future data from JSON (or adapt to your data source)
        data = request.get_json()
        future_df = pd.DataFrame(data['future_data'])
        future_df['Date'] = pd.to_datetime(future_df['Date'])
        future_df.set_index('Date', inplace=True)

        # Create TimeSeries objects
        series = TimeSeries.from_series(historical_df.y)
        
        # Create future covariates using historical data info
        future_covariates_transformed = create_future_covariates(future_df, series.start_time(), series.freq_str)
        
        # Make predictions
        forecast_horizon = len(future_df)
        forecast = model.predict(n=forecast_horizon, future_covariates=future_covariates_transformed)
        forecast_original_scale = scaler1.inverse_transform(forecast)

        # Generate plot
        plot_url = plot_forecast(historical_df.y, forecast_original_scale.pd_series()) 

        response = {'forecast': forecast_original_scale.pd_series().tolist(), 'plot_url': plot_url}
        return jsonify(response), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/report')
def generate_report():
    try:
        df = load_data()
        results = allocate_vehicles(df)
        report_data = process_report_data(results)
        return render_template('reports.html', report=report_data)
    
    except Exception as e:
        return render_template('error.html', error_message=str(e))


predictive_model = joblib.load(r'fedex-main\trained_model-2.pkl')

# Failure Type mapping
failure_type_map = {
    0: 'No Failure',
    1: 'Heat Dissipation Failure',
    2: 'Power Failure',
    3: 'Overstrain Failure',
    4: 'Tool Wear Failure',
    5: 'Random Failures'
}
@app.route('/predictive_maintenance')
def predictive_maintenance():
    return render_template('pm.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get input data from the request
    data = request.get_json(force=True)
    
    # Convert input data to a DataFrame
    input_data = pd.DataFrame([{
        'Temperature (°C)': data['temperature'],
        'Vibration (m/s²)': data['vibration'],
        'Sound Level (dB)': data['sound_level'],
        'Load (%)': data['load']
    }])
    
    # Make prediction
    prediction = predictive_model.predict(input_data)
    failure_status = prediction[0][0]
    failure_type = failure_type_map.get(prediction[0][1], 'Unknown')
    
    # Return the prediction as JSON
    return jsonify({
        'failure_status': 'Failure' if failure_status == 1 else 'No Failure',
        'failure_type': failure_type
    })

@app.route('/apriori')
def index_():
    return render_template('apriori1.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('apriori1.html', error='No file part')
    
    file = request.files['file']
    if file.filename == '':
        return render_template('apriori1.html', error='No selected file')
    
    min_support = float(request.form.get('min_support', 0.003))
    min_confidence = float(request.form.get('min_confidence', 0.2))
    min_lift = float(request.form.get('min_lift', 3))
    min_length = int(request.form.get('min_length', 2))
    max_length = int(request.form.get('max_length', 2))
    
    try:
        dataset = pd.read_csv(file, header=None)
    except Exception as e:
        return render_template('apriori1.html', error=f'Error reading CSV file: {str(e)}')
    
    transactions = []
    for i in range(0, len(dataset)):
        transactions.append([str(dataset.values[i, j]) for j in range(0, len(dataset.columns))])
    
    rules = apriori(transactions=transactions, 
                    min_support=min_support, 
                    min_confidence=min_confidence, 
                    min_lift=min_lift, 
                    min_length=min_length, 
                    max_length=max_length)
    
    results = list(rules)
    
    def inspect(results):
        lhs         = [tuple(result[2][0][0])[0] for result in results]
        rhs         = [tuple(result[2][0][1])[0] for result in results]
        supports    = [result[1] for result in results]
        confidences = [result[2][0][2] for result in results]
        lifts       = [result[2][0][3] for result in results]
        return list(zip(lhs, rhs, supports, confidences, lifts))
    
    results_df = pd.DataFrame(inspect(results), columns=['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
    
    return render_template('apriori1.html', 
                           frequent_itemsets=results_df.to_html(classes='data', index=False), 
                           association_rules=results_df.to_html(classes='data', index=False))

"""
# warehouse_data=dict()
# warehouse_data = {}

@app.route('/R-L')
def logistics():
    return render_template("rlogistics.html")"""

"""@app.route('/plan_route_l', methods=['GET', 'POST'])
def plan_route_l():
    if request.method == 'POST':
        base_warehouse = request.form.get('base_warehouse')
        if not base_warehouse:
            return "Error: Base warehouse is required", 400

        pund = []
        locations = request.form.getlist('location[]')
        
        for i in range(len(locations)):
            action_type = request.form.get(f'action_type_{i}')
            if action_type and locations[i]:
                data = {'action_type': action_type, 'location': locations[i]}
                pund.append(data)

        warehouse_data[base_warehouse] = pund
        print(warehouse_data)  # Debugging

        return redirect(url_for('plan_route_l'))

    return render_template('rlogistics.html')"""

"""@app.route('/data')


def show_data():
    for i in warehouse_data:
        base_warehouse_coords = get_coordinates(i)
        stopss=[]
        for j in warehouse_data[i]:
            location_coords = get_coordinates(j['location'])
            stopss.append(location_coords)
        if base_warehouse_coords and all(stopss):
            full_route_rl = [base_warehouse_coords] + stopss
            route_map = folium.Map(location=base_warehouse_coords, zoom_start=7)
            for i in range(len(full_route_rl) - 1):
                segment_start = full_route_rl[i]
                segment_end = full_route_rl[i+1]
                all_routes = get_osrm_route(segment_start, segment_end)
        for idx, (decoded_geometry,_,_) in enumerate(all_routes){
            folium.PolyLine(decoded_geometry, color='blue').add_to(route_map)
        }
    return {'warehouse_data': warehouse_data}"""



if __name__ == "__main__":
    app.run(debug=True)