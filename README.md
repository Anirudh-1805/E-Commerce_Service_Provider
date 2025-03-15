# Dynamic Routing System

## Overview
The *Dynamic Routing System* leverages real-time data to recommend efficient routes for vehicles. It integrates multiple APIs and uses advanced algorithms to optimize routing decisions based on traffic, weather, air quality, and vehicle-specific data. 

---

## Features

- *Real-Time Traffic Updates*: Integrates live traffic data for accurate route recommendations.
- *Weather-Aware Routing*: Considers weather conditions to avoid delays or unsafe routes.
- *Vehicle-Specific Recommendations*: Customizes routes based on vehicle characteristics (e.g., size, fuel type).
- *Multi-API Integration*: Combines data from APIs like TomTom, Weatherbit, AQICN and OSRM.

---

## Technologies Used

- *Programming Language*: Python
- *APIs*: 
  - TomTom Traffic API
  - Weatherbit
  - AQICN (Air Quality Index China Network)
  - OSRM (Open Source Routing Machine)
  (Caution: Continuous or excessive usage of these API may result in rate limiting or suspension of your API key.)
- *Libraries*: 
  - Pandas (Data Manipulation)
  - NumPy (Numerical Computations)
  - Requests (API Integration)
  - Flask/FastAPI (Web Framework for API Deployment)

---
## Setup and Installation

1. *Clone the Repository*:
   bash
   git clone https://github.com/yourusername/fedex.git
   cd dynamic-routing-system
   

2. *Create a Virtual Environment*:
   bash
   python -m venv venv
   source venv/bin/activate   # On Windows: venv\Scripts\activate
   

3. *Install Dependencies*:
   bash
   pip install -r requirements.txt
   

4. *Set Up API Keys*:
   Create a .env file in the root directory and add your API keys:
   env
   TOMTOM_API_KEY=your_tomtom_api_key
   WEATHERBIT=your_weather_api_key
   OSRM_URL=your_osrm_server_url
   AQICN_API_KEY=your_aqicn_api_key
   

---
## Usage

1. *Run the Application*:
   bash
   python src/main.py
   

2. *API Endpoints*:
   - /route: Get the best route based on real-time data.
     - *Request*: POST /route
       json
       {
         "start": "latitude,longitude",
         "end": "latitude,longitude",
         "vehicle": {
           "type": "car",
           "fuel": "electric"
         }
       }
       
       
---

## API Integration Details

1. *TomTom Traffic API*:
   - Provides real-time traffic updates.
   - Endpoint: https://api.tomtom.com/traffic/services

2. *Weatherbit*:
   - Provides real-time weather updates.
   - Endpoint: https://api.weatherbit.io/v2.0/forecast/daily
   
3. *OSRM*:
   - Processes open-source routing logic.
   - Endpoint: http://router.project-osrm.org
   - 
4. *AQICN*:
   - provides real-time and historical air quality data from global monitoring stations.
   - Endpoint:  https://api.waqi.info

---



## License
This project is licensed under the [GNU General Public License v3.0](LICENSE).

---

## Contact
For questions or feedback, please reach out to:
- *Anirudhan R||Hemprasad AC||Raghavan R*
- Email: anirudh2005.1805@gmail.com||hemprasad.ac2023@vitstudent.ac.in||raghavan.r2023@vitstudent.ac.in

---

Happy Routing! 
