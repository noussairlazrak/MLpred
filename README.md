# WELCOME TO GEOS-CF Localized Forecasts

#### GEOS-CF Localized Forecasts - A lightweight Python Library For Automating local forecasts generation based on GEOS-CF and local observation data 

This library focuses on generating localized forecasts for select locations based on NASA GMAO's GEOS Composition Forecasting (GEOS-CF). Some of the location data are provided by OpenAQ. Using this library, you can train and load models based on you location data online and measure the uncertainty of the forecast based on the data provided.

### Documentation
(TBD) Link to Documentation: https://geos-cf-localized-forecasts.readthedocs.io/en/latest/

### Aim and Scope

A forecasting library, equipped with  features to provide localized air quality forecasts. By integrating NASA GMAO's GEOS Composition Forecasting (GEOS-CF) data with location information from OpenAQ and location observation as availabe. One of the key features is the ability to dynamically train models online, ensuring that forecasts remain current and reflective of evolving atmospheric conditions. This adaptive approach, coupled with robust uncertainty measurement techniques, enables users to assess the reliability of predictions and make informed decisions.


### Core Features
* Researchers and ML practitioners can conveniently use an existing model instead of training their location-based model each time.
* Users can train and execute a model for each location online.
* Users can explore the config and information for each location.
* Users can generate forecasts plots using predefined models
* Users and contributors can help provide data and save a trained model for future usage. 
* Simply call the generate forecasts function and specify the output format (dataframe, plot, Shap values)

### Getting Started Example

```python
import sys
sys.path.insert(1,'MLpred')
from MLpred import mlpred
from MLpred import funcs
import datetime as dt
```


```python
# Set location parameters
site_settings = {'l_name': f'ACO Mexico City', 
             'species': 'no2', 
             'silent': True, 
             'lat': None, 
             'lon': None, 
             'model_src': 'local',
             'obs_src': 'local',
             'openaq_id': None,
             'model_tuning' : False,
             'model_url': None,
             'obs_url': 'link to model if availabe otherwise None',
             'resample' : '5D',
             'unit' : 'ppb',
             'interpolation': True,
             'remove_outlier': False,
             'start' : None,   
             'end': dt.datetime.today()
            }
# Set observation file parameters
obs_settings = {'time_col': 'date', 
                 'date_format': '%Y-%m-%d %H:%M:%S', 
                 'obs_val_col': 'valor', 
                 'lat_col': 'latitud', 
                 'lon_col': 'longitud',
                }

all_frcsts = mlpred.get_localised_forecast(site_settings = site_settings, obs_settings=obs_settings)
