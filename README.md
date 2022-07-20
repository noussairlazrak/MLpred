# WELCOME TO GEOS-CF Localized Forecasts

#### GEOS-CF Localized Forecasts - A lightweight Python Library For Automating local forecasts generation based on GEOS-CF and local observation data 

This library focuses on generating localized forecasts for select locations based on NASA GMAO's GEOS Composition Forecasting (GEOS-CF). Some of the location data are provided by OpenAQ. Using this library, you can train and load models based on you location data online and measure the uncertainty of the forecast based on the data provided.

### Documentation
Link to Documentation: https://geos-cf-localized-forecasts.readthedocs.io/en/latest/

### Aim and Scope
This library focuses on generating localized forecasts for select locations based on NASA GMAO's GEOS Composition Forecasting (GEOS-CF). Some of the location data are provided by OpenAQ. 
Using this library, you can train and load models based on you location data online and measure the uncertainty of the forecast based on the data provided.

### Core Features
* Researchers and ML practitioners can conveniently use an existing model instead of training their location-based model each time.
* Users can train and execute a model for each location online.
* Users can explore the config and information for each location.
* Users can generate forecasts plots using predefined models
* Users and contributors can help provide data and save a trained model for future usage. 
* Simply call the generate forecasts function and specify the output format (dataframe, plot, Shap values)

### Getting Started Example

```python
import MLpred.mlpred as mlpred
import MLpred.funcs
```


```python
# train the model on the cite data and generate a forecast plot based on GEOS-CF and OpenAQ data
site_init = mlpred.ObsSite(OPENAQID,model_source='s3',species= "SPECIES (NO2, PM25, O3)")
forecasts = site_init.get_location_forecasts(dt.datetime(START_DATE),end_date=dt.datetime(END_DATE))
