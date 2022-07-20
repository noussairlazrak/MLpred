.. GEOS CF Localized Forecasts documentation master file, created by
   sphinx-quickstart on Wed Jul  6 20:32:27 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GEOS CF Localized Forecasts's documentation
=======================================================


Aim and Scope
============================================
This library focuses on generating localized forecasts for select locations based on NASA GMAO's GEOS Composition Forecasting (GEOS-CF). Some of the location data are provided by OpenAQ. 
Using this library, you can train and load models based on you location data online and measure the uncertainty of the forecast based on the data provided.


Core Features
============================================
* Researchers and ML practitioners can conveniently use an existing model instead of training their location-based model each time.
* Users can train and execute a model for each location online.
* Users can explore the config and information for each location.
* Users can generate forecasts plots using predefined models
* Users and contributors can help provide data and save a trained model for future usage. 
* Simply call the generate forecasts function and specify the output format (dataframe, plot, Shap values)

Getting Started Example
============================================

.. code-block:: Python

    import MLpred.mlpred as mlpred
    import MLpred.funcs


.. code-block:: Python

    site_init = mlpred.ObsSite(OPENAQID,model_source='s3',species= "SPECIES (NO2, PM25, O3)")
    forecasts_time_series = site_init.get_location_forecasts(dt.datetime(START_DATE),end_date=dt.datetime(END_DATE), output ="plot")


.. toctree::
   :maxdepth: 2
   
    licence
    
   :caption: Contents:
   
   modules
   



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
