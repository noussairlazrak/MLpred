#!/bin/python
import sys
import os
import numpy as np
import pandas as pd
import datetime as dt
import xarray as xr
import xgboost as xgb
import plotly.express as px
import plotly.graph_objects as go
import requests
import pickle
from dateutil.relativedelta import relativedelta 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
from tqdm import tqdm as tqdm

#ZARR_TEMPLATE = ["geos-cf/zarr/geos-cf.met_tavg_1hr_g1440x721_x1.zarr","geos-cf/zarr/geos-cf.chm_tavg_1hr_g1440x721_v1.zarr"]
ZARR_TEMPLATE = ["geos-cf/zarr/geos-cf-rpl.zarr"]
OPENDAP_TEMPLATE = "https://opendap.nccs.nasa.gov/dods/gmao/geos-cf/fcast/met_tavg_1hr_g1440x721_x1.latest"
M2_TEMPLATE = "/home/ftei-dsw/Projects/SurfNO2/data/M2/{c}/small/*.{c}.%Y%m*.nc4"
M2_COLLECTIONS = ["tavg1_2d_flx_Nx","tavg1_2d_lfo_Nx","tavg1_2d_slv_Nx"]
OPENAQ_TEMPLATE = 'https://docs.openaq.org/v2/measurements?date_from={Y1}-{M1}-01T00%3A00%3A00%2B00%3A00&date_to={Y2}-{M2}-01T00%3A00%3A00%2B00%3A00&limit=10000&page=1&offset=0&sort=asc&radius=1000&location_id={ID}&parameter={PARA}&order_by=datetime'

# list with gas names. Used to identify fields that need to be converted from v/v to ppbv
DEFAULT_GASES = ['co', 'hcho', 'no2', 'noy', 'o3']

PPB2UGM3 = {'no2':1.88,'o3':1.97}
VVtoPPBV = 1.0e9

class ObsSiteList:
    def __init__(self,ifile=None):
        '''
        Initialize ObsSiteList object (read from file if provided).
        '''
        self._site_list = None
        if ifile is not None:
            self.load(ifile)


    def save(self,ofile='site_list.pkl'):
        '''Write out a site_list, discarding all model and observation data beforehand (but keeping the trained XGBoost instances)'''
        for isite in self._site_list:
            isite._obs = None
            isite._mod = None
        pickle.dump( self._site_list, open(ofile,'wb'), protocol=4 )
        print('{} sites written to {}'.format(len(self._site_list),ofile))
        return
    
    
    def load(self,ifile):
        '''Reads a previously saved site list'''
        self._site_list = pickle.load(open(ifile,'rb'))
        print('Read {} sites from {}'.format(len(self._site_list),ifile))
        return


    def filter_sites(self,year=2018,minobs=72,minvalue=15.0,silent=True):
        '''Wrapper routine to get dataframe with average values for all sites with at least nobs observations for the first day of each month of the given year'''
        allmonths = []
        for imonth in tqdm(range(12)):
            testurl = 'https://docs.openaq.org/v2/measurements?date_from={0:d}-{1:02d}-01T00%3A00%3A00%2B00%3A00&date_to={0:d}-{1:02d}-02T00%3A00%3A00%2B00%3A00&limit=100000&page=1&offset=0&sort=asc&parameter=no2&order_by=datetime'.format(year,imonth+1)
            allmonths.append(read_openaq(testurl,silent=silent))
        tmp = pd.concat(allmonths)
        cnt = tmp.groupby(['locationId','unit']).count().reset_index()
        sites = list(cnt.loc[cnt.value>minobs,'locationId'].values)
        subdf = tmp.loc[tmp['locationId'].isin(sites)].copy()
        meandf = subdf.groupby(['locationId','unit']).mean().reset_index()
        meandf.loc[meandf['unit']=='µg/m³','value'] = meandf.loc[meandf['unit']=='µg/m³','value']*1./1.88
        meandf.loc[meandf['unit']=='ppm','value'] = meandf.loc[meandf['unit']=='ppm','value']*1000.
        site_ids = list(meandf.loc[meandf['value']>=minvalue,'locationId'].values)
        print('Found {} sites with average concentration above {} ppbv and more than {} observations'.format(len(site_ids),minvalue,minobs))
        self._minvalue = minvalue
        return site_ids 
    
    
    def create_list(self,site_ids,minobs=240,silent=True,model_source='nc4',log=False,xgbparams={"booster":"gbtree","eta":0.5},**kwargs):
        '''Create a list of observation sites by training all sites listed in site_ids that have at least minobs number of observations in the training window'''
        self._site_list = []
        for i in tqdm(site_ids):
            isite = ObsSite(location_id=i,silent=silent,model_source=model_source)
            isite.read_obs(**kwargs)
            if isite._obs is None:
                if not isite._silent:
                    print('No observations found for site {}'.format(i))
                continue
            if isite._obs.shape[0] < minobs:
                if not isite._silent:
                    print('Not enough observations found for site {}'.format(i))
                continue
            isite.read_mod()
            rc = isite.train(mindat=minobs,log=log,xgbparams=xgbparams)
            if rc==0:
                self._site_list.append(isite)
        return 


    def calc_ratios(self,start,end):
        '''Get ratios between prediction and observation for each site in site_list'''
        predictions = self.predict_sites(start,end)
        siteIds = []; siteNames=[]
        siteLats = []; siteLons=[]
        ratios = []; meanObs=[]; meanPred=[]
        for p in predictions:
            ip = predictions[p]
            idf = ip['prediction']
            if idf is None:
                continue
            siteIds.append(p)
            siteNames.append(ip['name'])
            siteLats.append(ip['lat'])
            siteLons.append(ip['lon'])
            ratios.append(idf['observation'].values.mean()/idf['prediction'].values.mean())
            meanObs.append(idf['observation'].values.mean())
            meanPred.append(idf['prediction'].values.mean())
        siteRatios = pd.DataFrame({'Id':siteIds,'name':siteNames,'lat':siteLats,'lon':siteLons,'ratio':ratios,'obs':meanObs,'pred':meanPred})
        siteRatios['relChange'] = (siteRatios['ratio']-1.0)*100.0
        return siteRatios
    
    
    def predict_sites(self,start,end):
        '''Predict concentrations at all sites in the list of ObsSite objects'''
        predictions = {}
        for isite in tqdm(self._site_list):
            isite.read_obs_and_mod(start=start,end=end)
            df = isite.predict(start=start,end=end)
            predictions[isite._id] = {'name':isite._name,'lat':isite._lat,'lon':isite._lon,'prediction':df}
        return predictions


    def plot_deviation(self,siteRatios,title='NO2 deviation',minval=-30.,maxval=30.,mapbox_access_token=None):
        '''Make global map showing deviation betweeen predictions and observations'''
        siteRatios['text'] = ['{0:} (ID {1:}, Pred={2:.2f}ppbv, Deviation={3:.2f}%'.format(i,j,k,l) for i,j,k,l in zip(siteRatios['name'],siteRatios['Id'],siteRatios['pred'],siteRatios['relChange'])]
        fig = go.Figure(data=go.Scattermapbox(
                lon = siteRatios['lon'],
                lat = siteRatios['lat'],
                text = siteRatios['text'],
                mode = 'markers',
                marker = go.scattermapbox.Marker(
                    size = siteRatios['pred'],
                    sizemode = 'area',
                    color = siteRatios['relChange'],
                    cmin = minval,
                    cmax = maxval,
                    colorscale = 'RdBu',
                    opacity = 0.8,
                    autocolorscale = False,
                    reversescale = True,
                    colorbar_title=title,
                ),
                #name = siteRatios['name'],
                ))
        #fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(hovermode='closest',
                          mapbox_accesstoken=mapbox_access_token,
                          mapbox_style='dark',
                         )
        fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        return fig


class ObsSite:
    def __init__(self,location_id,read_obs=False,silent=False,model_source='nc4',species='no2',**kwargs):
        '''
        Initialize ObsSite object.
        '''
        self._init_site(location_id,species,silent,model_source)
        if read_obs:
            self.read_obs(**kwargs)


    def read_obs_and_mod(self,**kwargs):
        '''Convenience wrapper to read both observations and model data'''
        self.read_obs(**kwargs)
        self.read_mod(**kwargs)
        return


    def read_obs(self,data=None,**kwargs):
        '''Wrapper routine to read observations'''
        if data is None:
            data = self._read_openaq(**kwargs)
        if data is None:
            if not self._silent:
                print('Warning: no observations found!')
            return
        if 'lat' not in data.columns:
            if not self._silent:
                print('Warning: no latitude entry found in observation data - cannot process information')
            return
        ilat = np.round(data['lat'].median(),2)
        ilon = np.round(data['lon'].median(),2)
        iname = data['location'].values[0]
        if not self._silent:
            print('Found {:d} observations for {:} (lon={:.2f}; lat={:.2f})'.format(data.shape[0],iname,ilon,ilat))
        self._lat = ilat if self._lat is None else self._lat 
        self._lon = ilon if self._lon is None else self._lon 
        assert(ilat==self._lat)
        assert(ilon==self._lon)
        self._name = iname if self._name is None else self._name
        if iname != self._name and not self._silent:
            print('Warning: new station name is {}, vs. previously {}'.format(iname,self._name))
            self._name = iname
        self._obs = self._obs.merge(data,how='outer') if self._obs is not None else data
        return


    def read_mod(self,**kwargs):
        '''Wrapper routine to read model data'''
        assert(self._lon is not None and self._lat is not None)
        if 'start' not in kwargs:
            kwargs['start'] = self._obs['time'].min()
        if 'end' not in kwargs:
            kwargs['end'] = self._obs['time'].max()
        mod = self._read_model(self._lon,self._lat,**kwargs)
        self._mod = self._mod.merge(mod,how='outer') if self._mod is not None else mod
        return


    def train(self,target_var='value',skipvar=['time','location','lat','lon'],mindat=None,test_size=0.2,log=False,xgbparams={'booster':'gbtree'},**kwargs):
        '''Train XGBoost model using data in memory'''
        dat = self._merge(**kwargs)
        if dat is None:
            return -2
        if mindat is not None:
            if dat.shape[0]<mindat:
                print('Warning: not enough data - only {} rows vs. {} requested'.format(dat.shape[0],mindat))
                return -1
        yvar = [target_var]
        blacklist = yvar + skipvar
        xvar = [i for i in dat.columns if i not in blacklist]
        X = dat[xvar]
        y = dat[yvar]
        if log:
            y = np.log(y)
        Xtrain, Xtest, ytrain, ytest = train_test_split( X, y, test_size=test_size)
        train = xgb.DMatrix(Xtrain,ytrain)
        if not self._silent:
            print('training model ...')
        bst = xgb.train(xgbparams,train)
        ptrain = bst.predict(xgb.DMatrix(Xtrain))
        ptest = bst.predict(xgb.DMatrix(Xtest))
        ytrainf = np.array(ytrain).flatten()
        ytestf = np.array(ytest).flatten()
        if log:
            ytrainf = np.exp(ytrainf)
            ytestf  = np.exp(ytestf)
            ptrain  = np.exp(ptrain)
            ptest   = np.exp(ptest)
        if not self._silent:
            print('Training:')
            print('r2 = {:.2f}'.format(r2_score(ytrainf,ptrain)))
            print('nrmse = {:.2f}'.format( sqrt(mean_squared_error(ytrainf,ptrain))/np.std(ytrainf)))
            print('nmb = {:.2f}'.format(np.sum(ptrain-ytrainf)/np.sum(ytrainf)))
            print('Test:')
            print('r2 = {:.2f}'.format(r2_score(ytestf,ptest)))
            print('nrmse = {:.2f}'.format( sqrt(mean_squared_error(ytestf,ptest))/np.std(ytestf)))
            print('nmb = {:.2f}'.format(np.sum(ptest-ytestf)/np.sum(ytestf)))
        self._bst = bst
        self._xcolumns = X.columns
        self._log = log
        return 0


    def predict(self,add_obs=True,**kwargs):
        '''Make prediction for given time window and return predicted values along with observations'''
        if add_obs:
            dat = self._merge(**kwargs)
        else:
            start = kwargs['start'] if 'start' in kwargs else dat['time'].min()
            end = kwargs['end'] if 'end' in kwargs else dat['time'].max()
            dat = self._mod.loc[(self._mod['time']>=start)&(self._mod['time']<=end)].copy()
            if 'value' not in dat:
                dat['value'] = [np.nan for i in range(dat.shape[0])]
        if dat is None:
            return None
        pred = self._bst.predict(xgb.DMatrix(dat[self._xcolumns]))
        if self._log:
            pred = np.exp(pred)
        df = dat[['time','value']].copy()
        df['prediction'] = pred
        df.rename(columns={'value':'observation'},inplace=True)
        return df


    def plot(self,df,y=['observation','prediction'],ylabel=r'$\text{NO}_{2}\,[\text{ppbv}]$',**kwargs):
        '''Make plot of prediction vs. observation, as generated by self.predict()'''
        title = 'Site = {0} ({1:.2f}N, {2:.2f}E)'.format(self._name,self._lat,self._lon)
        fig = px.line(df,x='time',y=y,labels={'value':ylabel},title=title,**kwargs)
        fig.update_layout(xaxis_title="Date (UTC)",yaxis_title=ylabel)
        fig.update_layout(legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1.),legend_title='')
        return fig


    def _merge(self,start=None,end=None,mod_blacklist=['lat','lon','lev']):
        '''Merge model and observation and limit to given time window'''
        if self._mod is None or self._obs is None:
            if not self._silent:
                print('Warning: cannot merge because mod or obs is None')
            return None
        # toss model variables that are blacklisted.
        ivars = [i for i in self._mod.columns if i not in mod_blacklist] 
        # interpolate model data to openaq time stamps
        mdat = self._mod[ivars].merge(self._obs,on=['time'],how='outer').sort_values(by='time')
        idat = mdat.set_index('time').interpolate(method='slinear').reset_index()
        dat = idat.loc[idat['time'].isin(self._obs['time'])].copy()
        start = start if start is not None else dat['time'].min()
        end = end if end is not None else dat['time'].max()
        testvar = ivars[-1]
        idat = dat.loc[(dat['time']>=start)&(dat['time']<=end)&(~np.isnan(dat[testvar]))]
        if idat.shape[0]==0:
            idat = None
        return idat


    def _init_site(self,location_id,species,silent,model_source):
        '''Create an empty site object'''
        self._id      = location_id
        self._species = species
        self._silent  = silent
        self._modsrc  = model_source
        self._lat     = None
        self._lon     = None
        self._name    = None
        self._obs     = None
        self._mod     = None
        self._log     = False
        return


    def _read_model(self,ilon,ilat,start,end,resample=None,source=None,template=None,collections=None,remove_outlier=0,gases=DEFAULT_GASES,**kwargs):
        '''Read model data'''
        dfs = []
        source = self._modsrc if source is None else source
        if source=='opendap':
            template = OPENDAP_TEMPLATE if template is None else template
            template = template if isinstance(template,type([])) else [template]
            for t in template:
                if not self._silent:
                    print('Reading {}...'.format(t))
                ids = xr.open_dataset(t).sel(lon=ilon,lat=ilat,lev=1,method='nearest').sel(time=slice(start,end)).load().to_dataframe().reset_index()
                dfs.append(ids)
        if source=='nc4':
            template = M2_TEMPLATE if template is None else template
            collections = M2_COLLECTIONS if collections is None else collections
            for c in collections:
                itemplate = template.replace("{c}",c)
                ifiles = start.strftime(itemplate)
                if not self._silent:
                    print('Reading {}...'.format(c))
                ids = xr.open_mfdataset(ifiles).sel(lon=ilon,lat=ilat,method='nearest').sel(time=slice(start,end)).load().to_dataframe().reset_index()
                dfs.append(ids)
        if source=='zarr':
            template = ZARR_TEMPLATE if template is None else template
            template = template if isinstance(template,type([])) else [template]
            for t in template:
                if not self._silent:
                    print('Reading {}...'.format(t))
                ids = xr.open_zarr(t).sel(lon=ilon,lat=ilat,lev=1,method='nearest').sel(time=slice(start,end)).load().to_dataframe().reset_index()
                dfs.append(ids)
        mod = dfs[0]
        for d in dfs[1:]:
            merge_on = ['time','lat','lon']
            if 'lev' in d and 'lev' in mod:
                merge_on.append('lev')
            mod = mod.merge(d,on=merge_on)
        mod['time'] = [pd.to_datetime(i) for i in mod['time']]
        if resample is not None:
            mod = mod.set_index('time').resample(resample).mean().reset_index()
        mod['month'] = [i.month for i in mod['time']]
        mod['hour'] = [i.hour for i in mod['time']]
        mod['weekday'] = [i.weekday() for i in mod['time']]
        # convert trace gases from v/v to ppbv
        for g in gases:
            if g in mod:
                print('Convert from v/v to ppbv: {}'.format(g))
                mod[g] = mod[g] * VVtoPPBV
        return mod


    def _read_openaq(self,start=dt.datetime(2018,1,1),end=None,normalize=False,**kwargs):
        '''Read OpenAQ observations and return in ppbv'''
        end = start+relativedelta(years=1) if end is None else end
        url = OPENAQ_TEMPLATE.replace('{ID}',str(self._id)).replace('{PARA}',self._species).replace('{Y1}',str(start.year)).replace('{M1}','{:02d}'.format(start.month)).replace('{D1}','{:02d}'.format(start.day)).replace('{Y2}',str(end.year)).replace('{M2}','{:02d}'.format(end.month)).replace('{D2}','{:02d}'.format(end.day))
        allobs = read_openaq(url,silent=self._silent,**kwargs)
        if allobs is None:
            return None
        obs = allobs.loc[(allobs['parameter']==self._species)&(~np.isnan(allobs['value']))&(allobs['value']>=0.0)].copy()
        # convert everything to ppbv
        if self._species != 'pm25':
            assert(self._species in PPB2UGM3)
            conv_factor = PPB2UGM3[self._species]
            obs.loc[obs['unit']=='ppm','value'] = obs.loc[obs['unit']=='ppm','value']*1000.0
            obs.loc[obs['unit']=='µg/m³','value'] = obs.loc[obs['unit']=='µg/m³','value']*1./conv_factor
        # subset to relevant columns
        outobs = obs[['time','location','value']].copy()
        if normalize:
            outobs['value'] = (outobs['value']-outobs['value'].mean())/outobs['value'].std()
        if 'coordinates.latitude' in obs.columns and 'coordinates.longitude' in obs.columns:
            outobs['lat'] = obs['coordinates.latitude']
            outobs['lon'] = obs['coordinates.longitude']
        else:
            if not self._silent:
                print('Warning: no coordinates in dataset')
        return outobs

  
def read_openaq(url,reference_grade_only=True,silent=False,remove_outlier=0,**kwargs):
    '''Helper routine to read OpenAQ via API (from given url) and create a dataframe of the data'''
    if not silent:
        print('Quering {}'.format(url))
    r = requests.get( url )
    assert(r.status_code==200)
    allobs = pd.json_normalize(r.json()['results'])
    if allobs.shape[0]==0:
        if not silent:
            print('Warning: no OpenAQ data found for specified url')
        return None
    allobs = allobs.loc[(allobs['value']>=0.0)&(~np.isnan(allobs['value']))].copy()
    if reference_grade_only:
        allobs = allobs.loc[allobs['sensorType']=='reference grade'].copy()
    allobs['time'] = [dt.datetime.strptime(i,'%Y-%m-%dT%H:%M:%S+00:00') for i in allobs['date.utc']]
    if remove_outlier > 0:
        std = allobs['value'].std()
        mn  = allobs['value'].mean()
        minobs = mn - remove_outlier*std
        maxobs = mn + remove_outlier*std
        norig = allobs.shape[0]
        allobs = allobs.loc[(allobs['value']>=minobs)&(allobs['value']<=maxobs)].copy()
        if not silent:
            nremoved = norig - allobs.shape[0]
            print('removed {:.0f} of {:.0f} values because considered outliers ({:.2f}%)'.format(nremoved,norig,np.float(nremoved)/np.float(norig)*100.0))
    return allobs


def nsites_by_threshold(df,maxconc=50):
    '''Write number of sites with mean concentration above concentration threshold for concentrations ranging from 0 to maxconc ppbv'''
    concrange = np.arange(maxconc+1)*1.0
    ns = []
    for ival in concrange: 
        nsit = df.loc[df.value>ival].shape[0]
        ns.append(nsit)
    nsites = pd.DataFrame()
    nsites['threshold'] = concrange 
    nsites['nsites'] = ns
    return nsites


def plot_deviation_orig(siteRatios,title=None,minval=-30.,maxval=30.):
    '''Make global map showing deviation betweeen predictions and observations'''
    siteRatios['text'] = ['{0:}, Deviation={1:.2f}%'.format(i,j) for i,j in zip(siteRatios['name'],siteRatios['relChange'])]
    fig = go.Figure(data=go.Scattergeo(
            lon = siteRatios['lon'],
            lat = siteRatios['lat'],
            text = siteRatios['text'],
            mode = 'markers',
            marker = dict(
                size = siteRatios['obs'],
                sizemode = 'area',
                color = siteRatios['relChange'],
                cmin = minval,
                cmax = maxval,
                colorscale = 'RdBu',
                autocolorscale = False,
                reversescale = True,
                line_color='rgb(40,40,40)',
                line_width=0.5,
                colorbar_title='NO2 deviation',
            ),
            ))
    fig.update_layout(title_text = 'Test',
                      showlegend = False,
                      height=300,
                      geo=dict(landcolor='rgb(217,217,217)'),
                      margin={"r":0,"t":0,"l":0,"b":0})
    return fig


