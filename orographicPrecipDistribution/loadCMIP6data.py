import numpy as np
import pandas as pd
import os
from IPython.display import display, HTML
import xarray as xr
import copy

# Function to check for presence of files for a given MIP, scenario, frequency, and variable; and prepare list of files to use.
# Also displays status of each model and number of usable ensemble members.
def checkModels(mip,scenario,freq,varname,nminens=1,verbose=True):
    #================================================================================================================
    # ARGUMENT LIST:
    # mip (string) = Intercomparison project, e.g. 'CMIP', 'ScenarioMIP', etc
    # scenario (string) = experiment or scenario, e.g. 'historical', 'ssp585', etc
    # freq (string) = frequency, e.g. 'Amon' for monthly means
    # varname (string) = variable to extract, e.g. 'pr' for precip, 'ts' for surface temperature
    # nminens (integer, optional) = minimum number of ensemble members required for model to be included. 1 by default.
    # verbose (boolean, optional) = if True, table of models is displayed
    #=================================================================================================================
    # REQUIRED PACKAGES:
    #=================================================================================================================
    #import numpy as np
    #import pandas as pd
    #import os
    #from IPython.display import display, HTML
    #=================================================================================================================
    dirn = '/global/cfs/projectdirs/m3522/cmip6/CMIP6/'
    dirn = '/global/cfs/projectdirs/m3522/cmip6/CMIP6/'
    centers = os.listdir(dirn+mip+'/')
    modeln = []
    modellist = [] 
    shortmodeln = [] # These 2 are to display model status at the end.
    modelstatus = [] 
    incmodel = [] # List of included models' names
    for center in centers:
        centern = dirn + mip + '/' + center + '/'
        models = os.listdir(centern)
        for model in models:
            thismodeln = centern + model + '/' + scenario + '/'
            shortmodeln.append(center+'/'+model)
            freqflag = 0
            varflag = 0
            if os.path.isdir(thismodeln): # Does this scenario for this model exist?
                ensemblelist = []
                for ensmem in os.scandir(thismodeln):
                    ensmemlist = []
                    if os.path.isdir(ensmem.path+'/'+freq+'/'): # Does this frequency exist?
                        if os.path.isdir(ensmem.path+'/'+freq+'/'+varname+'/'): # Does this variable exist?
                            for grid in os.scandir(ensmem.path+'/'+freq+'/'+varname+'/'):
                            # Choose only most recent version:
                                versions = sorted(os.scandir(grid),key=lambda e: e.name,reverse=True)
                                for f in os.scandir(versions[0]):
                                    ensmemlist.append(f.path)
                                if ensmemlist:    # Check whether files exist: empty version directories can cause issues
                                    ensemblelist.append(ensmemlist)
                                if not(thismodeln in modeln):
                                    modeln.append(thismodeln)
                        else:
                            varflag = varflag + 1
                    else:
                        freqflag = freqflag + 1 
                    
                if freqflag==len(os.listdir(thismodeln)): # Missing frequency
                    modelstatus.append(freq+' missing')
                elif varflag==len(os.listdir(thismodeln)): # Missing variable
                    modelstatus.append(varname+' missing')
                elif varflag+freqflag>=len(os.listdir(thismodeln)): # Missing a combination of frequency and variable
                    modelstatus.append('Too many incomplete ensemble members')
                elif len(ensemblelist)<nminens: # Too few ensemble members
                    modelstatus.append('Too few ensemble members')
                elif varflag>0 and varflag<len(os.listdir(thismodeln)): # Some ensemble members lack the variable
                    modelstatus.append('Included '+str(len(ensemblelist))+' ensemble members, excluded '+str(varflag)+' due to missing '+varname)
                    modellist.append(ensemblelist)
                    incmodel.append(center+'/'+model)
                elif freqflag>0 and freqflag<len(os.listdir(thismodeln)): # Some ensemble members lack the frequency
                    modelstatus.append('Included '+str(len(ensemblelist))+' ensemble members, excluded '+str(freqflag)+' due to missing '+freq)
                    modellist.append(ensemblelist)
                    incmodel.append(center+'/'+model)
                else:                                # Everything's ok
                    modelstatus.append('Included '+str(len(ensemblelist))+' ensemble members') 
                    modellist.append(ensemblelist)
                    incmodel.append(center+'/'+model)
            else:                                    # Scenario missing
                modelstatus.append(scenario+' missing') 
    # End of loops
    
    # Display status of all models
    if verbose:
        d1 = {'Model':shortmodeln,'Status':modelstatus}
        df1 = pd.DataFrame(d1, columns=['Model','Status'])
        pd.options.display.max_rows
        pd.set_option('display.max_colwidth', None)
        display(df1)
    # Display numbers of ensemble members for each included model
    d2 = {'Included models':incmodel,'Number of Ensemble Members':[len(modellist[x]) for x in np.arange(len(modellist))]}
    df2 = pd.DataFrame(d2, columns=['Included models','Number of Ensemble Members'])
    if verbose:
        display(df2)
    return modellist, df2 # modellist is a list of lists of filenames. Elements can be accessed as modellist[model][ensemblemember][filenumber], as required for ensemble-mean scripts.
# End of function

# Function to check whether the models have output for the required range of years. 
# Ensemble members and models for which the years are not available are removed from the list.
# For monthly data, incomplete ensemble members (years in the middle missing) are detected and removed. 
# For other frequencies, the output includes ensemble members with gaps as long as years within the range are available.
def checkYears(modellist,modelnamesFrame,st,en,nminens=1,verbose=True):
    #================================================================================================================
    # ARGUMENT LIST:
    # modellist (list of lists) = list of files with model output, organized as output from checkModels
    # st (integer) = first year in range
    # en (integer) = last year in range
    # nminens (integer, optional) = minimum number of ensemble members required for model to be included. 1 by default.
    # verbose (boolean, optional) = Prints the model and ensemble numbers of the deleted files. True by default.
    #=================================================================================================================
    # REQUIRED PACKAGES:
    #=================================================================================================================
    #import numpy as np
    #import pandas as pd
    #import os
    #from IPython.display import display, HTML
    #import xarray as xr
    #import copy
    #=================================================================================================================
    shortlist = copy.deepcopy(modellist)
    shortFrame = modelnamesFrame.copy(deep=True)
    modelnum = 0
    ctr1 = 0
    for model in modellist:
        ensnum = 0
        ctr2 = 0
        for ensemblemember in model:  # ensemblemember is a list of files (separated by years) in that member of the ensemble
            # wrb, decode all time to cf_time rather than default numpy.datetime64 because 
            #  this otherwise fails when attempting to load data with proleptic gregorian calendar
            #  that goes beyond pandas year range of 2262 (CMIP6 output often goes to year 2300)
            #print(ensemblemember)
            #dat = xr.open_mfdataset(ensemblemember,combine='by_coords')
            dat = xr.open_mfdataset(ensemblemember,combine='by_coords',use_cftime=True)
            # not successful:  this deals with overlapping times (e.g. NorESM model has one file ending Dec 2040, and the next starting Jan 2040)
            #dat = xr.open_mfdataset(ensemblemember,use_cftime=True,compat='override',coords='minimal')
            if '/Amon/' in ensemblemember[0]: #Check if this is monthly data
                monlen = 12*(1+(int(dat.time.dt.year[-1])-int(dat.time.dt.year[0])))
                if int(dat.time.dt.year[0])>st or int(dat.time.dt.year[-1])<en or len(dat.time.values)<monlen:
                    if verbose:
                        if len(dat.time.values)<monlen:
                            print('Removing ensemble member '+str(ctr2)+' from model number '+str(ctr1)+'. Incomplete record. ')
                        else:
                            print('Removed ensemble member '+str(ctr2)+' from model number '+str(ctr1)+'. Years: '+str(int(dat.time.dt.year[0]))+' to '+str(int(dat.time.dt.year[-1])))
                    del shortlist[modelnum][ensnum]
                    ensnum = ensnum - 1
            else:
                if int(dat.time.dt.year[0])>st or int(dat.time.dt.year[-1])<en:
                    if verbose:
                        print('Removing ensemble member '+str(ctr2)+' from model number '+str(ctr1)+'. Years: '+str(int(dat.time.dt.year[0]))+' to '+str(int(dat.time.dt.year[-1])))
                    del shortlist[modelnum][ensnum]
                    ensnum = ensnum - 1
            ensnum = ensnum + 1
            ctr2 = ctr2+1
        if len(shortlist[modelnum])<nminens:
            del shortlist[modelnum]
            shortFrame = shortFrame.drop([modelnum])
            if verbose:
                print('Removing model number '+str(ctr1))
            modelnum = modelnum-1
        modelnum = modelnum + 1
        ctr1 = ctr1+1
        shortFrame.index = range(len(shortFrame))
    return shortlist, shortFrame

# Function to grab all available data for a given MIP, scenario, frequency, variable. Returns a list of dataArrays corresponding to ensemble members.
def loadData(modellist,varname,chunks=None):
    #================================================================================================================
    # ARGUMENT LIST:
    # varname (string) = variable to extract, e.g. 'pr' for precip, 'ts' for surface temperature
    #===========================================================
    # REQUIRED PACKAGES (the rest are loaded in modellist below)
    #===========================================================
    #import xarray as xr
    #===========================================================
    # LOAD MODELLIST BEFORE RUNNING THIS
    #===========================================================
    modelnum = 0
    allDat = []
    for model in modellist:
        ensnum = 0
        for ensemblemember in model:  # ensemblemember is a list of files (separated by years) in that member of the ensemble
            # wrb: convert all to cf_time to avoid weird issues with proleptic gregorian calendar:
            #dat = xr.open_mfdataset(ensemblemember,combine='by_coords')
            dat = xr.open_mfdataset(ensemblemember,combine='by_coords',use_cftime=True,chunks=chunks,parallel=True)
            datvar = dat[varname]
            if ensnum:
                ensDatVar = xr.concat([ensDatVar, datvar.expand_dims('ensmember')],dim='ensmember')
            else:
                ensDatVar = datvar.expand_dims('ensmember')
            ensnum = ensnum + 1
        allDat.append(ensDatVar)
        del ensDatVar
        modelnum = modelnum + 1
    return allDat   

# Function to get ensemble-mean, monthly mean of a given variable for a given MIP, scenario, frequency, variable. Returns a dataArray w/ dimensions model(nModels),month(12),lat,lon
def loadMonthMMM2D(modellist,yrstart,yrend):
    modelnum = 0
    datSeas1 = modellist[0].sel(time=slice(yrstart+'-01-01',yrend+'-12-31')).groupby('time.month').mean(
        dim=('time','ensmember'))
    datSeas1 = datSeas1.drop('height',errors='ignore')
    #print(datSeas1)
    for model in modellist:
        datSeas = model.sel(time=slice(yrstart+'-01-01',yrend+'-12-25')).groupby('time.month').mean(dim=('time','ensmember'))
        datSeas = datSeas.drop('height',errors='ignore')
        for k in datSeas.coords:
            if 'longitude' in k:
                datSeas = datSeas.rename({'longitude':'lon'})
                continue
            if 'latitude' in k:
                datSeas = datSeas.rename({'latitude':'lat'})
        if modelnum:
            # interpolate to first model's grid (by inspection we know this is 128x256, most common)
#             print(datSeas1)
#             print(datSeas)
            datSeasInterp = datSeas.chunk({'month': -1}).interp_like(datSeas1.chunk({'month': -1})).squeeze()
            #print(datSeasInterp)
            datSeasMMM = xr.concat([datSeasMMM, datSeasInterp.expand_dims(dim='model',axis=0)],dim='model')
        else:
            datSeasMMM = datSeas.expand_dims(dim='model',axis=0)
        modelnum = modelnum + 1
        #print(modelnum)
    return datSeasMMM
    