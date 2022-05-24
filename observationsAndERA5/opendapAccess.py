paths= ["2003/SNDR.AQUA.AIRS_IM.20031001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210621143903.nc",
       "2004/SNDR.AQUA.AIRS_IM.20041001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210618014329.nc",
       "2005/SNDR.AQUA.AIRS_IM.20051001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210618043421.nc",
       "2006/SNDR.AQUA.AIRS_IM.20061001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210618042716.nc",
       "2007/SNDR.AQUA.AIRS_IM.20071001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210619055548.nc",
       "2008/SNDR.AQUA.AIRS_IM.20081001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210617092945.nc",
       "2009/SNDR.AQUA.AIRS_IM.20091001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210616013021.nc",
       "2010/SNDR.AQUA.AIRS_IM.20101001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210616070617.nc",
       "2011/SNDR.AQUA.AIRS_IM.20111001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210616071401.nc",
       "2012/SNDR.AQUA.AIRS_IM.20121001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210615024728.nc",
       "2013/SNDR.AQUA.AIRS_IM.20131001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210612032722.nc",
       "2014/SNDR.AQUA.AIRS_IM.20141001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210612050102.nc",
       "2015/SNDR.AQUA.AIRS_IM.20151001.M01.L3_CLIMCAPS_QCC.std.v02_38.G.210612045556.nc"]
urls = ["https://sounder.gesdisc.eosdis.nasa.gov/opendap/Aqua_Sounder_Level3/SNDRAQIML3CMCCP.2/"+ p for p in paths]



import xarray as xr
from pydap.client import open_url
from pydap.cas.urs import setup_session

username = 'qnicolas'
password = 'Surf@b3rkeley!'

session = setup_session(username, password, check_url=url)
store = xr.backends.PydapDataStore.open(url, session=session)
ds = xr.open_dataset(store)
ds=ds.rename(lat='latitude',lon='longitude')
ds=ds.assign_coords(level=ds['air_pres']/100)

ds_empty = 0.*ds.air_temp.expand_dims({'year':np.arange(2003,2016,1)})#.expand_dims("year")

ds_empty[1]=ds.air_temp.rename(lat='latitude',lon='longitude')

for y in range(2005,2016):
    url = urls[y-2003]
    print(y,url)
    session = setup_session(username, password, check_url=url)
    store = xr.backends.PydapDataStore.open(url, session=session)
    ds = xr.open_dataset(store)
    ds_empty[y-2003]=ds.air_temp.rename(lat='latitude',lon='longitude')
    

t_climcaps_all = ds_empty
