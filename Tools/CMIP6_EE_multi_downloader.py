import ee
import geemap
import logging
import multiprocessing
from retry import retry
import geopandas as gpd
import requests

try:
    ee.Initialize()
except Exception as e:
    ee.Authenticate()         # authenticate when using GEE for the first time
    ee.Initialize()
import os
os.chdir("/home/phillip/Seafile/EBA-CA/Repositories/ee_download_test")

"""
This tool downloads data from Earth Engine using parallel requests.
It extracts the timeseries of 8-day max LST from MODIS MOD11A2 per GAUL level-2 region
for all regions in South America, with each time-series written to its own file.
"""

output_gpkg = '/home/phillip/Seafile/EBA-CA/Repositories/matilda_edu/output/catchment_data.gpkg'

catchment_new = gpd.read_file(output_gpkg, layer='catchment_new')
catchment = geemap.geopandas_to_ee(catchment_new)
##

def getRequests(starty, endy):
    """Generates a list of years as work items to be downloaded."""
    return [i for i in range(starty, endy)]


@retry(tries=10, delay=1, backoff=2)
def getResult(index, year):
    """Handle the HTTP requests to download one year of CMIP6 data."""

    start = str(year) + '-01-01'
    end = str(year + 1) + '-01-01'

    startDate = ee.Date(start)
    endDate = ee.Date(end)
    n = endDate.difference(startDate, 'day').subtract(1)

    def getImageCollection(var):
        collection = ee.ImageCollection('NASA/GDDP-CMIP6') \
            .select(var) \
            .filterDate(startDate, endDate) \
            .filterBounds(catchment)
        return collection

    def renameBandName(b):
        split = ee.String(b).split('_')
        return ee.String(split.splice(split.length().subtract(2), 1).join("_"))

    def buildFeature(i):
        t1 = startDate.advance(i, 'day')
        t2 = t1.advance(1, 'day')
        # feature = ee.Feature(point)
        dailyColl = collection.filterDate(t1, t2)
        dailyImg = dailyColl.toBands()
        # renaming and handling names
        bands = dailyImg.bandNames()
        renamed = bands.map(renameBandName)
        # Daily extraction and adding time information
        dict = dailyImg.rename(renamed).reduceRegion(
            reducer=ee.Reducer.mean(),
            geometry=catchment,
        ).combine(
            ee.Dictionary({'system:time_start': t1.millis(), 'isodate': t1.format('YYYY-MM-dd')})
        )
        return ee.Feature(None, dict)

    # Create features for all days in the respective year
    collection = getImageCollection('tas')
    year_feature = ee.FeatureCollection(ee.List.sequence(0, n).map(buildFeature))

    # Create a download URL for a CSV containing the feature collection
    url = year_feature.getDownloadURL()

    # Handle downloading the actual annual csv
    r = requests.get(url, stream=False)
    if r.status_code != 200:
        r.raise_for_status()

    filename = 'results_%d.csv' % year
    with open(filename, 'w') as f:
        f.write(r.text)

    print("Done: ", index)


# if __name__ == '__main__':
logging.basicConfig()
items = getRequests(1979, 2100)

pool = multiprocessing.Pool(25)
pool.starmap(getResult, enumerate(items))

pool.close()
pool.join()


##
# --> tas und var getrennt anschmeissen
# --> mögliche "IDs": models, jahre, scenarios
#       --> je mehr prozesse desto besser? Also einfach einzelne jahre (2x 122 files
#       oder gar tage (über 44k files...)?
# --> buildFeature als mappable function:
    # Schreibt daily values für einen tag in ein feature. i ist dabei der zeitschritt seit start in tagen
    # wird dann gemapped und mit täglichen zeitschritten von 0 bis n gefüttert (sequence(0,n).map(buildFeatures))
# --> getImageCollection als region = ... function

# --> getRequest muss

##
def getRequests():
  """Generates a list of work items to be downloaded.
  Extract the ADM2_CODEs from the GAUL level 2 dataset as work units.
  """
  southAmerica = ee.Geometry.BBox(-84.0, -56.5, -32.0, 12.5)
  gaul2 = (ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level2')
           .filterBounds(southAmerica))
  return gaul2.aggregate_array('ADM2_CODE').getInfo()


@retry(tries=10, delay=1, backoff=2)
def getResult(index, regionID):
  """Handle the HTTP requests to download one result."""
  region = (ee.FeatureCollection('FAO/GAUL_SIMPLIFIED_500m/2015/level2')
            .filter(ee.Filter.eq('ADM2_CODE', regionID))
            .first())

  def maxLST(image):
    # Mappable function to aggregate the max LST for one image.
    # It builds an output tuple of (max_LST, date, regionID)
    # This function uses -999 to indicate no data.
    date = image.date().format('YYYY-MM-dd')
    image = image.multiply(0.02).subtract(273.15)
    maxValue = (image.reduceRegion(ee.Reducer.max(), region.geometry())
                # set a default in case there's no data in the region
                .combine({'LST_Day_1km': -999}, False)
                .getNumber('LST_Day_1km')
                # format to 2 decimal places.
                .format('%.2f'))

    return image.set('output', [maxValue, date, regionID])

  # Get the max LST for this region, in each image.
  timeSeries = (ee.ImageCollection('MODIS/006/MOD11A2')
                .select('LST_Day_1km')
                .map(maxLST))
  result = timeSeries.aggregate_array('output').getInfo()

  # Write the results to a file.
  filename = 'results_%d.csv' % regionID
  with open(filename, 'w') as out_file:
    for items in result:
      line = ','.join([str(item) for item in items])
      print(line, file=out_file)

  print("Done: ", index)


if __name__ == '__main__':
  logging.basicConfig()
  items = getRequests()

  pool = multiprocessing.Pool(25)
  pool.starmap(getResult, enumerate(items))   # enumerate(items) returns tuples with an index starting with 0 and the regionIDs returned by getRequests() like (index, regionID)
                                            # starmap unpacks enumerate(items) as arguments and passes them to getResult()
  pool.close()
  pool.join()