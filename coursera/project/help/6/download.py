from itertools import product
from tqdm import tqdm
import requests


#  = "https://s3.amazonaws.com/nyc-tlc/trip+data/yellow_tripdata_2009-12.csv"
# response = requests.get(url, stream=True)
#
# with open("/media/vad/DATA", "wb") as handle:
#     for data in tqdm(response.iter_content()):
#         handle.write(data)

for year,month in product(range(10,17),range(1,23)):
    data = "yellow_tripdata_20%02d-%02d.csv"%(year,month)
    url="https://s3.amazonaws.com/nyc-tlc/trip+data/"+data
    print url
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0));
    with open("/media/vad/DATA/"+data, "wb") as handle:
        for data in tqdm(response.iter_content(32*1024),total=total_size, unit='B', unit_scale=True):
            handle.write(data)