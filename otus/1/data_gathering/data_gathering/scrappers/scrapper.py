import json
import logging

import requests

logger = logging.getLogger(__name__)

COUNT = 100


class Scrapper(object):
    def __init__(self, skip_objects=None):
        self.skip_objects = skip_objects

    def scrap_process(self, storage):
        text_array=[]
        for i in range(10):
            offset = i * COUNT
            url = 'https://api.vk.com/method/wall.get?domain=ria&count=100&offset={}&access_token' \
                  '=1c341220ef87a3859825174bb8211833e54378b24d79da8643ecc63cdf8bc799f38403f7c430aec6d78d4&v=5.92'.format(
                offset)
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.9; rv:45.0) Gecko/20100101 Firefox/45.0'
            }
            response = requests.get(url, headers=headers)
            # r.encoding = 'cp1251'
            if not response.ok:
                logger.error(response.text)
            # then continue process, or retry, or fix your code

            else:
                print(response.json())
                text_array.append(response.json())

        storage.write_data([json.dumps(text_array)])
