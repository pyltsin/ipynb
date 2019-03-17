import json

from scrapper.parsers.parser import Parser


class JsonParser(Parser):
    """This is example class. There is no big purpose for it"""

    def __init__(self, filters):
        self._filters = filters

    def parse(self, data_file):
        rows = []
        for line in data_file.read_data():
            array_json = json.loads(line)
            for responce in array_json:
                print(responce)
                for news in responce['response']:
                    row = self._filters.parse(news)
                    rows.append(row)
        return rows
