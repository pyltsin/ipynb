from parsers.parser import Parser


class FilterParser(Parser):
    """This is example class. There is no big purpose for it"""
    def __init__(self, columns):
        self._columns = columns
        pass

    def parse(self, data):
        row =[]
        for column in self._columns:
            inner = data
            for path in column.split('.'):
                if not path in inner:
                    inner = None
                    break
                inner = inner[path]
                if isinstance(inner, list):
                    if len(inner)>0:
                        inner = inner[0]
                    else:
                        inner = None
                        break
            row.append(inner)
        return row