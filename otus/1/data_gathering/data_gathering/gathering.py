"""
ЗАДАНИЕ

Выбрать источник данных и собрать данные по некоторой предметной области.

Цель задания - отработать навык написания программ на Python.
В процессе выполнения задания затронем области:
- организация кода в виде проекта, импортирование модулей внутри проекта
- unit тестирование
- работа с файлами
- работа с протоколом http
- работа с pandas
- логирование

Требования к выполнению задания:

- собрать не менее 1000 объектов

- в каждом объекте должно быть не менее 5 атрибутов
(иначе просто будет не с чем работать.
исключение - вы абсолютно уверены что 4 атрибута в ваших данных
невероятно интересны)

- сохранить объекты в виде csv файла

- считать статистику по собранным объектам


Этапы:

1. Выбрать источник данных.

Это может быть любой сайт или любое API

Примеры:
- Пользователи vk.com (API)
- Посты любой популярной группы vk.com (API)
- Фильмы с Кинопоиска
(см. ссылку на статью ниже)
- Отзывы с Кинопоиска
- Статьи Википедии
(довольно сложная задача,
можно скачать дамп википедии и распарсить его,
можно найти упрощенные дампы)
- Статьи на habrahabr.ru
- Объекты на внутриигровом рынке на каком-нибудь сервере WOW (API)
(желательно англоязычном, иначе будет сложно разобраться)
- Матчи в DOTA (API)
- Сайт с кулинарными рецептами
- Ebay (API)
- Amazon (API)
...

Не ограничивайте свою фантазию. Это могут быть любые данные,
связанные с вашим хобби, работой, данные любой тематики.
Задание специально ставится в открытой форме.
У такого подхода две цели -
развить способность смотреть на задачу широко,
пополнить ваше портфолио (вы вполне можете в какой-то момент
развить этот проект в стартап, почему бы и нет,
а так же написать статью на хабр(!) или в личный блог.
Чем больше у вас таких активностей, тем ценнее ваша кандидатура на рынке)

2. Собрать данные из источника и сохранить себе в любом виде,
который потом сможете преобразовать

Можно сохранять страницы сайта в виде отдельных файлов.
Можно сразу доставать нужную информацию.
Главное - постараться не обращаться по http за одними и теми же данными много раз.
Суть в том, чтобы скачать данные себе, чтобы потом их можно было как угодно обработать.
В случае, если обработать захочется иначе - данные не надо собирать заново.
Нужно соблюдать "этикет", не пытаться заддосить сайт собирая данные в несколько потоков,
иногда может понадобиться дополнительная авторизация.

В случае с ограничениями api можно использовать time.sleep(seconds),
чтобы сделать задержку между запросами

3. Преобразовать данные из собранного вида в табличный вид.

Нужно достать из сырых данных ту самую информацию, которую считаете ценной
и сохранить в табличном формате - csv отлично для этого подходит

4. Посчитать статистики в данных
Требование - использовать pandas (мы ведь еще отрабатываем навык использования инструментария)
То, что считаете важным и хотели бы о данных узнать.

Критерий сдачи задания - собраны данные по не менее чем 1000 объектам (больше - лучше),
при запуске кода командой "python3 -m gathering stats" из собранных данных
считается и печатается в консоль некоторая статистика

Код можно менять любым удобным образом
Можно использовать и Python 2.7, и 3

Зачем нужны __init__.py файлы
https://stackoverflow.com/questions/448271/what-is-init-py-for

Про документирование в Python проекте
https://www.python.org/dev/peps/pep-0257/

Про оформление Python кода
https://www.python.org/dev/peps/pep-0008/


Примеры сбора данных:
https://habrahabr.ru/post/280238/

Для запуска тестов в корне проекта:
python3 -m unittest discover

Для запуска проекта из корня проекта:
python3 -m gathering gather
или
python3 -m gathering transform
или
python3 -m gathering stats


Для проверки стиля кода всех файлов проекта из корня проекта
pep8 .

"""

import logging
import sys
import pandas as pd

from parsers.filter_parser import FilterParser
from parsers.json_parser import JsonParser
from scrappers.scrapper import Scrapper
from storages.file_storage import FileStorage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

SCRAPPED_FILE = 'scrapped_data.txt'
TABLE_FORMAT_FILE = 'data.csv'


def gather_process():
    logger.info("gather")
    storage = FileStorage(SCRAPPED_FILE)

    # You can also pass a storage
    scrapper = Scrapper()
    scrapper.scrap_process(storage)


def convert_data_to_table_format():
    logger.info("transform")
    storage = FileStorage(SCRAPPED_FILE)
    FileStorage(TABLE_FORMAT_FILE)
    columns = ['id', 'text','date', 'attachments.link.url', 'attachments.link.title', 'comments.count', 'reposts.count', 'views.count']
    filter_parser = FilterParser(columns)
    json_parser = JsonParser(filter_parser)
    parsed_data = json_parser.parse(storage)
    df = pd.DataFrame(data=parsed_data, columns=columns)
    df.to_csv(TABLE_FORMAT_FILE)

def stats_of_data():
    logger.info("stats")
    df = pd.DataFrame.from_csv(TABLE_FORMAT_FILE)
    print('Первые 10 записей')
    print(df.head(10))
    
    print()
    print('describe')
    print(df.describe())
    
    print()
    print('shape')
    print(df.shape)
    
    print()    
    print('info')
    print(df.info())
    pd.set_option('display.max_colwidth', -1)
    print()
    print('Статистика комментариев, репостов, просмотров')
    print(df[['comments.count', 'reposts.count', 'views.count']].describe())


    print()
    print()
    print('Наиболее комментируемые статьи')
    print(df.loc[df['comments.count']==df['comments.count'].max(), ['text', 'comments.count']])
    print('Наиболее репостируемые статьи')
    print(df.loc[df['reposts.count']==df['reposts.count'].max(), ['text', 'reposts.count']])
    print('Наиболее просмотренные статьи')
    print(df.loc[df['views.count']==df['views.count'].max(), ['text','views.count']])


    print()
    print()    
    print('Наименее комментируемые статьи')
    print(df.loc[df['comments.count']==df['comments.count'].min(), ['text', 'attachments.link.title', 'comments.count']])
    print('Наименее репостируемые статьи')
    print(df.loc[df['reposts.count']==df['reposts.count'].min(), ['text', 'attachments.link.title','reposts.count']])
    print('Наименее просмотренные статьи')
    print(df.loc[df['views.count']==df['views.count'].min(), ['text',  'attachments.link.title', 'views.count']])
    
    #Насчет Сталина и Кадырова - удивило

    print()
    print()    
    df['date_pd'] = pd.to_datetime(df['date'],unit='s')
    df['year'] = df.apply(lambda row: row['date_pd'].year, axis=1)
    df['month'] = df.apply(lambda row: row['date_pd'].month, axis=1)
    df['day'] = df.apply(lambda row: row['date_pd'].day, axis=1)
    df['is_weekend'] = df.apply(lambda row: row['date_pd'].weekday()>4, axis=1)
    df['weekday'] = df.apply(lambda row: row['date_pd'].weekday(), axis=1)
    print('За какие периоды у нас данные')
    print(df[['year', 'month', 'day', 'weekday']].describe())
    gb = df.groupby(by = 'weekday')
    print('Статистика суммарных просмотров по дням недели в среднем')
    print(gb[['comments.count','reposts.count','views.count']].mean())
    
if __name__ == '__main__':
    """
    why main is so...?
    https://stackoverflow.com/questions/419163/what-does-if-name-main-do
    """
    logger.info("Work started")

    if sys.argv[1] == 'gather':
        gather_process()

    elif sys.argv[1] == 'transform':
        convert_data_to_table_format()

    elif sys.argv[1] == 'stats':
        stats_of_data()

    logger.info("work ended")
