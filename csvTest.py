import csv
import matplotlib as plt


headers = ['Symbol', 'Price', 'Date', 'Time', 'Change', 'Volume']
rows = [{'Symbol':'AA', 'Price':39.48, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.18, 'Volume':181800},
        {'Symbol':'AIG', 'Price': 71.38, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.15, 'Volume': 195500},
        {'Symbol':'AXP', 'Price': 62.58, 'Date':'6/11/2007',
        'Time':'9:36am', 'Change':-0.46, 'Volume': 935000},
        ]

with open('stocks.csv','w') as f:
    # 有序字典方式写入数据
    f_csv = csv.DictWriter(f, headers)
    f_csv.writeheader()
    f_csv.writerows(rows)

with open('result.csv', 'r') as fo:


    # 一般方式读入，无法按照列名查询数据
    # fo_csv = csv.reader(fo)

    # 有序词典方式读入，可按照列名查询数据
    fo_csv = csv.DictReader(fo)
    for row in fo_csv:
        print(row['name'])