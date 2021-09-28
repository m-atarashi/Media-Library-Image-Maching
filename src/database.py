import json
from collections import Counter

import pymysql.cursors


def shelf_counts_json(answers):
    """マッチング結果を集計し、SQLのデータベースに保存する

    入力は「B書架番号-0or1」のリスト。
    書架番号のハイフン以下を削除する。

    書架番号：B1-B42、B29+をkeyにとり、初期値0をとるordereddictを作成。
    [(B1, 0), ..., (B42, 0)]
    Counterで書架番号の出現回数を集計する。
    orderddictの値を更新。

    Args:
        answers ([type]): [description]
    """

    shelf_labels = ['B'+str(i) for i in range(1, 43)]
    shelf_labels.insert(29, 'B29+')

    shelf_counts = dict()
    answers_counts = Counter([answer[:answer.find('-')] for answer in answers])
    for key in shelf_labels:
        if key in answers_counts:
            shelf_counts[key] = str(answers_counts[key])
        else:
            shelf_counts[key] = '0'
    
    return json.dumps(shelf_counts)


def insert_logdata_into_database(host, user_name, password, db_name, sql, values):
    """[summary]
    sqlのデータベースを読み込む。
    json_objectをデータベースに挿入する。

    Args:
        host ([type]): [description]
        user_name ([type]): [description]
        db_name ([type]): [description]
        sql ([type]): [description]
        values ([type]): [description]
    """

    con = pymysql.connect(host=host,
                    user=user_name,
                    password=password,
                    db=db_name,
                    charset='utf8mb4',
                    cursorclass=pymysql.cursors.DictCursor)
    try:
        with con.cursor() as cursor:
            cursor.execute(sql, values)
        con.commit()
    finally:
        con.close()
