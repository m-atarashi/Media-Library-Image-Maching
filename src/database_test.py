import database
import json

with open('/src/shelf_matching/data/settings.json', 'r') as f:
    SETTINGS = json.loads(f)

HOST     = SETTINGS['HOST']
ROOT     = SETTINGS['ROOT']
PASSWORD = SETTINGS['PASSWORD']
DATABASE = SETTINGS['DATABASE']


# This programe depends on the structure of the table "logdata"
# +----------+-------------+------+-----+-------------------+-------------------+
# | Field    | Type        | Null | Key | Default           | Extra             |
# +----------+-------------+------+-----+-------------------+-------------------+
# | id       | int         | NO   | PRI | NULL              | auto_increment    |
# | userid   | varchar(16) | YES  |     | NULL              |                   |
# | datetime | datetime    | YES  |     | CURRENT_TIMESTAMP | DEFAULT_GENERATED |
# | log      | longtext    | YES  |     | NULL              |                   |
# +----------+-------------+------+-----+-------------------+-------------------+
if __name__=='__main__':
    answers = ['B1-2', 'B1-2', 'B20-1', 'B41-2', 'None', 'B46-1']
    shelf_counts_json = database.shelf_counts_json(answers)
    print(shelf_counts_json)

    user_id = 'm-atarashi'
    sql = 'INSERT INTO logdata (userid, log) VALUES (%s, %s)'
    values = (user_id, shelf_counts_json)
    database.insert_logdata_into_database(HOST, ROOT, PASSWORD, DATABASE, sql, values)
