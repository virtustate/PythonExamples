import logging
import sqlalchemy
import pg8000
#import psycopg2
import json
import os

logger = logging.getLogger()
db_user = os.environ['DB_USER']
db_password = os.environ['DB_PASSWORD']
db_name = os.environ['DB_NAME']
connection_name = os.environ['CONNECTION_NAME']
driver_name = 'postgres+pg8000'
db_socket_dir = "/cloudsql"
# other ways of creating connection
# url = 'postgres+pg8000://db_user:db_password@/message?unix_sock=/cloudsql/conneciton_name/.s.PGSQL.5432'
# db = sqlalchemy.create_engine(url)
# db = sqlalchemy.create_engine("postgresql://db_user:db_password@server_ip5432/message")
db = sqlalchemy.create_engine(
    sqlalchemy.engine.url.URL(
        drivername=driver_name,
        username=db_user,
        password=db_password,
        database=db_name,
        query={"unix_sock": "{}/{}/.s.PGSQL.5432".format(db_socket_dir, connection_name)}
    ),
    pool_size=5,
    max_overflow=2,
    pool_timeout=30,
    pool_recycle=1800
)


def insert(request):
    data = {"json": json.dumps(request.get_json())}

    stmt = sqlalchemy.text("""insert into message(json) values(:json)""")
    try:
        with db.connect() as conn:
            conn.execute(stmt, **data)
    except Exception as e:
        return 'Error: {}'.format(str(e))
    return 'ok'


def hello_world(request):
    logger.warning(insert(request))
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        return request.args.get('message')
    elif request_json:
        return request.json.get('challenge', '')
    else:
        return f'Hello World!'
