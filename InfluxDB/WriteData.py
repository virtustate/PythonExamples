from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS
import math
import datetime
import json


def get_influx_client():
    with open('./config.json', 'r') as f:
        config = json.load(f)
    return InfluxDBClient(url=config['URL'], org=config['ORG'], token=config['TOKEN'])


def get_write_api():
    return get_influx_client().write_api(write_options=SYNCHRONOUS)


def write_points():
    write_api = get_write_api()
    number_of_points = 2500
    current_point_time = datetime.datetime.utcnow() - datetime.timedelta(seconds=number_of_points)
    for i in range(number_of_points):
        p = Point("m1").tag('test', 'test').field("x10", round(math.sin(float(i) / 10.0), 6)). \
            field("x100", round(math.sin(float(i) / 100.0), 6)). \
            field("x1000", round(math.sin(float(i) / 1000.0), 6)).time(time=current_point_time)
        current_point_time = current_point_time + datetime.timedelta(seconds=1)
        write_api.write(bucket='waves', record=p)
    print('done')


def write_pandas():
    write_api = get_write_api()


if __name__ == "__main__":
    write_points()
