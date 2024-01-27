import asyncio
import aiohttp
from datetime import datetime
import os
import requests
import time
import config


def init_coords():
    try:
        with open(f'{config.IMG_DIR}/log.txt') as f:
            logs = f.read().split('\n')

        for log in reversed(logs):
            if 'Bbox' in log:
                lat, lng = log.split(': ')[1].split(', ')
                print(f'Continuing from prev coords: {lat}, {lng}')
                break

    except FileNotFoundError:
        lat = config.MIN_LAT
        lng = config.MIN_LNG
        print(f'Using new coords: {lat}, {lng}')

    return float(lat), float(lng)


async def get(img_url, img_name):
    async with aiohttp.ClientSession() as session:
        async with session.get(img_url) as response:
            content = await response.read()
    with open(os.path.join(config.IMG_DIR, f'{img_name}.jpg'), 'wb') as f:
        f.write(content)
    print(img_name)
    with open(os.path.join(config.IMG_DIR, 'log.txt'), 'a') as f:
        f.write(f'{img_name}\n')


async def download_imgs(img_urls):
    gets = []
    for img_name, img_url in img_urls.items():
        gets.append(get(img_url, img_name))
    await asyncio.gather(*gets)


if __name__ == '__main__':
    if config.IMG_DIR not in os.listdir():
        os.mkdir(config.IMG_DIR)
    lat, lng = init_coords()
    url = 'https://graph.mapillary.com/images'
    params = {
        'fields': 'geometry,captured_at,thumb_1024_url',
        'is_pano': False,
        'limit': config.BBOX_NUM_IMG,
    }
    headers = {
        'Authorization': f'OAuth {config.TOKEN}',
    }

    while round(lat+config.BBOX_SIZE, config.COORD_ACC) < config.MAX_LAT:
        next_lat = round(lat+config.BBOX_SIZE, config.COORD_ACC)

        while round(lng+config.BBOX_SIZE, config.COORD_ACC) < config.MAX_LNG:
            bbox_log = f'Bbox: {lat}, {lng}'
            print(bbox_log)
            with open(os.path.join(config.IMG_DIR, 'log.txt'), 'a') as f:
                f.write(f'{bbox_log}\n')

            next_lng = round(lng+config.BBOX_SIZE, config.COORD_ACC)
            params['bbox'] = f'{lng},{lat},{next_lng},{next_lat}' # minLon, minLat, maxLon, maxLat of "bbox"
            r = requests.get(url, params=params, headers=headers)
            data = r.json()['data']

            img_urls = {}
            for i in data:
                if i['geometry']['type'] == 'Point':
                    img_lng, img_lat = i['geometry']['coordinates']
                    captured_at = datetime.fromtimestamp(i['captured_at']//1000)
                    captured_at_str = captured_at.strftime("%Y.%m.%d_%H.%M.%S")
                    try:
                        img_url = i['thumb_1024_url']
                        img_name = f'{img_lat}_{img_lng}@{captured_at_str}'
                        img_urls[img_name] = img_url
                    except KeyError as e:
                        print(e)

            if config.ASYNC:
                asyncio.run(download_imgs(img_urls))
            else:
                for img_name, img_url in img_urls.items():
                    r = requests.get(img_url)
                    with open(os.path.join(config.IMG_DIR, f'{img_name}.jpg'), 'wb') as f:
                        f.write(r.content)
                    print(img_name)
                    with open(os.path.join(config.IMG_DIR, 'log.txt'), 'a') as f:
                        f.write(f'{img_name}\n')

            lng = next_lng
            time.sleep(0.01)

        lng = config.MIN_LNG
        lat = next_lat

    print('Done!')
