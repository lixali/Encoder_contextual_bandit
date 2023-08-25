<<<<<<< HEAD
import requests
import json
import time
import random

api_key = "vvfmGQCTjP0bmz8DpXBYSq8kHVD8odrn"
url = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

for year in range(2019, 2023):
    month=1
    while month <13:
        filename = f"news_{year}_{month}.json"
        with open(filename, "w") as f:
            params = {
                "api-key": api_key
            }
            url_cur = url.format(year=year, month=month)
            print(url_cur)
            response = requests.get(url_cur, params=params)
            if response.status_code == 200:
                data = json.loads(response.text)
                json.dump(data, f)
                month+=1
            else:
                print(f"Error: {response.status_code}")
            time.sleep(random.randint(12, 17))



=======
import requests
import json
import time
import random

api_key = "vvfmGQCTjP0bmz8DpXBYSq8kHVD8odrn"
url = "https://api.nytimes.com/svc/archive/v1/{year}/{month}.json"

for year in range(2019, 2023):
    month=1
    while month <13:
        filename = f"news_{year}_{month}.json"
        with open(filename, "w") as f:
            params = {
                "api-key": api_key
            }
            url_cur = url.format(year=year, month=month)
            print(url_cur)
            response = requests.get(url_cur, params=params)
            if response.status_code == 200:
                data = json.loads(response.text)
                json.dump(data, f)
                month+=1
            else:
                print(f"Error: {response.status_code}")
            time.sleep(random.randint(12, 17))
>>>>>>> 0f0e8e7bffd20db1438c43edfd650580b466fdbf
