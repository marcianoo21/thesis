import json
from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="hid_training")


with open("output_files/lodz_restaurants_cafes_ready_for_embd.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        geo_coords = record["Współrzędne"]
        location  = geolocator.reverse(geo_coords)
        # location  = geolocator.reverse("51.767730, 19.468667")
        print(location)