#!/usr/bin/env python3
"""Displays the upcoming SpaceX launch."""
import requests


if __name__ == '__main__':
    url = "https://api.spacexdata.com/v4/launches/upcoming"
    launches = requests.get(url).json()

    upcoming = sorted(launches, key=lambda x: x['date_unix'])[0]

    name = upcoming['name']
    date = upcoming['date_local']

    rocket_id = upcoming['rocket']
    rocket_url = "https://api.spacexdata.com/v4/rockets/" + rocket_id
    rocket = requests.get(rocket_url).json()['name']

    pad_id = upcoming['launchpad']
    pad_url = "https://api.spacexdata.com/v4/launchpads/" + pad_id
    pad = requests.get(pad_url).json()
    pad_name = pad['name']
    pad_loc = pad['locality']

    print("{} ({}) {} - {} ({})".format(
        name, date, rocket, pad_name, pad_loc))
