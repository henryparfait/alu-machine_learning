#!/usr/bin/env python3
"""Displays the number of launches per SpaceX rocket."""
import requests


if __name__ == '__main__':
    launches = requests.get(
        "https://api.spacexdata.com/v4/launches").json()

    counts = {}
    for launch in launches:
        rocket_id = launch['rocket']
        counts[rocket_id] = counts.get(rocket_id, 0) + 1

    rockets = {}
    for rocket_id in counts:
        name = requests.get(
            "https://api.spacexdata.com/v4/rockets/" + rocket_id
        ).json()['name']
        rockets[name] = counts[rocket_id]

    ordered = sorted(rockets.items(), key=lambda x: (-x[1], x[0]))
    for name, count in ordered:
        print("{}: {}".format(name, count))
