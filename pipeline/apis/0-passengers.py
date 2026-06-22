#!/usr/bin/env python3
"""Returns the list of ships that can hold a given number of passengers."""
import requests


def availableShips(passengerCount):
    """Returns ships from the SWAPI that hold >= passengerCount."""
    ships = []
    url = "https://swapi-api.alx-tools.com/api/starships/"
    while url:
        response = requests.get(url).json()
        for ship in response['results']:
            passengers = ship['passengers'].replace(',', '')
            if passengers not in ('n/a', 'unknown') and \
                    int(passengers) >= passengerCount:
                ships.append(ship['name'])
        url = response['next']
    return ships
