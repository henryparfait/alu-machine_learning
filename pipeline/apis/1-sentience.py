#!/usr/bin/env python3
"""Returns the home planets of all sentient species."""
import requests


def sentientPlanets():
    """Returns the list of home planet names of all sentient species."""
    planets = []
    url = "https://swapi-api.alx-tools.com/api/species/"
    while url:
        response = requests.get(url).json()
        for species in response['results']:
            if species['designation'] == 'sentient' or \
                    species['classification'] == 'sentient':
                home = species['homeworld']
                if home is not None:
                    planets.append(requests.get(home).json()['name'])
        url = response['next']
    return planets
