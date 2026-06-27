#!/usr/bin/env python3
"""Module that finds schools that teach a specific topic."""


def schools_by_topic(mongo_collection, topic):
    """Return the list of schools whose topics include topic."""
    return list(mongo_collection.find({"topics": topic}))
