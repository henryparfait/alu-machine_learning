#!/usr/bin/env python3
"""Module that lists all documents in a MongoDB collection."""


def list_all(mongo_collection):
    """Return a list of all documents, or an empty list if none."""
    return list(mongo_collection.find())
