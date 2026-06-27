#!/usr/bin/env python3
"""Module that inserts a new document in a MongoDB collection."""


def insert_school(mongo_collection, **kwargs):
    """Insert a document built from kwargs and return its new _id."""
    result = mongo_collection.insert_one(kwargs)
    return result.inserted_id
