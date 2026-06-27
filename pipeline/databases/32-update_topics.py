#!/usr/bin/env python3
"""Module that updates the topics of school documents by name."""


def update_topics(mongo_collection, name, topics):
    """Replace the topics list of all schools matching name."""
    mongo_collection.update_many(
        {"name": name},
        {"$set": {"topics": topics}}
    )
