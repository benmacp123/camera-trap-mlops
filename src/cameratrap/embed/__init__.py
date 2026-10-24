"""Crop embeddings, computed once per backbone and cached.

One of two subpackages allowed to import the model stack (torch and
friends); everything downstream reads the cached tables.
"""
