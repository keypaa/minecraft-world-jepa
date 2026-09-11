"""Compatibility entrypoint for Modal inference.

The canonical inference implementation lives in app.py so training and serving
share the same model, VAE, latent-stat, and Modal API assumptions.
"""

from app import WorldModelInference, app
