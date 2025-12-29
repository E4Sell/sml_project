"""
Storage Factory - Unified interface for local and production storage

This module provides a consistent API that works with both:
- Local storage (Parquet files) for testing
- Hopsworks Feature Store for production

Usage:
    storage = get_storage(mode='local')  # or mode='production'
    fs = storage.get_feature_store()
    fg = fs.get_or_create_feature_group(name='electricity_price', version=1)
    fg.insert(df)
"""
from dotenv import load_dotenv
load_dotenv()

import os
from typing import Literal

StorageMode = Literal['local', 'production']


class StorageFactory:
    """Factory to create storage backends based on mode"""

    @staticmethod
    def get_storage(mode: StorageMode = 'local'):
        """
        Get storage backend based on mode

        Args:
            mode: 'local' for local Parquet storage, 'production' for Hopsworks

        Returns:
            Storage backend (LocalProject or HopsworksProject)
        """
        if mode == 'local':
            from functions.local_storage import get_local_project
            return get_local_project()
        elif mode == 'production':
            return HopsworksStorage()
        else:
            raise ValueError(f"Invalid mode: {mode}. Must be 'local' or 'production'")


class HopsworksStorage:
    """
    Wrapper for Hopsworks that provides same interface as LocalProject
    """

    def __init__(self):
        try:
            import hopsworks
            self._hopsworks = hopsworks
            self._project = None
            self._fs = None
        except ImportError:
            raise ImportError(
                "Hopsworks not installed. Install with: pip install hopsworks\n"
                "Or use --mode local for local testing"
            )

    def get_feature_store(self):
        """Connect to Hopsworks and return feature store"""
        if self._project is None:
            # Check for API key
            api_key = os.getenv('HOPSWORKS_API_KEY')
            if not api_key:
                raise ValueError(
                    "HOPSWORKS_API_KEY not found in environment.\n"
                    "Set it with: export HOPSWORKS_API_KEY='your-key'\n"
                    "Or use --mode local for local testing"
                )

            print("Connecting to Hopsworks...")
            self._project = self._hopsworks.login()
            self._fs = self._project.get_feature_store()
            print(f"✅ Connected to Hopsworks project: {self._project.name}")

        return self._fs

    def get_model_registry(self):
        """Get Hopsworks model registry"""
        if self._project is None:
            self.get_feature_store()  # Initialize connection
        return self._project.get_model_registry()


def get_storage(mode: StorageMode = 'local'):
    """
    Convenience function to get storage backend

    Args:
        mode: 'local' or 'production'

    Returns:
        Storage backend with consistent API

    Example:
        >>> storage = get_storage(mode='local')
        >>> fs = storage.get_feature_store()
        >>> fg = fs.get_or_create_feature_group('electricity_price', version=1)
        >>> fg.insert(df)
    """
    return StorageFactory.get_storage(mode)


def detect_mode() -> StorageMode:
    """
    Auto-detect mode based on environment

    Returns:
        'production' if HOPSWORKS_API_KEY is set, otherwise 'local'
    """
    if os.getenv('HOPSWORKS_API_KEY'):
        return 'production'
    return 'local'
