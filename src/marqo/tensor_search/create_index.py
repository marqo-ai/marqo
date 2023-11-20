"""Functions used to fulfill the create_vector_index endpoint"""
import copy
from contextlib import contextmanager

import math
import threading
import random

from typing import List, Optional, Tuple, ContextManager, Union
import PIL
from PIL.ImageFile import ImageFile
from marqo.s2_inference import clip_utils
from marqo.tensor_search.telemetry import RequestMetricsStore, RequestMetrics
import marqo.errors as errors
from marqo.tensor_search import utils
from marqo.tensor_search.enums import IndexSettingsField, ModelProperties


def override_prefixes_in_model_properties(index_settings: dict) -> dict:
    """
    Modifies index settings.
    Use overrides specified in text_preprocessing to override:
    1. text_chunk_prefix in model_properties
    2. text_query_prefix in search_model_properties
    3. text_query_prefix in model_properties (for backwards compatibility with v1.4.0 and earlier)

    If override is None, do NOT override the prefix.
    """

    needed_settings_keys = [IndexSettingsField.text_preprocessing, IndexSettingsField.model_properties]
    for key in needed_settings_keys:
        if key not in index_settings:
            raise errors.InternalError(f"Missing required field {key} in index_settings.")
    
    copied_settings = index_settings.copy()

    # Chunk prefix
    if IndexSettingsField.override_text_chunk_prefix in index_settings[IndexSettingsField.text_preprocessing]:
        # Ignore if override is None
        if index_settings[IndexSettingsField.text_preprocessing][IndexSettingsField.override_text_chunk_prefix] is not None:
            # Override chunk prefix in model properties
            copied_settings[IndexSettingsField.model_properties][ModelProperties.text_chunk_prefix] = \
                index_settings[IndexSettingsField.text_preprocessing][IndexSettingsField.override_text_chunk_prefix]
    
    # Query prefix
    if IndexSettingsField.override_text_query_prefix in index_settings[IndexSettingsField.text_preprocessing]:
        # Ignore if override is None
        if index_settings[IndexSettingsField.text_preprocessing][IndexSettingsField.override_text_query_prefix] is not None:

            # Override query prefix in search model properties (if it exists)
            if IndexSettingsField.search_model_properties in index_settings:
                copied_settings[IndexSettingsField.search_model_properties][ModelProperties.text_query_prefix] = \
                    index_settings[IndexSettingsField.text_preprocessing][IndexSettingsField.override_text_query_prefix]
            
            # Override query prefix in model properties
            copied_settings[IndexSettingsField.model_properties][ModelProperties.text_query_prefix] = \
                index_settings[IndexSettingsField.text_preprocessing][IndexSettingsField.override_text_query_prefix]
    
    return copied_settings


def autofill_search_model(index_settings: dict):
    """
    Autofills `search_model` and `search_model_properties` in index settings if not provided.
    This defaults to the same values as `model` and `model_properties` respectively.

    `model` should always be set at this point, since this function runs after autofilling the other fields.
    """
    new_index_settings = copy.deepcopy(index_settings)
    if IndexSettingsField.search_model not in new_index_settings[IndexSettingsField.index_defaults]:
        # set search_model to model
        try:
            new_index_settings[IndexSettingsField.index_defaults][IndexSettingsField.search_model] = new_index_settings[IndexSettingsField.index_defaults][IndexSettingsField.model]
        except KeyError:
            raise errors.InternalError("Index settings is missing `model` key. Nothing to set `search_model` to.")
        
        # set search_model_properties to model_properties (if they exist)
        if IndexSettingsField.model_properties in new_index_settings[IndexSettingsField.index_defaults]:
            new_index_settings[IndexSettingsField.index_defaults][IndexSettingsField.search_model_properties] = new_index_settings[IndexSettingsField.index_defaults][IndexSettingsField.model_properties]

    return new_index_settings
