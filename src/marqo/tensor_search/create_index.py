"""Functions used to fulfill the add_documents endpoint"""
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
from marqo.tensor_search.enums import IndexSettingsField
from marqo.tensor_search import constants


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