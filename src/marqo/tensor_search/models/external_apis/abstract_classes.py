"""
These are abstract classes that shouldn't be instantiated
"""


class ExternalAuth:
    """Authentication used to download an object
    """
    pass


class ObjectLocation:
    """Reference to an object location (for example a pointer to a model file
    in s3
    """
    pass

