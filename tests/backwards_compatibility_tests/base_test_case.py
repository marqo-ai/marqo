import abc
import os
import sys
from marqo_test import MarqoTestCase

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

class BaseTestCase(MarqoTestCase):

    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

    def prepare(self):
        pass

