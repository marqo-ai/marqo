import unittest

import httpx

from marqo import documentation


class TestDocumentation(unittest.TestCase):
    def test_urls(self):
        # Retrieve all public functions in the module
        public_functions = [func for func in dir(documentation)
                            if callable(getattr(documentation, func)) and not func.startswith("_")]

        # Verify all URLs return a 200 response
        for func in public_functions:
            with self.subTest(func=func):
                url = getattr(documentation, func)()
                response = httpx.get(url, follow_redirects=True)
                response.raise_for_status()
                self.assertFalse('404.html' in response.content.decode(), f"{func} URL returned a 404 response")
