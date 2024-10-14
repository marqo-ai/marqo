from fastapi.testclient import TestClient
from marqo.tensor_search.api import app
from tests.marqo_test import MarqoTestCase


class OpenApiTests(MarqoTestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_docs_endpoint(self):
        """Test if the /docs endpoint is accessible and returns status code 200."""
        response = self.client.get("/docs")
        self.assertEqual(response.status_code, 200, "The /docs endpoint should be accessible.")

    def test_openapi_json_endpoint(self):
        """Test if the /openapi.json endpoint is accessible and returns status code 200."""
        response = self.client.get("/openapi.json")
        self.assertEqual(response.status_code, 200, "The /openapi.json endpoint should be accessible.")