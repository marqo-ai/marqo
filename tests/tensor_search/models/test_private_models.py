import unittest
from marqo.tensor_search.models.private_models import ModelAuth, ModelLocation
from marqo.api.exceptions import InvalidArgError
from marqo.tensor_search.models.external_apis.hf import HfAuth, HfModelLocation
from marqo.tensor_search.models.external_apis.s3 import S3Auth, S3Location

class TestModelAuth(unittest.TestCase):
    def test_no_auth(self):
        with self.assertRaises(InvalidArgError):
            ModelAuth()

    def test_multiple_auth(self):
        with self.assertRaises(InvalidArgError):
            ModelAuth(
                s3=S3Auth(aws_secret_access_key="test", aws_access_key_id="test"),
                hf=HfAuth(token="test"))

    def test_s3_auth(self):
        try:
            ModelAuth(s3=S3Auth(aws_secret_access_key="test", aws_access_key_id="test"))
        except InvalidArgError:
            self.fail("ModelAuth raised InvalidArgError unexpectedly!")

    def test_hf_auth(self):
        try:
            ModelAuth(hf=HfAuth(token="test"))
        except InvalidArgError:
            self.fail("ModelAuth raised InvalidArgError unexpectedly!")

class TestModelLocation(unittest.TestCase):
    def test_no_location(self):
        with self.assertRaises(InvalidArgError):
            ModelLocation()

    def test_multiple_locations(self):
        with self.assertRaises(InvalidArgError):
            ModelLocation(
                s3=S3Location(Bucket="test", Key="test"),
                hf=HfModelLocation(repo_id="test", filename="test"))

    def test_s3_location(self):
        try:
            ModelLocation(s3=S3Location(Bucket="test", Key="test"))
        except InvalidArgError:
            self.fail("ModelLocation raised InvalidArgError unexpectedly!")

    def test_hf_location(self):
        try:
            ModelLocation(hf=HfModelLocation(repo_id="test", filename="test"))
        except InvalidArgError:
            self.fail("ModelLocation raised InvalidArgError unexpectedly!")
