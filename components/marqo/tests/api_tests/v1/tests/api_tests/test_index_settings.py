import uuid

from tests.marqo_test import MarqoTestCase


class TestIndexSettings(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        cls.marqtuned_index_name = "marqtuned_" + str(uuid.uuid4()).replace('-', '')
        cls.non_marqtuned_index_name = "non_marqtuned_" + str(uuid.uuid4()).replace('-', '')

        cls.create_indexes([
            {
                "indexName": cls.marqtuned_index_name,
                "model": "marqtune/model-id/checkpoint",
                "modelProperties": {
                    "isMarqtuneModel": True,
                    "name": "ViT-B-32",
                    "dimensions": 512,
                    "model_location": {
                        "s3": {
                            "Bucket": "marqtune-public-bucket",
                            "Key": "marqo-test-open-clip-model/epoch_2.pt",
                        },
                        "auth_required": False
                    },
                    "type": "open_clip",
                }
            },
            {
                "indexName": cls.non_marqtuned_index_name,
                "model": "non-marqtune/model-id/checkpoint",
                "modelProperties": {
                    "isMarqtuneModel": False,
                    "name": "ViT-B-32",
                    "dimensions": 512,
                    "model_location": {
                        "s3": {
                            "Bucket": "marqtune-public-bucket",
                            "Key": "marqo-test-open-clip-model/epoch_2.pt",
                        },
                        "auth_required": False
                    },
                    "type": "open_clip",
                }
            }
        ])

        cls.indexes_to_delete = [cls.marqtuned_index_name, cls.non_marqtuned_index_name]

    def test_marqtune_index_settings(self):
        with self.subTest(msg="Hide all model properties except isMarqtuneModel"):
            index_settings = self.client.index(self.marqtuned_index_name).get_settings()
            self.assertEqual("marqtune/model-id/checkpoint", index_settings['model'])
            self.assertEqual({
                "isMarqtuneModel": True,
            }, index_settings['modelProperties'])

        with self.subTest(msg="Don't hide any model properties"):
            index_settings = self.client.index(self.non_marqtuned_index_name).get_settings()
            self.assertEqual("non-marqtune/model-id/checkpoint", index_settings['model'])
            self.assertEqual({
                "isMarqtuneModel": False,
                "name": "ViT-B-32",
                "dimensions": 512,
                "model_location": {
                    "s3": {
                        "Bucket": "marqtune-public-bucket",
                        "Key": "marqo-test-open-clip-model/epoch_2.pt",
                    },
                    "auth_required": False
                },
                "type": "open_clip",
            }, index_settings['modelProperties'])
