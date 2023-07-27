from unittest import mock
import unittest

from marqo.tensor_search import tensor_search
from tests.marqo_test import MarqoTestCase
from marqo.tensor_search.enums import IndexSettingsField as NsField
from marqo.tensor_search import health
from marqo.errors import IndexNotFoundError, InternalError

class TestHealthCheck(MarqoTestCase):
    def setUp(self) -> None:
        self.index_name = "health-check-index"

    def tearDown(self) -> None:
        try:
            tensor_search.delete_index(index_name=self.index_name, config=self.config)
        except IndexNotFoundError as e:
            pass

    def test_health_check(self):
        health_check_status = tensor_search.check_health(self.config)
        assert 'backend' in health_check_status
        assert 'status' in health_check_status['backend']
        assert 'status' in health_check_status

    def test_health_check_red_backend(self):
        mock__get = mock.MagicMock()
        statuses_to_check = ['red', 'yellow', 'green']

        for status in statuses_to_check:
            mock__get.return_value = {
                'status': status
            }
            @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
            def run():
                health_check_status = tensor_search.check_health(self.config)
                assert health_check_status['status'] == status
                assert health_check_status['backend']['status'] == status
                return True
            assert run()

    def test_health_check_unknown_backend_response(self):
        mock__get = mock.MagicMock()
        mock__get.return_value = dict()
        @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
        def run():
            health_check_status = tensor_search.check_health(self.config)
            assert health_check_status['status'] == 'red'
            assert health_check_status['backend']['status'] == 'red'
            return True
        assert run()
    
    def test_convert_watermark_to_bytes(self):
        test_cases = [
            # byte watermarks (total_in_bytes is ignored)
            ("0b", 99999, 0),
            ("21b", 99999, 21),
            ("21B", 99999, 21),
            ("2.1B", 99999, 2.1),
            ("2.1 b", 99999, 2.1),
            ("2.1garbage1b", 99999, InternalError),
            # kb/gb/mb/tb watermarks (total_in_bytes is ignored)
            ("0kb", 99999, 0),
            ("21kb", 99999, 21 * 1024),
            ("2.1MB", 99999, 2.1 * 1024 ** 2),
            ("21GB", 99999, 21 * 1024 ** 3),
            ("2.1 TB", 99999, 2.1 * 1024 ** 4),
            ("2.1garbagePB", 99999, InternalError),
            ("2.1XB", 99999, InternalError),

            # percentage watermarks
            ("0%", 1000, 1000),
            ("80%", 1000, 200),
            ("95%", 10000000, 500000),    # 95% of 10MB Volume
            ("100%", 1000, 0),
            ("40.%", 1000, 600),
            ("0.5%", 1000, 995),
            ("1garbage2%", 1000, InternalError),
            ("-1%", 1000, InternalError),
            ("101%", 1000, InternalError),

            # ratio watermarks
            ("0", 1000, 1000),
            (".80", 1000, 200),
            ("1.00", 1000, 0),
            ("0.4", 1000, 600),
            (".005", 1000, 995),
            ("0.1garbage2", 1000, InternalError),
            ("-.01", 1000, InternalError),
            ("1.01", 1000, InternalError),

            # edge cases
            ("", 99999, InternalError),
            (" ", 99999, InternalError),
            (None, 99999, InternalError)
        ]

        for watermark, total_in_bytes, expected in test_cases:
            try:
                result = health.convert_watermark_to_bytes(watermark, total_in_bytes)
                self.assertAlmostEqual(result, expected)
            except Exception as e:
                assert isinstance(e, expected)
    

    def test_check_opensearch_disk_watermark_breach(self):
        # Note that percentages and ratios are USED SPACE CEILINGS (90%, 0.85, etc)
        # while bytes, kb, mb, gb, tb are MINIMUM AVAILABLE SPACE FLOORS (1gb [out of 10gb], 15kb [out of 100kb], etc])
        test_cases = [
            # Watermark in transient settings only
            {   
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "1kb"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 1025}
                    }
                },
                "EXPECTED": "green"
            },
            # Watermark in persistent settings only
            {   
                "SETTINGS_OBJECT": {
                    "transient": {},
                    "persistent": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {
                            "flood_stage": "1kb"
                        }}}}}
                    },
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 1025}
                    }
                },
                "EXPECTED": "green"
            },
            # Watermark in defaults settings only
            {   
                "SETTINGS_OBJECT": {
                    "transient": {},
                    "persistent": {},
                    "defaults": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {
                            "flood_stage": "1kb"
                        }}}}}
                    }
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 1025}
                    }
                },
                "EXPECTED": "green"
            },
            # Watermark in transient and persistent settings
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "1kb"}}}}}
                    },
                    "persistent": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "2kb"}}}}}
                    },
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 1025}
                    }
                },
                "EXPECTED": "green"
            },

            # Watermark in transient and defaults settings
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "1kb"}}}}}
                    },
                    "persistent": {},
                    "defaults": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "3kb"}}}}}
                    }
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 1025}
                    }
                },
                "EXPECTED": "green"
            },

            # Watermark in persistent and defaults settings
            {
                "SETTINGS_OBJECT": {
                    "transient": {},
                    "persistent": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "2kb"}}}}}
                    },
                    "defaults": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "3kb"}}}}}
                    }
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 2049}
                    }
                },
                "EXPECTED": "green"
            },

            # Watermark in all 3 settings
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "1kb"}}}}}
                    },
                    "persistent": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "2kb"}}}}}
                    },
                    "defaults": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "3kb"}}}}}
                    }
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 1025}
                    }
                },
                "EXPECTED": "green"
            },
            
            # Watermark in none of the settings
            {
                "SETTINGS_OBJECT": {
                    "transient": {},
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 2000}
                    }
                },
                "EXPECTED": InternalError
            },

            # Watermark in B format, not breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "1b"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 2}
                    }
                },
                "EXPECTED": "green"
            },

            # Watermark in B format, EXACTLY ON THE LIMIT
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "2b"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 2}
                    }
                },
                "EXPECTED": "red"   # exactly on the limit is means BREACHED
            },

            # Watermark in B format, breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "3b"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 2}
                    }
                },
                "EXPECTED": "red"
            },

            # Watermark in GB format
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "1gb"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 1024**3 * 5, "available_in_bytes": 20}
                    }
                },
                "EXPECTED": "red"
            },

            # Watermark in percent format
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "90%"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 501}
                    }
                },
                "EXPECTED": "green"
            },

            # Watermark in percent format, EXACTLY ON THE LIMIT
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "90%"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 500}
                    }
                },
                "EXPECTED": "red"
            },

            # Watermark in percent format, breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "90%"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 499}
                    }
                },
                "EXPECTED": "red"
            },

            # Watermark in ratio format
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "0.9"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 501}
                    }
                },
                "EXPECTED": "green"
            },
            # Watermark in ratio format, EXACTLY ON THE LIMIT
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "0.9"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 500}
                    }
                },
                "EXPECTED": "red"
            },
            # Watermark in ratio format, breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "0.9"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": 5000, "available_in_bytes": 499}
                    }
                },
                "EXPECTED": "red"
            },
            # Realistic values: Percentage, not breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "95%"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {
                            "total_in_bytes": 100094816512,     # ~100GB Storage
                            "available_in_bytes": 5123456789    # ~5GB Available
                        },
                    }
                },
                "EXPECTED": "green"
            },
            # Realistic values: Percentage, breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "95%"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {
                            "total_in_bytes": 100094816512,     # ~100GB Storage
                            "available_in_bytes": 4123456789    # ~4GB Available
                        },
                    }
                },
                "EXPECTED": "red"
            },
            # Realistic values: GB, not breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "0.48404636383056643gb"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {
                            "total_in_bytes": 10394816512,
                            "available_in_bytes": 492272384
                        },
                    }
                },
                "EXPECTED": "red"
            },
            # Realistic values: GB, breached
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "0.48404636383056643gb"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {
                            "total_in_bytes": 10394816512,
                            "available_in_bytes": 352272384
                        },
                    }
                },
                "EXPECTED": "red"
            },
            # Negative total_in_bytes
            {
                "SETTINGS_OBJECT": {
                    "transient": {
                        "cluster": {"routing": {"allocation": {"disk": {"watermark": {"flood_stage": "1kb"}}}}}
                    },
                    "persistent": {},
                    "defaults": {}
                },
                "STATS_OBJECT": {
                    "nodes": {
                        "fs": {"total_in_bytes": -1000, "available_in_bytes": 1001}
                    }
                },
                "EXPECTED": InternalError
            }
        ]

        for test_case in test_cases:
            def mock_http_get(path):
                if path == "_cluster/settings?include_defaults=true&filter_path=**.disk*":
                    return test_case["SETTINGS_OBJECT"]
                elif path == "_cluster/stats":
                    return test_case["STATS_OBJECT"]
                else:
                    raise Exception(f"Unexpected path: {path}")
            
            with mock.patch("marqo._httprequests.HttpRequests.get", side_effect=mock_http_get):
                try:
                    assert health.check_opensearch_disk_watermark_breach(self.config) == test_case["EXPECTED"]
                except Exception as e:
                    if isinstance(e, AssertionError):
                        raise e
                    assert isinstance(e, test_case["EXPECTED"])

    def test_index_health_check(self):
        tensor_search.create_vector_index(index_name=self.index_name, config=self.config)
        health_check_status = tensor_search.check_index_health(index_name=self.index_name, config=self.config)
        assert 'backend' in health_check_status
        assert 'status' in health_check_status['backend']
        assert 'status' in health_check_status

    def test_index_health_check_red_backend(self):
        tensor_search.create_vector_index(index_name=self.index_name, config=self.config)
        mock__get = mock.MagicMock()
        statuses_to_check = ['red', 'yellow', 'green']

        for status in statuses_to_check:
            mock__get.return_value = {
                'status': status
            }
            @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
            def run():
                health_check_status = tensor_search.check_index_health(index_name=self.index_name, config=self.config)
                assert health_check_status['status'] == status
                assert health_check_status['backend']['status'] == status
                return True
            assert run()

    def test_index_health_check_path(self):
        tensor_search.create_vector_index(index_name=self.index_name, config=self.config)
        with mock.patch("marqo._httprequests.HttpRequests.get") as mock_get:
            tensor_search.check_index_health(index_name=self.index_name, config=self.config)
            args, kwargs = mock_get.call_args
            self.assertIn(f"_cluster/health/{self.index_name}", kwargs['path'])

    def test_index_health_check_unknown_backend_response(self):
        mock__get = mock.MagicMock()
        mock__get.return_value = dict()

        # Ensure the index does not exist
        with self.assertRaises(IndexNotFoundError):
            tensor_search.delete_index(index_name=self.index_name, config=self.config)
        @mock.patch("marqo._httprequests.HttpRequests.get", mock__get)
        def run():
            health_check_status = tensor_search.check_index_health(index_name=self.index_name, config=self.config)
            assert health_check_status['status'] == 'red'
            assert health_check_status['backend']['status'] == 'red'
            return True
        assert run()


class TestAggregateStatus(unittest.TestCase):
    def test_status_green(self):
        result = health.aggregate_status(marqo_status="green", marqo_os_status="green")
        self.assertEqual(result, ('green', 'green'))

    def test_status_yellow(self):
        result = health.aggregate_status(marqo_status="green", marqo_os_status="yellow")
        self.assertEqual(result, ('yellow', 'yellow'))

    def test_status_red(self):
        result = health.aggregate_status(marqo_status="green", marqo_os_status="red")
        self.assertEqual(result, ('red', 'red'))

