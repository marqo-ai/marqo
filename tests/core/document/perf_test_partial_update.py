import random
import string

import numpy as np

from marqo.core.models.add_docs_params import AddDocsParams
from marqo.tensor_search.telemetry import RequestMetricsStore
from tests.marqo_test import MarqoTestCase


class Metrics:
    def __init__(self, name):
        self.name = name
        self._metrics = []

    def add_metric(self, telemetry: dict, prefix: str):
        self._metrics.append([
            telemetry['timesMs'][f'{prefix}.total'],
            telemetry['timesMs'][f'{prefix}.vespa._get_batch'],  # batch get all docs
            telemetry['timesMs'][f'{prefix}.vespa._bulk'],  # batch write to vespa
        ])

    def get_metrics(self):
        return {
            "mean": np.mean(self._metrics, axis=0),
            "percentile": np.percentile(self._metrics, [50, 90, 95], axis=0),
        }

    def print_metrics(self, batch_size):
        metrics = self.get_metrics()
        print(f'\nMetrics(timeMs) for {self.name} of batch_size: {batch_size}, total_batch: {len(self._metrics)}')
        print(f'mean: {metrics["mean"]}')
        print(f'p50: {metrics["percentile"][0]}')
        print(f'p90: {metrics["percentile"][1]}')
        print(f'p95: {metrics["percentile"][2]}')


class TestPartialUpdatePerf(MarqoTestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()

        # semi_structured_index_request = cls.unstructured_marqo_index_request(name='test_partial_update_semi_structured')
        # cls.create_indexes([semi_structured_index_request])
        # cls.index = cls.indexes[0]
        cls.index = cls.config.index_management.get_index('test_partial_update_semi_structured')

    # Perf test cases
    # 100 docs

    # Update 100 docs 10 times
    # * update only 1 text field
    # * update only 1 bool field
    # * update only 1 int field
    # * update only 1 int map
    # * update only 1 float field
    # * update only 1 float map
    # * update only 1 string array
    # * update 1/10 of the fields
    # * update 1/2 of the fields
    # * update all available fields

    # Add doc override 100 docs 10 times, 10 per batch, with using existing tensor enabled

    def test_perf(self):
        total_docs = 100
        batch_size = 10
        repeat_times = 10

        initial_docs = [random_doc(str(doc_id)) for doc_id in range(100)]
        self._add_docs(initial_docs)

        def run(test_name, test_function, update_fields_lambda):
            metrics = Metrics(test_name)
            for _ in range(repeat_times):
                for batch_number in range(int(total_docs / batch_size)):
                    docs_to_update = initial_docs[batch_number * batch_size:(batch_number + 1) * batch_size]

                    RequestMetricsStore.clear_metrics_for(self.mock_request)
                    RequestMetricsStore.set_in_request(self.mock_request)

                    res = test_function(docs_to_update, update_fields_lambda)

                    telemetry = RequestMetricsStore.for_request(self.mock_request).json()
                    metrics.add_metric(telemetry, test_name)

            metrics.print_metrics(batch_size)

        test_cases = [
            ('change 1 short string field', lambda: {'ss1': random_string(10)}),
            ('change 1 long string field', lambda: {'ls1': random_string(70)}),
            # TODO add more test cases
        ]

        for (test_case, update_fields_lambda) in test_cases:
            print(f'\n{test_case} Result: ')
            run('add_documents', self._add_document, update_fields_lambda)
            run('partial_update', self._partial_update, update_fields_lambda)

    def _add_docs(self, docs):
        return self.add_documents(self.config, add_docs_params=AddDocsParams(
            index_name=self.index.name,
            docs=docs,
            use_existing_tensors=True,
            tensor_fields=['title', 'desc', 'multimodal_combo_field'],
            mappings={
                "multimodal_combo_field": {
                    "type": "multimodal_combination",
                    "weights": {"title": 1.0, "desc": 2.0}
                }
            }
        ))

    def _add_document(self, docs_to_update, update_fields_lambda):
        for doc in docs_to_update:
            doc.update(update_fields_lambda())

        with RequestMetricsStore.for_request().time("add_documents.total"):
            return self._add_docs(docs_to_update)

    def _partial_update(self, docs_to_update, update_fields_lambda):
        docs_to_partial_update = [{'_id': doc['_id'], **update_fields_lambda()} for doc in docs_to_update]

        with RequestMetricsStore.for_request().time("partial_update.total"):
            return self.config.document.partial_update_documents_by_index_name(self.index.name, docs_to_partial_update)


def random_doc(doc_id: str):
    doc = {'_id': doc_id, 'title': random_string(10), 'desc': random_string(20)}

    for i in range(5):
        doc[f'ss{i}'] = random_string(10)
        doc[f'ls{i}'] = random_string(60)
        doc[f'b{i}'] = bool(random.randint(0, 1))
        doc[f'i{i}'] = random.randint(0, 100)
        doc[f'f{i}'] = random.uniform(0.0, 100.0)
        doc[f'sa{i}'] = [random_string(10) for _ in range(5)]
        doc[f'im{i}'] = {f'k{j}': random.randint(0, 100) for j in range(5)}
        doc[f'fm{i}'] = {f'k{j}': random.uniform(0.0, 100.0) for j in range(5)}

    return doc


def random_string(length=10):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))