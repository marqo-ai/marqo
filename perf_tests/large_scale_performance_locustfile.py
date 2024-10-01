from __future__ import annotations

import random
import os
import csv
import time
import json
from datetime import datetime

from locust import events, task, between, FastHttpUser, LoadTestShape
from locust.env import Environment
import marqo
import marqo.errors
import numpy as np
import pandas as pd
import sys
import argparse
import os
import re


# Load queries from the CSV file
def load_queries(csv_path):
    queries = []
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        queries.append({
            'search_keywords': row['search_keywords'],
            'search_category': row['search_category']
        })
    return queries

# Load necessary data from extracted CSV files
def load_data(data_dir):
    data = {}

    # Load available product codes
    data['available_product_codes'] = set()
    available_product_codes_csv = os.path.join(data_dir, 'extracted_available_product_codes.csv')
    df = pd.read_csv(available_product_codes_csv)
    data['available_product_codes'] = set(df['available_ia_code'].dropna().unique())

    # Load ia_codes
    data['ia_codes'] = set()
    ia_codes_csv = os.path.join(data_dir, 'extracted_ia_codes.csv')
    df = pd.read_csv(ia_codes_csv)
    data['ia_codes'] = set(df['ia_code'].dropna().unique())

    # Load ro_queries
    data['ro_queries'] = set()
    ro_queries_csv = os.path.join(data_dir, 'extracted_ro_queries.csv')
    df = pd.read_csv(ro_queries_csv)
    data['ro_queries'] = set(df['query'].dropna().unique())

    # Load truncated_tags
    data['truncated_tags'] = set()
    truncated_tags_csv = os.path.join(data_dir, 'extracted_truncated_tags.csv')
    df = pd.read_csv(truncated_tags_csv)
    data['truncated_tags'] = set(df['truncated_tags'].dropna().unique())

    return data

# Paths
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
QUERIES_CSV = os.path.join(os.path.dirname(__file__), 'bquxjob_5a025798_19241877e37.csv')

# Load data
DATA = load_data(DATA_DIR)

# Load queries
QUERIES = load_queries(QUERIES_CSV)
NUM_QUERIES = len(QUERIES)

# Regions and their probabilities
REGIONS = ['US', 'AU', 'CA', 'GB', 'DE', 'FR', 'JP']
REGION_PROBABILITIES = [0.7, 0.15, 0.05, 0.05, 0.02, 0.02, 0.01]

# Index name from environment variable
INDEX_NAME = os.getenv('MARQO_INDEX_NAME', 'locust-test')

class MarqoUser(FastHttpUser):
    wait_time = between(1, 5)  # Simulate user think time
    client = None
    telemetry_data = []

    def on_start(self):
        host = self.environment.host
        api_key = os.getenv('MARQO_CLOUD_API_KEY', None)
        if api_key:
            self.client = marqo.Client(url=host, api_key=api_key, return_telemetry=True)
        else:
            self.client = marqo.Client(url=host, return_telemetry=True)

    @task
    def perform_search(self):
        max_retries = 3
        retry_delay = 1  # Start with 1 second
        attempt = 0
        while attempt < max_retries:
            attempt += 1
            # Randomly select a query
            query_info = random.choice(QUERIES)
            search_keywords = query_info['search_keywords']
            search_category = query_info['search_category']

            # Randomly select a region based on probabilities
            region = random.choices(REGIONS, weights=REGION_PROBABILITIES, k=1)[0]

            # Determine if the query is 'Topic Only' or 'Topic+product'
            if search_category.strip().lower() == 'all-departments':
                # Topic Only
                q = search_keywords
                # Randomly select an ia_code for the modifiers
                ia_code = random.choice(list(DATA['ia_codes']))
                score_modifiers = {
                    "add_to_score": [
                        {"field_name": f"artist_sales_scores.{ia_code}", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"recent_sales_scores.{ia_code}+{region}", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"recent_sales_scores.{ia_code}+ALL_WORLD", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"all_time_sales_scores.{ia_code}+ALL_WORLD", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"ro_scores.{search_keywords}+{ia_code}", "weight": random.uniform(0.0001, 0.005)},
                    ]
                }
                # No filters
                filter_string = None
            else:
                # Topic+product
                q = f"{search_category}: {search_keywords}"
                # Use the search_category as the ia_code if available, else random
                ia_code = search_category if search_category in DATA['available_product_codes'] else random.choice(list(DATA['ia_codes']))
                score_modifiers = {
                    "add_to_score": [
                        {"field_name": f"artist_sales_scores.{ia_code}", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"recent_sales_scores.{ia_code}+{region}", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"recent_sales_scores.{ia_code}+ALL_WORLD", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"all_time_sales_scores.{ia_code}+ALL_WORLD", "weight": random.uniform(0.0001, 0.005)},
                        {"field_name": f"ro_scores.{search_keywords}+{ia_code}", "weight": random.uniform(0.0001, 0.005)},
                    ]
                }
                # Add filter on available_product_codes
                filter_string = f"available_product_codes:{ia_code}"

            # Construct the search parameters
            search_params = {
                'q': q,
                'limit': 20,
                'search_method': 'HYBRID', 
                'hybrid_parameters': {
                    "retrievalMethod": "disjunction",
                    "rankingMethod": "rrf",
                    "alpha": 0.3,
                    "rrfK": 60,
                    "searchableAttributesLexical": ["tags"],
                    "scoreModifiersTensor": score_modifiers,
                    "scoreModifiersLexical": {
                        "add_to_score": [
                            {"field_name": f"artist_sales_scores.{ia_code}", "weight": random.uniform(0.1, 5)},
                            {"field_name": f"recent_sales_scores.{ia_code}+{region}", "weight": random.uniform(0.1, 5)},
                            {"field_name": f"recent_sales_scores.{ia_code}+ALL_WORLD", "weight": random.uniform(0.1, 5)},
                            {"field_name": f"all_time_sales_scores.{ia_code}+ALL_WORLD", "weight": random.uniform(0.1, 5)},
                            {"field_name": f"ro_scores.{search_keywords}+{ia_code}", "weight": random.uniform(1, 300)},
                        ]
                    },
                },
                'attributes_to_retrieve': ['work_id', 'tags', 'available_product_codes'],
            }
            if filter_string:
                search_params['filter_string'] = filter_string

            # Perform the search and capture telemetry data
            start_time = time.time()
            try:
                response = self.client.index(INDEX_NAME).search(**search_params)
                total_time = (time.time() - start_time) * 1000  # in ms
                # Extract telemetry data
                telemetry = response.get('telemetry', {})
                telemetry['total_time_ms'] = total_time
                self.telemetry_data.append(telemetry)
                # Record success
                self.environment.events.request.fire(
                    request_type='SEARCH',
                    name='perform_search',
                    response_time=total_time,
                    response_length=len(json.dumps(response)),
                    exception=None,
                    context={},
                )
                break  # Break out of the retry loop if successful
            except marqo.errors.MarqoWebError as e:
                if e.code == 'too_many_requests':
                    # Implement exponential backoff
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    total_time = (time.time() - start_time) * 1000  # in ms
                    # Record failure
                    self.environment.events.request.fire(
                        request_type='SEARCH',
                        name='perform_search',
                        response_time=total_time,
                        response_length=0,
                        exception=e,
                        context={},
                    )
                    break
            except Exception as e:
                total_time = (time.time() - start_time) * 1000  # in ms
                # Record failure
                self.environment.events.request.fire(
                    request_type='SEARCH',
                    name='perform_search',
                    response_time=total_time,
                    response_length=0,
                    exception=e,
                    context={},
                )
                break

    def on_stop(self):
        # Save telemetry data to a file
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        telemetry_filename = f"telemetry_data_{timestamp}.json"
        summary_filename = f"telemetry_summary_{timestamp}.json"
        with open(telemetry_filename, 'w') as f:
            json.dump(self.telemetry_data, f)

        # Process telemetry data and output summary
        if self.telemetry_data:
            total_times = [t.get('total_time_ms', 0) for t in self.telemetry_data]
            # Calculate average, median, percentiles, etc.
            avg_time = np.mean(total_times)
            median_time = np.median(total_times)
            p95_time = np.percentile(total_times, 95)
            # Calculate error rate
            total_requests = len(total_times)
            successful_requests = sum(1 for t in self.telemetry_data if 'total_time_ms' in t)
            error_rate = ((total_requests - successful_requests) / total_requests) * 100 if total_requests > 0 else 0.0
            # Save summary
            summary = {
                'avg_response_time_ms': avg_time,
                'median_response_time_ms': median_time,
                '95th_percentile_response_time_ms': p95_time,
                'total_requests': total_requests,
                'successful_requests': successful_requests,
                'error_rate_percent': error_rate,
            }
            with open(summary_filename, 'w') as f:
                json.dump(summary, f)

# Event listener to ensure telemetry data is saved even if the test stops prematurely
@events.quitting.add_listener
def save_telemetry_on_quit(environment, **kw):
    for user in environment.runner.user_classes:
        if hasattr(user, 'telemetry_data') and user.telemetry_data:
            timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
            telemetry_filename = f"telemetry_data_{timestamp}.json"
            summary_filename = f"telemetry_summary_{timestamp}.json"
            with open(telemetry_filename, 'w') as f:
                json.dump(user.telemetry_data, f)
            # Optionally, add summary processing here if needed


class BurstLoadShape(LoadTestShape):
    def __init__(self):
        super().__init__()
        self.stages = [
            {"duration": 120, "users": 10, "spawn_rate": 1},     # First 2 minutes: ramp up to 10 users
            {"duration": 300, "users": 55, "spawn_rate": 2},    # Next 5 minutes: ramp up to 55 users
            {"duration": 300, "users": 100, "spawn_rate": 5},   # Next 5 minutes: ramp up to 100 users
            {"duration": 600, "users": 250, "spawn_rate": 7},   # Next 10 minutes: ramp up to 250 users
            {"duration": 600, "users": 350, "spawn_rate": 10},  # Next 10 minutes: ramp up to 350 users
            {"duration": 600, "users": 250, "spawn_rate": 7},   # Next 10 minutes: ramp down to 250 users
            {"duration": 900, "users": 50, "spawn_rate": 5},    # Next 15 minutes: ramp down to 50 users
        ]
        self.stage_index = 0
        self.stage_start_time = 0

        # Parse the -t flag to get the maximum duration
        parser = argparse.ArgumentParser()
        parser.add_argument("-t", "--run-time", type=str, help="Stop after given time. e.g. 72h")
        args, _ = parser.parse_known_args()
        self.max_duration = self.parse_run_time(args.run_time) if args.run_time else sum(stage["duration"] for stage in self.stages)

    def parse_run_time(self, run_time_str):
        """
        Parse the run-time string from the -t flag and convert it to seconds.
        Supports formats like '72h', '30m', '1h30m', etc.
        """
        pattern = re.compile(r'^(?:(?P<hours>\d+)h)?(?:(?P<minutes>\d+)m)?(?:(?P<seconds>\d+)s)?$')
        match = pattern.fullmatch(run_time_str.strip())
        if not match:
            raise ValueError(f"Invalid run-time format: {run_time_str}")

        time_params = {name: int(value) if value else 0 for name, value in match.groupdict().items()}
        total_seconds = time_params['hours'] * 3600 + time_params['minutes'] * 60 + time_params['seconds']
        if total_seconds == 0:
            raise ValueError("Run-time must be greater than 0 seconds.")
        return total_seconds

    def tick(self):
        run_time = self.get_run_time()
        if run_time >= self.max_duration:
            return None  # Stop the test

        # Initialize stage start time
        if self.stage_start_time == 0:
            self.stage_start_time = run_time

        current_stage = self.stages[self.stage_index]
        stage_elapsed = run_time - self.stage_start_time

        if stage_elapsed >= current_stage["duration"]:
            # Move to the next stage
            self.stage_index = (self.stage_index + 1) % len(self.stages)
            self.stage_start_time = run_time
            current_stage = self.stages[self.stage_index]

        return (current_stage["users"], current_stage["spawn_rate"])

