#!/bin/bash

# This script checks the result of a set of test workflow
# A workflow is successful if either the SHOULD_TEST_PROCEED parameter indicates the tests were skipped, or 
# if all Start-Runner, Test-Marqo, Stop-Runner job were successful.
# Args:
#    - SHOULD_TEST_PROCEED: 
#      - If tests weren't meant to proceed this should be set to "false". Anything else is considered "true"
#    - START_RUNNER_JOB_RESULT, TEST_JOB_RESULT, STOP_RUNNER_JOB_RESULT: 
#      - These values for these args should come from the GH workflow. For example, from "${{ needs.Start-Runner.result }}"
#      - These should be set to "success" if the corresponding job succeeded, anything else is considered "failure" 
# Usage:
# ./check_test_success.sh [SHOULD_TEST_PROCEED] [START_RUNNER_JOB_RESULT] [TEST_JOB_RESULT] [STOP_RUNNER_JOB_RESULT]


# Fetch the input parameters
SHOULD_TEST_PROCEED="$1"
START_RUNNER_JOB_RESULT="$2"
TEST_JOB_RESULT="$3"
STOP_RUNNER_JOB_RESULT="$4"

# Logic to check if tests should proceed or if all jobs succeeded
if [ "$SHOULD_TEST_PROCEED" == "false" ]; then
  echo "Tests were skipped."
  exit 0
elif [ "$START_RUNNER_JOB_RESULT" == "success" ] && [ "$TEST_JOB_RESULT" == "success" ] && [ "$STOP_RUNNER_JOB_RESULT" == "success" ]; then
  echo "All jobs passed successfully."
  exit 0
else
  echo "One or more jobs failed."
  exit 1
fi
