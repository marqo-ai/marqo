#!/bin/bash
# Copyright Yahoo. Licensed under the terms of the Apache 2.0 license. See LICENSE in the project root.

set -e

if [ $# -gt 1 ]; then
    echo "Allowed arguments to entrypoint are {configserver,services}."
    exit 1
fi

# Set the hostname to the FQDN name if possible
oldhn=$(hostname)
fullhn=$(hostname -f)
if [ "${oldhn}" != "${fullhn}" ]; then
    myuid=$(id -u)
    if [ "${myuid}" = 0 ]; then
        echo "WARNING - trying to change hostname '${oldhn}' -> '${fullhn}'"
        hostname "$fullhn" || true
    else
        echo "WARNING - want to change hostname '${oldhn}' -> '${fullhn}' (to avoid this warning, change your setup)"
    fi
fi

trap cleanup TERM INT

cleanup() {
    /opt/vespa/bin/vespa-stop-services
    exit $?
}

if [ -n "$1" ]; then
    if [ -z "$VESPA_CONFIGSERVERS" ]; then
        echo "VESPA_CONFIGSERVERS must be set with '-e VESPA_CONFIGSERVERS=<comma separated list of config servers>' argument to docker."
        exit 1
    fi
    case $1 in
        configserver)
            cleanup() {
                /opt/vespa/bin/vespa-stop-configserver
                exit $?
            }
            /opt/vespa/bin/vespa-start-configserver
            ;;
        services)
            /opt/vespa/bin/vespa-start-services
            ;;
        services,configserver | configserver,services)
            cleanup() {
                /opt/vespa/bin/vespa-stop-configserver
                /opt/vespa/bin/vespa-stop-services
                exit $?
            }
            /opt/vespa/bin/vespa-start-configserver
            /opt/vespa/bin/vespa-start-services
            ;;
        *)
            echo 'Allowed arguments to entrypoint are "configserver", "services" or "configserver,services".'
            exit 1
            ;;
    esac
else
    if [ -z "$VESPA_CONFIGSERVERS" ]; then
        export VESPA_CONFIGSERVERS=$(hostname)
    fi
    /opt/vespa/bin/vespa-start-configserver
    /opt/vespa/bin/vespa-start-services
fi

if [ "$VESPA_LOG_STDOUT" = "true" ]; then
	FORMAT="${VESPA_LOG_FORMAT:-vespa}"
	/opt/vespa/bin/vespa-logfmt --follow --format "$FORMAT" ${VESPA_LOGFMT_ARGUMENTS} &
	wait
else
	sleep infinity &
	wait
fi