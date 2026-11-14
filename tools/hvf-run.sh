#!/bin/sh
# Cargo runner: ad-hoc sign the binary with the hypervisor entitlement, then exec it.
set -e
codesign --sign - --force --entitlements "$(dirname "$0")/hypervisor.entitlements" "$1" 2>/dev/null
exec "$@"
