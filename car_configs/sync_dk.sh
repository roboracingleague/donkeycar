#! /bin/bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
source "${SCRIPT_DIR}/env.sh"

echo SENDING DONKEYCAR FILES
rsync -rv --progress --partial --delete \
  --exclude=.DS_Store --exclude=.git --exclude=doc --exclude=notebooks --exclude=car_configs \
  "${LOCAL_DK_PATH}/" "$USER"@"$CAR_HOSTNAME":"${REMOTE_DK_PATH}/"

echo -e "\nCOPYING TEMPLATE manage.py"
scp "$USER"@"$CAR_HOSTNAME":"$TEMPLATE_MANAGE_PATH" "$USER"@"$CAR_HOSTNAME":"$REMOTE_MANAGE_PATH"
