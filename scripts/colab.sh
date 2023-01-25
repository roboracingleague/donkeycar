#!/bin/sh

set -e

PROJECT_ID="donkeycar-rma"
BUCKET="gs://donkeycar-rma"
ARCHIVE="donkey.tar.gz"
MODELS_ARCHIVE="models.tar"

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
ROOT_PATH=$( readlink -f "${SCRIPT_DIR}/.." )
TAR_PATH="${ROOT_PATH}/../${ARCHIVE}"
MODELS_PATH="${ROOT_PATH}/models"
MODELS_BACKUP_PATH="${ROOT_PATH}/models_backup"

login(){
  gcloud auth login
  gcloud config set project "${PROJECT_ID}"
}

upload(){
  echo 'Creating archive'
  tar zcf "${TAR_PATH}" --exclude="${MODELS_PATH}/*" . -C "${ROOT_PATH}"
  echo 'Uploading archive'
  gsutil cp "${TAR_PATH}" "${BUCKET}"
  #rm -rf "${TAR_PATH}"
  echo 'DON T FORGET TO'
  echo '- REMOVE LOCAL ARCHIVE'
  echo '- UPLOAD myconfig.py'
}

download(){
  #echo 'Copying models to models_backup'
  # cp -r "${MODELS_PATH}/" "${MODELS_BACKUP_PATH}"
  #echo 'Removing model folder'
  # rm -rf "${MODELS_PATH}"
  #echo 'Downloading model'
  # gsutil cp "${BUCKET}/${MODELS_ARCHIVE}" "${ROOT_PATH}"
  echo 'Uncompress model archive'
  tar zxf "${ROOT_PATH}/${MODELS_ARCHIVE}"
  #echo 'Removing remote archive file'
  #gsutil rm "${BUCKET}/${MODELS_ARCHIVE}"
  #echo 'Removing local archive file'
  #rm -f "${ROOT_PATH}/${MODELS_ARCHIVE}"
}

usage(){
  echo "-- Upload training files to GCS --"
  echo "Usage: togs.sh <command>"
  echo "Available commands :"
  echo " login    : GCP login and set project"
  echo " upload   : upload the current directory"
  echo " download : download the models directory"
  echo "          : default to uploading"
}

case "$1" in
  login) login ;;
  upload) upload ;;
  download) download ;;
  -h) usage ;;
  *) echo "'$1' is not a valid command"; usage ;;
esac
