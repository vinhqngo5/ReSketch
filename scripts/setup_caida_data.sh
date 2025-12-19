#!/bin/bash

set -e

# Dataset file to download
CAIDA_DOWNLOAD_BASEURL="https://data.caida.org/datasets/passive-2018/equinix-nyc/20180315-130000.UTC/"
CAIDA_DOWNLOAD_FILE="equinix-nyc.dirA.20180315-125910.UTC.anon.pcap.gz"

# Destination directory
DEST_DIR="data/CAIDA"
DEST_FILE="only_ip"

help() {
  echo "Download and preprocess CAIDA dataset file for experiments."
  echo
  echo "Downloading CAIDA data requires a username and password. Apply for access at:"
  echo -e "\thttps://www.caida.org/catalog/datasets/request_user_info_forms/passive_dataset_request/"
  echo
  echo "Syntax: $0 <username>"
}

# If no arguments are passed, show help
if [ $# -eq 0 ]; then
  help
  exit 1
fi

caida_user="$1"

# Download the file
echo "Downloading..."
echo "${CAIDA_DOWNLOAD_BASEURL}/${CAIDA_DOWNLOAD_FILE}"
curl --user "${caida_user}" --progress-bar \
  --create-dirs --output-dir "${DEST_DIR}" --output "${CAIDA_DOWNLOAD_FILE}" \
  --continue-at - \
  "${CAIDA_DOWNLOAD_BASEURL}/${CAIDA_DOWNLOAD_FILE}"

# Decompress
echo "Decompressing..."
gzip -dk "${DEST_DIR}/${CAIDA_DOWNLOAD_FILE}"

# Extract source IPs to text file
echo "Extracting source IPs from ${CAIDA_DOWNLOAD_FILE%.*}"
tshark -r "${DEST_DIR}/${CAIDA_DOWNLOAD_FILE%.*}" -T fields -e ip.src > "${DEST_DIR}/${DEST_FILE}"

echo "Finished"
