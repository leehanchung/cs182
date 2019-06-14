#!/bin/bash
# This is a script designed to download the datasets you will need from the internet
echo "Downloading data... Please wait, this might take a while..."
rm cs182_homework.zip 2&> /dev/null
rm -r ./dataset 2&> /dev/null

wget https://bcourses.berkeley.edu/files/74751488/download?download_frd=1  -O cs182_homework.zip

echo "Finished downloading. Unzipping and Verifying Data Integrity..."
if [ $(md5sum cs182_homework.zip | cut -d' ' -f 1)='eabf180bc6edf3655d9fce9086b89a64' ]; then
    # The MD5 sum matched
    echo "Checksum ok! Unzipping."
    unzip cs182_homework.zip
    rm cs182_homework.zip
else
    echo "Error: md5 hash for \"cs182_homework.zip\" wasn't as expected. Verify your data downloaded correctly or re-run this script."
fi

echo "Done downloading!"
