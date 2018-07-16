#!/bin/bash

echo 'Check problem sets directory:'

(test -e ${BENCHMARKS_PATH}/kaluza.zip && test $(stat -c%s ${BENCHMARKS_PATH}/kaluza.zip) -gt 35000 \
 && echo 'kaluza found' ) || \
(echo 'kaluza.zip not found, downloading...' \
 && python3 /download.py "https://drive.google.com/uc?id=15ZgdMB4r1sqHtne4CfNEQvwEc0ujB1dc&export=download" /kaluza.zip 35 \
 && echo "Unpack and move into: ${BENCHMARKS_PATH}/kaluza" \
 && unzip -d /tmp /kaluza.zip && mv /tmp/Kaluza ${BENCHMARKS_PATH}/kaluza && mv /kaluza.zip ${BENCHMARKS_PATH})

(test -e ${BENCHMARKS_PATH}/pisa.zip && test $(stat -c%s ${BENCHMARKS_PATH}/pisa.zip) -gt 5000 \
 && echo 'pisa found') || \
(echo 'pisa.zip not found, downloading...' \
 && python3 /download.py "https://drive.google.com/uc?id=1LJar0OXSRBD03mRBZEfnQRs3-QsuRSJp&export=download" /pisa.zip 5 \
 && echo "Unpack and move into: ${BENCHMARKS_PATH}/piza" \
 && unzip -d /tmp /pisa.zip && mv /tmp/pisa ${BENCHMARKS_PATH}/pisa && mv /pisa.zip ${BENCHMARKS_PATH})

(test -e ${BENCHMARKS_PATH}/appscan.zip && test $(stat -c%s ${BENCHMARKS_PATH}/appscan.zip) -gt 4000 \
 && echo 'appscan found') || \
(echo 'appscan.zip not found, downloading...' \
 && python3 /download.py "https://drive.google.com/uc?id=1nbjEj3BzywfbMxkyub4Sn0iYKiegmhJx&export=download" /appscan.zip 4 \
 && echo "Unpack and move into: ${BENCHMARKS_PATH}/appscan" \
 && unzip -d /tmp /appscan.zip && mv /tmp/appscan ${BENCHMARKS_PATH}/appscan && mv /appscan.zip ${BENCHMARKS_PATH})

(test -e ${BENCHMARKS_PATH}/pyex.zip && test $(stat -c%s ${BENCHMARKS_PATH}/pyex.zip) -gt 4000 \
 && echo 'pyex found') || \
(echo 'pyex.zip not found, downloading...' \
 && python3 /download.py "https://drive.google.com/uc?id=1V_g8SPNuT0GWoXVUqkyW3ez7BstYXYN2&export=download" /pyex.zip 31 \
 && echo "Unpack and move into: ${BENCHMARKS_PATH}/pyex" \
 && unzip -d /tmp /pyex.zip && mv /tmp/PyEx ${BENCHMARKS_PATH}/pyex && mv /pyex.zip ${BENCHMARKS_PATH})

exec "$@"
