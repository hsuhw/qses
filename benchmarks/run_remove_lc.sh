#!/bin/bash

dir="$(dirname "${BASH_SOURCE[0]}")"

tgt_dir="${@%/}"
if [[ -z "${tgt_dir}" ]]; then
  echo "No target DIR provided.  Doing nothing."; exit 0;
fi

dest_dir="${tgt_dir}.mod"
mkdir -p "${dest_dir}"

for file_path in ${tgt_dir}/*; do
  file="${file_path##*/}"
  echo "${file}"
  pipenv run ${dir}/../src/main/scripts/remove_lc "${file_path}" > "${dest_dir}/${file}"
done
