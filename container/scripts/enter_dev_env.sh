#!/bin/bash

cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit

sudo docker-compose run --rm main bash
