#!/bin/bash

# Install dari requirements.txt
pip install -q -r ./requirements.txt

# Install numpy versi < 2
pip install -q "numpy<2"

# Reinstall transformers
pip uninstall --yes -q transformers
pip install -q transformers

# Install kornia
pip install kornia
