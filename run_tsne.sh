#!/bin/bash

# Define the path to the YAML configuration file
yaml_path="conf/data/mymove_10s_5c_v2.yaml"
# "conf/data/realworld_wisdm.yaml"
# conf/data/realworld_10s_5c.yaml
# conf/data/wisdm_10s_few_class.yaml
# conf/data/mymove_10s_5c_v2.yaml
# Run the t-SNE script with the YAML file
python t-sne.py "$yaml_path"

# Check if the script ran successfully
if [ $? -ne 0 ]; then
    echo "An error occurred while running the script."
    exit 1
fi

echo "Script executed successfully."