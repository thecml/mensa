#!/bin/bash

# Run each script in sequence
./run_single_event_dgp.sh
./run_single_event.sh
./run_competing_risks.sh
./run_multi_event.sh