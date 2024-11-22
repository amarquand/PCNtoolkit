#!/bin/bash

# take the pcntoolkit function from the first argument specified
func="$1"
shift

# run using all remaining arguments
${func} "$@"