#!/bin/bash

if [ -z "$1" ]; then
    echo "Usage: bash run_gbsa.sh <ROOT_DIR>"
    exit 1
fi

ROOT=$1

if [ ! -d "$ROOT" ]; then
    echo "Error: Directory $ROOT does not exist."
    exit 1
fi


GBSA_INPUT="Sample input file with decomposition analysis
&general
    startframe=1, endframe=1,
/
&gb
    igb=5, saltcon=0.150,
/
&decomp
    idecomp=2, dec_verbose=0,
/
"

echo "Root directory: $ROOT"
echo "========================="


for dir in ${ROOT}/*; do
    [ -d "$dir" ] || continue

    echo "Processing: $dir"

    mkdir -p "$dir/gbsa"

    echo "$GBSA_INPUT" > "$dir/gbsa/gbsa.in"

    (
        cd "$dir/gbsa" || exit

        MMPBSA.py -O \
            -i gbsa.in \
            -cp ../com.top \
            -rp ../pro.top \
            -lp ../lig.top \
            -y ../min.pdb
    )

done

echo "========================="
echo "All GBSA jobs finished."