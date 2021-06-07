#!/bin/bash
set -o xtrace

# =================================================================================
# Boilerplate for parsing options, mostly from https://stackoverflow.com/a/29754866
# =================================================================================

# -allow a command to fail with !’s side effect on errexit
# -use return value from ${PIPESTATUS[0]}, because ! hosed $?
! getopt --test > /dev/null
if [[ ${PIPESTATUS[0]} -ne 4 ]]; then
    echo 'I’m sorry, `getopt --test` failed in this environment.'
    exit 1
fi

# ===================================================
# List of possible options (both short and long form)
# ===================================================
OPTIONS=hn:s:
LONGOPTS=help,number:,sleep:

# -regarding ! and PIPESTATUS see above
# -temporarily store output to be able to check for errors
# -activate quoting/enhanced mode (e.g. by writing out “--options”)
# -pass arguments only via   -- "$@"   to separate them correctly
! PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTS --name "$0" -- "$@")
if [[ ${PIPESTATUS[0]} -ne 0 ]]; then
    # e.g. return value is 1
    #  then getopt has complained about wrong arguments to stdout
    exit 2
fi
# read getopt’s output this way to handle the quoting right:
eval set -- "$PARSED"

# ===================================================
# End of boilerplate - our script begins
# ===================================================

function help {
    echo -e "\nUsage: $0 -n instances -- [command to run]"
    echo -e "\nRun command multiple times"
    echo -e "\nOptions:"
    echo -e "  -s, --sleep       How long to sleep between commands (seconds)"
    echo -e "  -n, --number      Amount of instances to run (default: 3)"
    exit 0
}



instances=2
sleep_time=0

while true; do
    case "$1" in
        -h|--help)
            help
            shift
            ;;
        -n|--number)
            instances="$2"
            shift 2
            ;;
        -s|--sleep)
            sleep_time="$2"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            echo "Unexpected error (option $1)"
            exit 3
            ;;
    esac
done

screen -dmS workers bash
echo "Starting commands instances"
for i in $(seq 1 $instances); do
  screen -S workers -X screen -t "worker_$i" -- bash -c "$@; bash"
  sleep $sleep_time
done
