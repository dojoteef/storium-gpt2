#!/usr/bin/env bash
#shellcheck shell=bash

export LANG=C

set -e

usage()
{
  cat <<EOF
USAGE: $0 [-h] [-f] [-d DATA_DIR] [-o OUTPUT_DIR] [-v]
Options:
 -d             Data directory
 -f             Whether to force preprocessing
 -h             Show this help and exit
 -o             Output directory
 -v             Verbose output
EOF
  exit "$1"
}

# Defaults
DATA_DIR=""
OUTPUT_DIR=""
VERBOSE=""
FORCE=""

while getopts 'd:fho:v' o; do
  case "$o" in
    d)  DATA_DIR="$OPTARG";;
    f)  FORCE="-f";;
    h)  usage 0;;
    o)  OUTPUT_DIR="$OPTARG";;
    v)  VERBOSE="-v";;
    *)  usage 1 >&2;;
  esac
done
shift $((OPTIND - 1))

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PREPROCESSING_SCRIPT="$SCRIPT_DIR/../main.py"

if [[ -z "$DATA_DIR" ]]; then
  read -r -n 1 -p "No data directory specified. Use current directory '$PWD'? [Y/n] " USE_CURRENT_DIR; echo

  if [[ "$USE_CURRENT_DIR" =~ [Nn] ]]; then
    echo "No valid data directory specified, exiting."
    exit 0
  fi
elif [[ ! -d "$DATA_DIR" ]]; then
  read -r -n 1 -p "Output directory '$DATA_DIR' does not exist. Create it? [Y/n] " CREATE_DATA_DIR; echo

  if [[ "$CREATE_DATA_DIR" =~ [Nn] ]]; then
    echo "No valid data directory specified, exiting."
    exit 0
  fi

  if ! mkdir -p "$DATA_DIR" 2> /dev/null; then
    echo "Failed to create '$DATA_DIR'!"
    exit 1
  fi
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  read -r -n 1 -p "No output directory specified. Use current directory '$PWD'? [Y/n] " USE_CURRENT_DIR; echo

  if [[ "$USE_CURRENT_DIR" =~ [Nn] ]]; then
    echo "No valid output directory specified, exiting."
    exit 0
  fi
elif [[ ! -d "$OUTPUT_DIR" ]]; then
  read -r -n 1 -p "Output directory '$OUTPUT_DIR' does not exist. Create it? [Y/n] " CREATE_OUTPUT_DIR; echo

  if [[ "$CREATE_OUTPUT_DIR" =~ [Nn] ]]; then
    echo "No valid output directory specified, exiting."
    exit 0
  fi

  if ! mkdir -p "$OUTPUT_DIR" 2> /dev/null; then
    echo "Failed to create '$OUTPUT_DIR'!"
    exit 1
  fi
fi

# Given a sequence of tokens T1, decoding it into text, then re-encoding it back into a
# sequence of tokens T2 might lead to T1 != T2 due to the way huggingface's GPT-2 tokenization
# works. For example, it seems that sometimes it encodes "\n\n" as a single token, and other
# times it encodes it as two tokens. Therefore to ensure we don't accidentally go over the max
# 2048 tokens for a single example provided to GPT-3, we reduce the max example length a bit
# to try to account for this slop.
python "$PREPROCESSING_SCRIPT" "$VERBOSE" --data-dir "$DATA_DIR" --output-dir "$OUTPUT_DIR" preprocess \
    --dataset gpt3 --history 2 --character-history 2 --max-length 2042 --max-tokens 20000000 \
    --preferred-entry-length 256 --min-completion-length 100 "$FORCE"
