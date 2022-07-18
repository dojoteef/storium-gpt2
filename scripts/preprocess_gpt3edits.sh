#!/usr/bin/env bash
#shellcheck shell=bash

export LANG=C

set -e

usage()
{
  cat <<EOF
USAGE: $0 [-h] [-f] [-e EDITS_FILE] [-o OUTPUT_DIR] [-v]
Options:
 -e             Edits file
 -f             Whether to force preprocessing
 -h             Show this help and exit
 -o             Output directory
 -v             Verbose output
EOF
  exit "$1"
}

# Defaults
EDITS_FILE=""
OUTPUT_DIR=""
VERBOSE=""
FORCE=""

while getopts 'e:fho:v' o; do
  case "$o" in
    e)  EDITS_FILE="$OPTARG";;
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

if [[ ! -f "$EDITS_FILE" ]]; then
  echo "No valid edits file specified, exiting."
  exit 1
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
# to try to account for this slop. The largest GPT-3 model now supports 4000 tokens, so the
# max edit length needs to be reduced a bit since it's not 4096 as expected
python "$PREPROCESSING_SCRIPT" "$VERBOSE" --output-dir "$OUTPUT_DIR" preprocess \
    --dataset gpt3edits --history 2 --character-history 2 --max-length 2042 --max-tokens 20000000 \
    --preferred-entry-length 256 --max-edit-length 1946 $FORCE \
    --model-filter gpt3 --edits-file "$EDITS_FILE"
