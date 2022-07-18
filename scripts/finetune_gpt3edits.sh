#!/usr/bin/env bash
#shellcheck shell=bash

export LANG=C

set -e

usage()
{
  cat <<EOF
USAGE: $0 [-h] [-m MODEL] [-n] [-t TRAIN_FILE] [-v VALIDATION_FILE]
Options:
 -h             Show this help and exit
 -m             Which model to finetune
 -n             Dry run
 -t             Preprocessed training dataset jsonl file (required)
 -v             Preprocessed validation dataset jsonl file (optional)
EOF
  exit "$1"
}

# Defaults
DRY_RUN=0
MODEL="ada"
while getopts 'hm:nt:v:' o; do
  case "$o" in
    h)  usage 0;;
    m)  MODEL="$OPTARG";;
    n)  DRY_RUN=1;;
    t)  TRAIN_FILE="$OPTARG";;
    v)  VALIDATION_FILE="$OPTARG";;
    *)  usage 1 >&2;;
  esac
done
shift $((OPTIND - 1))

if [[ -z "$TRAIN_FILE" ]]; then
    usage 1 >&2
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

GIT_HASH="$( cd "$SCRIPT_DIR" && git describe --always --dirty)"
if [[ "$GIT_HASH" == *-dirty ]]; then
    echo "Git workspace is dirty! Commit, stash, or revert changes before finetuning."
    exit 1
fi

CMD=()
if [[ $DRY_RUN -eq 1 ]]; then
    CMD+=("echo")
fi

CMD+=('openai' 'api' 'fine_tunes.create')
CMD+=("-m" "$MODEL")
CMD+=('-t' "$TRAIN_FILE")
if [[ -n "$VALIDATION_FILE" ]]; then
    CMD+=("-v" "$VALIDATION_FILE")
fi
CMD+=("--suffix" "$GIT_HASH")

"${CMD[@]}"
