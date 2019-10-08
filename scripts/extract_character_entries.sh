#!/bin/bash
#shellcheck shell=bash

DATA_DIR="$1"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "Data directory '$DATA_DIR' does not exist"
  exit 1
fi

OUTPUT_DIR="$2"
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

OUTPUT_DIR="$PWD/$OUTPUT_DIR"
CSV_PATH="$OUTPUT_DIR/storium_entries.csv"

MAX_ENTRIES=${3:-200}
MAX_HAND_CARDS=0
MAX_PLAYED_CARDS=1
HEADERS="story_id,scene_index,entry_index,entry_text,character_name,character_description"
HEADERS+=",challenge_name,challenge_text,challenge_points,challenge_points_remaining,challenge_polarity,challenge_success_description,challenge_failure_description"
for ((i=1; i <= MAX_PLAYED_CARDS; i++)); do
  HEADERS+=",played_card_${i}_name,played_card_${i}_description,played_card_${i}_polarity"
done
for ((i=1; i <= MAX_HAND_CARDS; i++)); do
  HEADERS+=",players_hand_card_${i}_name,players_hand_card_${i}_description,players_hand_card_${i}_polarity"
done
echo "$HEADERS" > "$CSV_PATH"

# Kill all child jobs on exit
trap 'kill -- -$$' EXIT
readarray -t STORY_FILES <<< "$(find "$DATA_DIR" -iname '*.json')"

csv_escape() {
  if [[ "$#" -ne 1 ]]; then
    echo "Incorrect number of arguments for function csv_escape"
    exit 1
  fi
  echo "\"${1//\"/\"\"}\""
}

extract_entry() {
  while [[ -z "$COMPLETED_CSV_ENTRY" ]]; do
    FILENAME="${STORY_FILES[$((RANDOM % ${#STORY_FILES[@]}))]}"
    CSV_ENTRY="$(basename "$FILENAME" .json)"

    NUM_SCENES="$(jq -r '.scenes | length' "$FILENAME")"
    if [[ "$NUM_SCENES" -eq 0 ]]; then
      continue
    fi

    SCENE_IDX=$((RANDOM % NUM_SCENES))
    CSV_ENTRY+=",$SCENE_IDX"

    NUM_ENTRIES="$(jq -r ".scenes[$SCENE_IDX].entries | length" "$FILENAME")"
    if [[ "$NUM_ENTRIES" -eq 0 ]]; then
      continue
    fi

    ENTRY_IDX=$((RANDOM % NUM_ENTRIES))
    CSV_ENTRY+=",$ENTRY_IDX"

    SCENE_ENTRY=".scenes[$SCENE_IDX].entries[$ENTRY_IDX]"

    ENTRY_TEXT="$(jq -r "$SCENE_ENTRY.description" "$FILENAME")"
    CSV_ENTRY+=",$(csv_escape "$ENTRY_TEXT")"

    CHARACTER_ID="$(jq -r "$SCENE_ENTRY.character_seq_id" "$FILENAME")"
    if [[ "$CHARACTER_ID" == "null" ]]; then
      continue
    fi

    CHARACTER_NAME="$(jq -r ".characters[] | select(.character_seq_id == \"$CHARACTER_ID\").name" "$FILENAME")"
    CHARACTER_DESCRIPTION="$(jq -r ".characters[] | select(.character_seq_id == \"$CHARACTER_ID\").description" "$FILENAME")"
    CSV_ENTRY+=",$(csv_escape "$CHARACTER_NAME"),$(csv_escape "$CHARACTER_DESCRIPTION")"

    CHALLENGE_NAME="$(jq -r "$SCENE_ENTRY.target_challenge_card.name?" "$FILENAME")"
    if [[ "$CHALLENGE_NAME" == "null" ]]; then
      CSV_ENTRY+=",,,,,"
    else
      CSV_ENTRY+=",$(csv_escape "$CHALLENGE_NAME")"

      CHALLENGE_TEXT="$(jq -r "$SCENE_ENTRY.target_challenge_card.description" "$FILENAME")"
      CSV_ENTRY+=",$(csv_escape "$CHALLENGE_TEXT")"

      CHALLENGE_POINTS="$(jq -r "$SCENE_ENTRY.target_challenge_card.challenge_points" "$FILENAME")"
      CSV_ENTRY+=",$CHALLENGE_POINTS"

      CHALLENGE_CARDS_PLAYED="$(jq -r "$SCENE_ENTRY.target_challenge_card.challenge_points_polarities[:-1] | length" "$FILENAME")"
      CSV_ENTRY+=",$((CHALLENGE_POINTS - CHALLENGE_CARDS_PLAYED))"

      CHALLENGE_POLARITY="$(jq -r "$SCENE_ENTRY.target_challenge_card.challenge_points_polarities[:-1] | add | if . == null then 0 else . end" "$FILENAME")"
      CSV_ENTRY+=",$CHALLENGE_POLARITY"

      CHALLENGE_SUCCESS="$(jq -r "$SCENE_ENTRY.target_challenge_card.success_stakes" "$FILENAME")"
      CSV_ENTRY+=",$(csv_escape "$CHALLENGE_SUCCESS")"

      CHALLENGE_FAILURE="$(jq -r "$SCENE_ENTRY.target_challenge_card.failure_stakes" "$FILENAME")"
      CSV_ENTRY+=",$(csv_escape "$CHALLENGE_FAILURE")"
    fi

    NUM_CARDS_PLAYED="$(jq -r "$SCENE_ENTRY.cards_played_on_challenge | length" "$FILENAME")"
    if [[ "$NUM_CARDS_PLAYED" -eq 0 ]] || [[ "$NUM_CARDS_PLAYED" -gt "$MAX_PLAYED_CARDS" ]]; then
      continue
    fi

    for ((i=0; i < MAX_PLAYED_CARDS; i++)); do
      CARD_NAME="$(jq -r "$SCENE_ENTRY.cards_played_on_challenge[${i}].name | values" "$FILENAME")"
      CSV_ENTRY+=",$(csv_escape "$CARD_NAME")"

      CARD_TEXT="$(jq -r "$SCENE_ENTRY.cards_played_on_challenge[${i}].description | values" "$FILENAME")"
      CSV_ENTRY+=",$(csv_escape "$CARD_TEXT")"

      CARD_POLARITY="$(jq -r "$SCENE_ENTRY.cards_played_on_challenge[${i}].polarity | values" "$FILENAME")"
      CSV_ENTRY+=",$CARD_POLARITY"
    done

    for ((i=ENTRY_IDX; i >= 0; i--)); do
      PREV_SCENE_ENTRY=".scenes[$SCENE_IDX].entries[$i]"
      HAND_CONTEXT="$PREV_SCENE_ENTRY.hand_context.post"
      if [[ -n "$(jq -r "$HAND_CONTEXT | arrays | length" "$FILENAME")" ]]; then
        # The data is setup such that if you pickup a card and play it in the
        # same turn it will not give you the state of the hand after playing
        break
      fi

      HAND_CONTEXT="$PREV_SCENE_ENTRY.hand_context.pre"
      if [[ -n "$(jq -r "$HAND_CONTEXT | arrays | length" "$FILENAME")" ]]; then
        # What's even worse, is that it may not even mention what your current
        # hand is... Instead, you have to find continue looking up previous
        # entries until you find the player's current hand
          break
        fi
    done

    for ((i=0; i < MAX_HAND_CARDS; i++)); do
      CARD_ID="$(jq -r "${HAND_CONTEXT[${i}]}.card_id" "$FILENAME")"
      CARD_NAME="$(jq -r ".cards[] | select(.card_id == \"$CARD_ID\") | .name" "$FILENAME")"
      CSV_ENTRY+=",$(csv_escape "$CARD_NAME")"

      CARD_TEXT="$(jq -r ".cards[] | select(.card_id == \"$CARD_ID\") | .description" "$FILENAME")"
      CSV_ENTRY+=",$(csv_escape "$CARD_TEXT")"

      CARD_POLARITY="$(jq -r ".cards[] | select(.card_id == \"$CARD_ID\") | .polarity" "$FILENAME")"
      CSV_ENTRY+=",$CARD_POLARITY"
    done

    COMPLETED_CSV_ENTRY="$CSV_ENTRY"
  done

  echo "$COMPLETED_CSV_ENTRY" >> "$CSV_PATH"
}

for ((i=0; i < MAX_ENTRIES; i++)); do
  extract_entry &
done

wait
