#!/usr/bin/env bash
#shellcheck shell=bash

ANNOTATION_FILE="$1"
if [[ ! -r "$ANNOTATION_FILE" ]]; then
  echo "Annotation file '$ANNOTATION_FILE' does not exist"
  exit 1
fi

# Kill all child jobs on exit
trap 'kill -- -$$' EXIT
readarray -t ANNOTATION_ENTRIES <<< "$(cat "$ANNOTATION_FILE")"

BAD_EXAMPLES=0
PLAYED_CARD_INFLUENCED=0
SCENE_ADDRESSES_CHALLENGE=0
SCENE_AND_CHALLENGE_INFLUENCED=0
for entry in "${ANNOTATION_ENTRIES[@]}"; do
  if [[ "$(echo "$entry" | jq -r '.results.bad_example.agg')" == "true" ]]; then
    BAD_EXAMPLES=$((BAD_EXAMPLES + 1))
  else
    ANNOTATION_INFLUENCE=0
    if [[ "$(echo "$entry" | jq -r '.results.played_card_influence.agg')" == "true" ]]; then
      ANNOTATION_INFLUENCE=$((ANNOTATION_INFLUENCE + 1))
      PLAYED_CARD_INFLUENCED=$((PLAYED_CARD_INFLUENCED + 1))
    fi

    if [[ "$(echo "$entry" | jq -r '.results.scene_addresses_challenge.agg')" == "true" ]]; then
      ANNOTATION_INFLUENCE=$((ANNOTATION_INFLUENCE + 1))
      SCENE_ADDRESSES_CHALLENGE=$((SCENE_ADDRESSES_CHALLENGE + 1))
    fi

    if [[ "$ANNOTATION_INFLUENCE" -eq 2 ]]; then
      SCENE_AND_CHALLENGE_INFLUENCED=$((SCENE_AND_CHALLENGE_INFLUENCED + 1))
    fi
  fi
done

VALID_ANNOTATIONS=$((${#ANNOTATION_ENTRIES[@]} - BAD_EXAMPLES))
PLAYED_CARD_INFLUENCED_PERCENTAGE="$(echo "scale=4; $PLAYED_CARD_INFLUENCED / $VALID_ANNOTATIONS * 100" | bc)"
PLAYED_CARD_INFLUENCED_PERCENTAGE="${PLAYED_CARD_INFLUENCED_PERCENTAGE%%0*}" # Remove trailing zeros from bc output

SCENE_ADDRESSES_CHALLENGE_PERCENTAGE="$(echo "scale=4; $SCENE_ADDRESSES_CHALLENGE / $VALID_ANNOTATIONS * 100" | bc)"
SCENE_ADDRESSES_CHALLENGE_PERCENTAGE="${SCENE_ADDRESSES_CHALLENGE_PERCENTAGE%%0*}" # Remove trailing zeros from bc output

SCENE_AND_CHALLENGE_INFLUENCED_PERCENTAGE="$(echo "scale=4; $SCENE_AND_CHALLENGE_INFLUENCED / $VALID_ANNOTATIONS * 100" | bc)"
SCENE_AND_CHALLENGE_INFLUENCED_PERCENTAGE="${SCENE_AND_CHALLENGE_INFLUENCED_PERCENTAGE%%0*}" # Remove trailing zeros from bc output

echo "total #annotations=${#ANNOTATION_ENTRIES[@]}"
echo "#bogus entries=$BAD_EXAMPLES"
echo "#valid annotations=$VALID_ANNOTATIONS"
echo "#cards influencing scene entry=$PLAYED_CARD_INFLUENCED ($PLAYED_CARD_INFLUENCED_PERCENTAGE%)"
echo "#scene entries addressing challenge=$SCENE_ADDRESSES_CHALLENGE ($SCENE_ADDRESSES_CHALLENGE_PERCENTAGE%)"
echo "#intersection of card and entry influence=$SCENE_AND_CHALLENGE_INFLUENCED ($SCENE_AND_CHALLENGE_INFLUENCED_PERCENTAGE%)"
