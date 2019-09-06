#!/bin/bash
#shellcheck shell=bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
DATA_DIR="$1"
if [[ ! -d "$DATA_DIR" ]]; then
  echo "'$DATA_DIR' directory does not exist"
  exit 1
fi

TEMPFILE="$(mktemp)"
remove_tempfile() {
  rm -rf "$TEMPFILE"
}
trap remove_tempfile EXIT

get_stories() {
  CMD=("find" "$DATA_DIR" "-iname" "'*.json'")
  eval "${CMD[@]}" "$*"
}

compute_sum() {
  paste -sd+ - | bc
}

compute_stats() {
  awk '{num += 1; total += $1; delta = $1 - mean; mean += delta / num; delta2 = $1 - mean; m2 += delta * delta2;}
  END { printf "total=%d, mean=%.2f, std=%.3f", total, mean, sqrt(m2 / (num - 1)) }'
}

extract_total() {
  echo "$@" | awk -F '[=, ]+' '{printf $2}'
}

extract_std() {
  echo "$@" | awk -F '[=, ]+' '{printf $6}'
}

sum_totals() {
  SUM=0
  for stat in "$@"
  do
    SUM=$((SUM + $(extract_total "$stat")))
  done
  echo "$SUM"
}

compute_stats_and_plot() {
  STATS="$(tee "$TEMPFILE" | compute_stats)"
  echo "$STATS"
  PLOT_FILENAME="${1,,}"
  PLOT_FILENAME="${PLOT_FILENAME//[ \/]/_}.eps"

  BINWIDTH="$(extract_std "$STATS")"
  gnuplot << EOF
set terminal dumb
set style data histograms
set style histogram cluster
set xlabel "$1"
set ylabel "Count"
set xrange[0:]

set term post eps enhanced color
set out "$2.eps"

binstart=0
binwidth=0.5 * $BINWIDTH
load '$CURRENT_DIR/hist.fct'
plot "$TEMPFILE" i 0 @hist ls 1 title ""
EOF
}

echo "GENERAL STATS"
NUM_STORIES="$(get_stories | wc -l)"
echo "#stories: $NUM_STORIES"
echo "#completed stories: $(get_stories "-exec" "jq" "'.scenes | map(select(.is_final == true and .is_ended == true)) | length'" "{}" "\\+" | compute_sum)"
echo "#users: $(get_stories "-exec" "jq" "'.scenes[].entries[].user_pid'" "{}" "\\+" | sort | uniq | wc -l)"
echo "#characters created: $(get_stories "-exec" "jq" "'.characters | length'" "{}" "\\+" | compute_stats)"

PLAYED_CHARACTERS="$(get_stories "-exec" "jq" "'[.scenes[].entries[].character_seq_id | values] | unique | length'" "{}" "\\+" | compute_stats)"
echo "#characters played: $PLAYED_CHARACTERS"
echo "#total played roles: $((NUM_STORIES + $(extract_total "$PLAYED_CHARACTERS"))) (NOTE: each story has a narrator who is not counted as a character)"
echo "#scenes: $(get_stories "-exec" "jq" "'.scenes | length'" "{}" "\\+" | compute_stats)"
echo "#scene entries: $(get_stories "-exec" "jq" "'.scenes[].entries | length'" "{}" "\\+" | compute_stats_and_plot "Entries per Scene" "entries_per_scene")"
echo ""

echo "CARD STATS"
BOOL2INT='def bool2int: if . == true then 1 else 0 end'
#shellcheck disable=SC2016
USER_PIDS='(reduce {(.details.narrator_user_pid, .characters[].user_pid): true} as $users ({ }; $users + .)) as $users'
PLAYED_CARDS='(.scenes[].entries[].place_card, .scenes[].entries[].challenge_cards[], .scenes[].entries[].cards_played_on_challenge[])'
#shellcheck disable=SC2016
IS_USER_CREATED='select(.is_wild == false) | [(((.author_user_pid | select(. != null)) | in($users)), (.is_edited | select(. == true)))] | any | bool2int'
echo "#cards created/edited: $(get_stories "-exec" "jq" "'$BOOL2INT; $USER_PIDS | [.cards[] | $IS_USER_CREATED] | add'" "{}" "\\+" | compute_stats_and_plot "Cards Created/Edited per Story" "cards_created_or_edited_per_story")"
echo "#played cards created/edited by users: $(get_stories "-exec" "jq" "'$BOOL2INT; $USER_PIDS | [$PLAYED_CARDS | $IS_USER_CREATED] | add'" "{}" "\\+" | compute_stats_and_plot "Played Cards Created/Edited per Story" "played_cards_created_or_edited_per_story")"

LOCATION_CARDS="$(get_stories "-exec" "jq" "'.scenes[].entries | map(select(.place_card != null)) | length'" "{}" "\\+" | compute_stats)"
CHALLENGE_CARDS="$(get_stories "-exec" "jq" "'.scenes[].entries[].challenge_cards | length'" "{}" "\\+" | compute_stats)"
PLAYED_ON_CHALLENGE_CARDS="$(get_stories "-exec" "jq" "'.scenes[].entries[].cards_played_on_challenge | length'" "{}" "\\+" | compute_stats)"
echo "#location cards played by narrators: $LOCATION_CARDS"
echo "#challenge cards played by narrators: $CHALLENGE_CARDS"
echo "#cards played by characters: $PLAYED_ON_CHALLENGE_CARDS"
echo "#wild cards played by characters: $(get_stories "-exec" "jq" "'.scenes[].entries[].cards_played_on_challenge | map(select(.via_wild_exchanged_for == \"new\")) | length'" "{}" "\\+" | compute_stats)"
echo "#total played cards: $(sum_totals "$LOCATION_CARDS" "$CHALLENGE_CARDS" "$PLAYED_ON_CHALLENGE_CARDS")"
echo ""

echo "TOKEN STATS"
CHARACTERS='.characters[].description'
SCENES='.scenes[].entries[].description'
LOCATIONS='.scenes[].entries[].place_card.description'
CHALLENGES='.scenes[].entries[].cards_played_on_challenge[].description'
WILD_CARDS='.scenes[].entries[].cards_played_on_challenge | map(select(.via_wild_exchanged_for != "new")) | .[].description'

# Very simple tokenizer, equivalent to https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.regexp.WordPunctTokenizer
TOKENIZER='select(. != null) | match("\\w+|[^\\w\\s]+"; "g") | .string'

CHARACTER_TOKENS="$(get_stories "-exec" "jq" "'$CHARACTERS | [$TOKENIZER] | length'" "{}" "\\+" | compute_stats)"
echo "#tokens in character descriptions: $CHARACTER_TOKENS"

SCENE_TOKENS="$(get_stories "-exec" "jq" "'$SCENES | [$TOKENIZER] | length'" "{}" "\\+" | compute_stats)"
echo "#tokens in scene entries: $SCENE_TOKENS"

LOCATION_TOKENS="$(get_stories "-exec" "jq" "'$LOCATIONS | [$TOKENIZER] | length'" "{}" "\\+" | compute_stats)"
echo "#tokens in played location cards: $LOCATION_TOKENS"

CHALLENGE_TOKENS="$(get_stories "-exec" "jq" "'$CHALLENGES | [$TOKENIZER] | length'" "{}" "\\+" | compute_stats)"
echo "#tokens in played challenge cards: $CHALLENGE_TOKENS"

WILD_TOKENS="$(get_stories "-exec" "jq" "'$WILD_CARDS | [$TOKENIZER] | length'" "{}" "\\+" | compute_stats)"
echo "#tokens in played wild cards: $WILD_TOKENS"

echo "#total tokens: $(sum_totals "$CHARACTER_TOKENS" "$SCENE_TOKENS" "$LOCATION_TOKENS" "$WILD_TOKENS")"
echo "#unique tokens $(\
  get_stories "-exec" "jq" "-r" "'$CHARACTERS, $SCENES, $LOCATIONS, $CHALLENGES | $TOKENIZER'" "{}" "\\+" | \
  awk -F '\n' '{for(i = 1; i <= NF; i++) {a[$i]++}} END {for(k in a) print  k, a[k]}' | wc -l)"
