#!/bin/bash
#shellcheck shell=bash

CURRENT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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
STATS_PATH="$OUTPUT_DIR/storium_stats.tex"
stats_header() {
  cat << EOF > "$STATS_PATH"
\begin{tabular}{|l|c|c|c|}
\hline
\bf Feature & \bf Total & \bf Mean & \bf Std Dev \\\\
\hline
EOF
}

stats_footer() {
  cat << EOF >> "$STATS_PATH"
\end{tabular}
EOF
}

stats_header

# Kill all child jobs on exit
trap 'kill -- -$$' EXIT

get_stories() {
  CMD=("find" "$DATA_DIR" "-iname" "'*.json'")
  eval "${CMD[@]}" "$*"
}

process_parallel() {
  # We need a lockfile for writing to output otherwise xargs may behave in a
  # non-deterministic way due to parallel processing
  LOCKFILE="$(mktemp)"
  #shellcheck disable=SC2064
  trap "rm -rf $LOCKFILE" EXIT TERM RETURN

  # Note that the use of `bash -c` actually puts the first argument in $0
  # rather than $1 (which is normally) reserved for the name of the script. So
  # below we have to explicitly include $0 otherwise, the first argument will
  # be ignored :/

  #shellcheck disable=SC2016
  xargs -0 -n 16 -P 0 bash -c '
    {
      OUTPUT="$(jq '"$*"' "$0" $*)"
      flock -x 9
      printf "%s\n" "$OUTPUT"
    } 9>'"$LOCKFILE"
}

process_stories() {
  get_stories "-print0" "|" "process_parallel ${*@Q}"
}

compute_sum() {
  paste -sd+ - | bc
}

compute_stats() {
  awk '{num += 1; total += $1; delta = $1 - mean; mean += delta / num; delta2 = $1 - mean; m2 += delta * delta2;}
  END { printf "total=%d, mean=%.2f, std=%.2f", total, mean, sqrt(m2 / (num - 1)) }'
}

write_stats() {
  if [[ "$#" -ne 1 ]]; then
    echo "Incorrect number of arguments for function write_stats"
    exit 1
  fi

  STATS="$(cat)"
  IFS=',' read -ra STATS_ARRAY <<< "$STATS"

  while [[ ${#STATS_ARRAY[@]} -lt 3 ]]; do
    STATS_ARRAY+=("")
  done

  local IFS='&'
  echo "$1&${STATS_ARRAY[*]#*=} \\\\" >> "$STATS_PATH"
  echo '\hline' >> "$STATS_PATH"

  echo "#$1: $STATS"
}

filter_stats() {
  # Use a 95% confidence interval
  INTERVAL=$(awk -F '[=, ]+' '{mean=$4; std=$6;} END {printf "%f", mean + 1.96 * std}' <<<"$2")
  awk '$1 > '"$INTERVAL"' {next;} $0' < "$1"
}

extract_total() {
  echo "$@" | awk -F '[=, ]+' '{printf $2}'
}

extract_mean() {
  echo "$@" | awk -F '[=, ]+' '{printf $4}'
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

FIGURES_PATH="$OUTPUT_DIR/storium_figures.tex"
truncate -cs 0 "$FIGURES_PATH"

compute_stats_and_plot() {
  if [[ "$#" -lt 1 ]]; then
    echo "compute_stats_and_plot is missing a parameter!"
    exit 1
  fi

  TEMPFILE="$(mktemp)"
  #shellcheck disable=SC2064
  trap "rm -rf $TEMPFILE" EXIT TERM RETURN
  STATS="$(tee "$TEMPFILE" | compute_stats)"

  if [[ "$2" == "nofilter" ]]; then
    TEMPFILE2="$TEMPFILE"
  else
    TEMPFILE2="$(mktemp)"
    #shellcheck disable=SC2064
    trap "rm -rf $TEMPFILE2" EXIT TERM RETURN
    filter_stats "$TEMPFILE" "$STATS" > "$TEMPFILE2"
  fi

  PREV_DIR="$PWD"
  cd "$OUTPUT_DIR" || exit
  mkdir -p "figs"

  PLOT_FILENAME="${1,,}"
  PLOT_FILENAME="figs/${PLOT_FILENAME//[ \/]/_}"

  BINWIDTH="$(echo "($(extract_std "$STATS") + 2) / 2" | bc)"
  IFS=',' read -ra STATS_ARRAY <<< "$STATS"

  make_label() {
    echo "set label $1 '$2' at graph .95,$(echo ".95-.075*$1" | bc) right front"
  }

  STAT_LABELS=()
  for i in "${!STATS_ARRAY[@]}"; do
    STAT_LABELS+=("$(make_label "$((i + 1))" "${STATS_ARRAY[$i]}")")
  done
  STAT_LABELS+=("$(make_label "$((${#STATS_ARRAY[@]} + 1))" "bin width=$BINWIDTH")")

  local IFS=$'\n'
  gnuplot << EOF
set size 0.65, 0.65
set output "$PLOT_FILENAME.tex"
set term epslatex color colortext

set style data histograms
set style histogram cluster

set key noautotitle
${STAT_LABELS[*]}

set xlabel "$1"
set ylabel "Count"
set xtics rotate by -45
set xrange[0:]

binstart=0
binwidth=$BINWIDTH
load '$CURRENT_DIR/hist.fct'
plot "$TEMPFILE2" i 0 @hist ls 1
EOF

  cd "$PREV_DIR" || exit

  echo "\\input{$PLOT_FILENAME}" >> "$FIGURES_PATH"
  echo "$STATS"
}

NUM_STORIES="$(get_stories | wc -l)"
echo "$NUM_STORIES" | write_stats "stories"
process_stories "'.scenes | map(select(.is_final == true and .is_ended == true)) | length'" | compute_sum | write_stats "completed stories"
process_stories "'.scenes[].entries[].user_pid'" | sort | uniq | wc -l | write_stats "users"
process_stories "'.characters | length'" | compute_stats | write_stats "characters created"

compute_player_stats () {
  PLAYED_CHARACTERS="$(process_stories "'[.scenes[].entries[].character_seq_id | values] | unique | length'" | compute_stats)"
  echo "$PLAYED_CHARACTERS" | write_stats "characters played"
  echo "$((NUM_STORIES + $(extract_total "$PLAYED_CHARACTERS")))" | write_stats "total played roles"
}

compute_player_stats
process_stories "'.scenes | length'" | compute_stats | write_stats "scenes"
process_stories "'.scenes[].entries | length'" | compute_stats_and_plot "Entries per Scene" | write_stats "scene entries"

BOOL2INT='def bool2int: if . == true then 1 else 0 end'
#shellcheck disable=SC2016
USER_PIDS='(reduce {(.details.narrator_user_pid, .characters[].user_pid): true} as $users ({ }; $users + .)) as $users'
PLAYED_CARDS='(.scenes[].entries[].place_card, .scenes[].entries[].challenge_cards[], .scenes[].entries[].cards_played_on_challenge[])'
#shellcheck disable=SC2016
IS_USER_CREATED='select(.is_wild == false) | [(((.author_user_pid | select(. != null)) | in($users)), (.is_edited | select(. == true)))] | any | bool2int'
process_stories "'$BOOL2INT; $USER_PIDS | [.cards[] | $IS_USER_CREATED] | add'" | compute_stats_and_plot "Cards Created/Edited per Story" | write_stats "cards created/edited"
process_stories "'$BOOL2INT; $USER_PIDS | [$PLAYED_CARDS | $IS_USER_CREATED] | add'" | compute_stats_and_plot "Played Cards Created/Edited per Story" | write_stats "played cards created/edited by users"

process_stories "'.scenes[].entries | map(select(.place_card != null)) | length'" | compute_stats | write_stats "location cards played by narrators"
process_stories "'.scenes[].entries[].challenge_cards | length'" | compute_stats_and_plot "Challenge Cards per Entry" | write_stats "challenge cards played by narrators"
process_stories "'.scenes[].entries[].cards_played_on_challenge | length'" | compute_stats_and_plot "Played Cards per Entry" | write_stats "cards played by characters"
process_stories "'.scenes[].entries[].cards_played_on_challenge | map(select(.via_wild_exchanged_for == \"new\")) | length'" | compute_stats_and_plot "Played Wild Cards per Entry" "nofilter" | write_stats "wild cards played by characters"
process_stories "'.scenes[].entries[].cards_played_on_challenge | map(select(.via_wild_exchanged_for != \"new\")) | length'" | compute_stats_and_plot "Played Regular Cards per Entry" | write_stats "regular cards played by characters"
process_stories "'[.scenes[].entries[].cards_played_on_challenge | length] | add | select(. == 0)'" | wc -l | write_stats "stories played without cards"

CHARACTERS='.characters[].description'
SCENES='.scenes[].entries[].description'
LOCATIONS='.scenes[].entries[].place_card.description'
CHALLENGES='.scenes[].entries[].cards_played_on_challenge[].description'
WILD_CARDS='.scenes[].entries[].cards_played_on_challenge | map(select(.via_wild_exchanged_for == "new")) | .[].description'
REGULAR_CARDS='.scenes[].entries[].cards_played_on_challenge | map(select(.via_wild_exchanged_for != "new")) | .[].description'

# Very simple tokenizer, equivalent to https://www.nltk.org/api/nltk.tokenize.html#nltk.tokenize.regexp.WordPunctTokenizer
TOKENIZER='select(. != null) | match("\\w+|[^\\w\\s]+"; "g") | .string'

process_stories "'$CHARACTERS | [$TOKENIZER] | length'" | compute_stats_and_plot "Tokens per Character Description" | write_stats "tokens in character descriptions"
process_stories "'$SCENES | [$TOKENIZER] | length'" | compute_stats_and_plot "Tokens per Scene Entry" | write_stats "tokens in scene entries"
process_stories "'$LOCATIONS | [$TOKENIZER] | length'" | compute_stats_and_plot "Tokens per Played Location Card" | write_stats "tokens in played location cards"
process_stories "'$CHALLENGES | [$TOKENIZER] | length'" | compute_stats_and_plot "Tokens per Played Challenge Card" | write_stats "tokens in played challenge cards"
process_stories "'$REGULAR_CARDS | [$TOKENIZER] | length'" | compute_stats_and_plot "Tokens per Played Regular Card" | write_stats "tokens in played regular cards"
process_stories "'$WILD_CARDS | [$TOKENIZER] | length'" | compute_stats_and_plot "Tokens per Played Wild Card" | write_stats "tokens in played wild cards"

process_stories "-r" "'$CHARACTERS, $SCENES, $LOCATIONS, $CHALLENGES | $TOKENIZER'" | \
  awk -F '\n' '{for(i = 1; i <= NF; i++) {a[$i]++}} END {for(k in a) print  k, a[k]}' | wc -l | write_stats "unique tokens"

stats_footer
