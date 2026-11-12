#!/bin/zsh
## OE-LS only: every (target, config, rep) of compileLatency.jl --oe-only, NPAR processes at a time, then one trace run
## per (target, config). Writes benchmark/results/oe/.
##   benchmark/runOE.sh [NPAR] [REPS]     (from the WENDy.jl worktree root)
cd "${0:A:h}/.."
NPAR=${1:-4}; REPS=${2:-5}
## CONFIG labels the runs (default: the WENDy commit); PROJECT is the environment (default benchmark/, this worktree)
CONFIGS=(${CONFIG:-$(git rev-parse --short HEAD)})
JL=(~/.juliaup/bin/julia +1.13.3 --project=${PROJECT:-benchmark} --threads=1)
R=benchmark/results/oe
mkdir -p $R # results accumulate: run once per WENDy (CONFIG, PROJECT) to compare versions
for rep in $(seq $REPS); do for t in logistic hindmarshRose; do for c in $CONFIGS; do print "$t $c $rep"; done; done; done |
  xargs -P $NPAR -L 1 zsh -c '"$0" "${@:1:3}" benchmark/compileLatency.jl $4 $5 $6 '"$R/timings.tsv"' --oe-only > /dev/null 2>&1 || echo "FAILED $4 $5 $6"' $JL
for t in logistic hindmarshRose; do for c in $CONFIGS; do print "$t $c"; done; done |
  xargs -P $NPAR -L 1 zsh -c '"$0" "${@:1:3}" benchmark/compileLatency.jl $4 $5 0 '"$R/traces.tsv"' --oe-only --trace '"$R"'/trace_$4_$5.txt > /dev/null 2>&1 || echo "FAILED trace $4 $5"' $JL
wc -l $R/timings.tsv
