#!/bin/zsh
## Run every (target, config, rep) of compileLatency.jl, NPAR processes at a time, then one trace run per (target, config).
##   benchmark/run.sh [NPAR]     (from the WENDy.jl worktree root)
cd "${0:A:h}/.."
NPAR=${1:-4}
JL=(~/.juliaup/bin/julia +1.13.3 --project=benchmark --threads=1)
mkdir -p benchmark/results/all
OUT=benchmark/results/all/timings.tsv
rm -f benchmark/results/all/*(N)
jobs=()
for rep in 1 2 3; do for t in logistic hindmarshRose; do for c in base cse wrap oejac cse+wrap+oejac; do
  jobs+=("$t $c $rep")
done; done; done
print -l $jobs | xargs -P $NPAR -L 1 zsh -c '"$0" "${@:1:3}" benchmark/compileLatency.jl $4 $5 $6 '"$OUT"' > /dev/null 2>&1 || echo "FAILED $4 $5 $6"' $JL
for t in logistic hindmarshRose; do for c in base cse wrap oejac cse+wrap+oejac; do
  print "$t $c"
done; done | xargs -P $NPAR -L 1 zsh -c '"$0" "${@:1:3}" benchmark/compileLatency.jl $4 $5 0 benchmark/results/all/traces.tsv --trace benchmark/results/all/trace_$4_$5.txt > /dev/null 2>&1 || echo "FAILED trace $4 $5"' $JL
wc -l $OUT
