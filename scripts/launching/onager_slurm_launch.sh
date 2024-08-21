cd ../../

PRELAUNCH_NAMES='pocman_dreamer pocman_dreamer_ld'

for item in $PRELAUNCH_NAMES;
do
  onager launch \
      --backend slurm \
      --jobname "$item" \
      --mem 24 \
      --cpus 3 \
      --duration 0-12:00:00 \
      --venv venv \
      --gpus 1 \
      --partition 3090-gcondo \
      --exclude gpu2507,gpu2608
      # --tasks-per-node 5
done
