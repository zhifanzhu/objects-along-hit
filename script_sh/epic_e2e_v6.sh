#!/usr/bin/fish

HYDRA_FULL_ERROR=1 python potim/run_e2e_v6_segwise_ondemand.py \
    +exp=epic_e2e_v6 \
    dataset.json_path=./code_epichor/timelines/epic_hit.json \
    hydra.run.dir=outputs/(date "+%Y-%m-%d")-epic/e2e_v6 \
    segi_strategy='circle' \
    optim_mv.num_iters=200 \
    use_old_sca=False