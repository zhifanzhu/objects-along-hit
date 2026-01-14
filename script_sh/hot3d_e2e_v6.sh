#!/usr/bin/fish

HYDRA_FULL_ERROR=1 python potim/run_e2e_v6_segwise_ondemand.py \
    +exp=hot3d_e2e_v6 \
    hydra.run.dir=outputs/(date "+%Y-%m-%d")-hot3d/e2e_v6 \
    dataset.static_init_method='multi_upright' \
    optim_mv.num_iters=100
    # segi_strategy='circle', in arXiv results are run with 'circle'
