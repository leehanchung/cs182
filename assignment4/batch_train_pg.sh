python train_pg.py CartPole-v0 -n 100 -b 1000 -e 3      -dna --exp_name sb_no_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg -dna --exp_name sb_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 1000 -e 3 -rtg      --exp_name sb_rtg_na
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 3      -dna --exp_name lb_no_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg -dna --exp_name lb_rtg_dna
python train_pg.py CartPole-v0 -n 100 -b 5000 -e 3 -rtg      --exp_name lb_rtg_na
