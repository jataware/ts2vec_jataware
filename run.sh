

END=3
for ((seed=1;seed<=END;seed++)); do
for emb_dim in 64 128 256; do
for q_prop in .1 .5 .8; do
for n_samples in 100 1000 5000; do
for n_epochs in 50 100 300; do
    python horiz_exp.py \
        --emb_dim $emb_dim --q_prop $q_prop  \
        --q_samples 100 --n_samples $n_samples \
        --n_epochs $n_epochs --seed $seed
done
done
done
done
done