# simple
python run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -dsa --exp_name q1_sb_no_rtg_dsa
python run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg -dsa --exp_name q1_sb_rtg_dsa
python run_hw2.py --env_name CartPole-v0 -n 100 -b 1000 -rtg --exp_name q1_sb_rtg_na
python run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -dsa --exp_name q1_lb_no_rtg_dsa
python run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg -dsa --exp_name q1_lb_rtg_dsa
python run_hw2.py --env_name CartPole-v0 -n 100 -b 5000 -rtg --exp_name q1_lb_rtg_na

# InvertedPendulum

# python run_hw2.py --env_name InvertedPendulum-v4 --ep_len 1000 --discount 0.9 -n 100 -l 2 -s 64 -b <b*> -lr <r*> -rtg \
# --exp_name q2_b<b*>_r<r*>


# python cs285/scripts/run_hw2.py \
# --env_name LunarLanderContinuous-v2 --ep_len 1000 \
# --discount 0.99 -n 100 -l 2 -s 64 -b 40000 -lr 0.005 \
# --reward_to_go --nn_baseline --exp_name q3_b40000_r0.005


# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b> -lr <r> -rtg --nn_baseline \
# --exp_name q4_search_b<b>_lr<r>_rtg_nnbaseline


# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> \
# --exp_name q4_b<b*>_r<r*>
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> -rtg \
# --exp_name q4_b<b*>_r<r*>_rtg
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> --nn_baseline \
# --exp_name q4_b<b*>_r<r*>_nnbaseline
# python cs285/scripts/run_hw2.py --env_name HalfCheetah-v4 --ep_len 150 \
# --discount 0.95 -n 100 -l 2 -s 32 -b <b*> -lr <r*> -rtg --nn_baseline \
# --exp_name q4_b<b*>_r<r*>_rtg_nnbaseline


# python cs285/scripts/run_hw2.py \
# --env_name Hopper-v4 --ep_len 1000
# --discount 0.99 -n 300 -l 2 -s 32 -b 2000 -lr 0.001 \
# --reward_to_go --nn_baseline --action_noise_std 0.5 --gae_lambda <λ> \
# --exp_name q5_b2000_r0.001_lambda<λ>