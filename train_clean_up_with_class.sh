python main_ppo_clean_up.py \
--exp_name="runner_clean_up_with_collective_reward_save_model_K=16" --use_rnn --device="cuda:0" --batch_size=2 --mini_batch_size=1  --use_wandb  --train_inner_steps=512 --train_episode=4096 --env_parallel_num=4 --use_collective_reward


