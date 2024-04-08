python main.py \
--exp_name="refactor_clean_up_ppo" --use_rnn \
--device="cuda:1" --batch_size=2 --mini_batch_size=1  --use_wandb  \
--train_inner_steps=512 --train_episode=1024 --env_parallel_num=16 \
--use_collective_reward --wandb_api_key="45c5299c29cdd6b3159ce61b98e1a08cca7401cb"


