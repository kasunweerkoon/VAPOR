import os
import numpy as np
from collections import deque
import torch
import wandb
import argparse
import glob
from utils import save, collect_random
import random
from agent import CQLSAC 
from torch.utils.data import DataLoader, TensorDataset
from import_off_data import ImportData
from off_env import Env

def get_config():
    parser = argparse.ArgumentParser(description='RL')
    parser.add_argument("--run_name", type=str, default="CQL", help="Run name, default: CQL")
    parser.add_argument("--env", type=str, default="CQL-outdoor-w-proprioception-w-attention", help="Gym environment name, default: CQL-outdoor")
    parser.add_argument("--episodes", type=int, default=250, help="Number of episodes, default: 100")
    parser.add_argument("--seed", type=int, default=1, help="Seed, default: 1")
    parser.add_argument("--log_video", type=int, default=0, help="Log agent behaviour to wanbd when set to 1, default: 0")
    parser.add_argument("--save_every", type=int, default=100, help="Saves the network every x epochs, default: 25")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size, default: 256")
    parser.add_argument("--hidden_size", type=int, default=256, help="")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="") # default 3e-4
    parser.add_argument("--temperature", type=float, default=1.0, help="")
    parser.add_argument("--cql_weight", type=float, default=1.0, help="")
    parser.add_argument("--target_action_gap", type=float, default=10, help="")
    parser.add_argument("--with_lagrange", type=int, default=0, help="")
    parser.add_argument("--tau", type=float, default=5e-3, help="")
    parser.add_argument("--eval_every", type=int, default=1, help="")
    
    args = parser.parse_args()
    return args

def prep_dataloader(batch_size, seed=1):

    env = Env(True,'/offlineRL/dataset')
    train_dataset = ImportData('/Media/offRL/dataset_w_prop/')

    dataloader  = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    return dataloader , env 

def evaluate(env, policy, eval_runs=5): 
    """
    Makes an evaluation run with the current policy
    """
    reward_batch = []
    for i in range(eval_runs):
        state = env.reset()

        rewards = 0
        while True:
            action = policy.get_action(state, eval=True)

            state, reward, done, _ = env.step(action)
            rewards += reward
            if done:
                break
        reward_batch.append(rewards)
    return np.mean(reward_batch)

def train(config):
    model_path = '/home/kasun/offlineRL/CQL/CQL-SAC-w-prop/training'
    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    dataloader, env = prep_dataloader(batch_size=config.batch_size, seed=config.seed)


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    batches = 0
    average10 = deque(maxlen=10)
    
    with wandb.init(project="Outdoor-CQL-offline_with_proprioception", name=config.run_name, config=config):
        
        agent = CQLSAC(state_size= 3, #env.observation_space.shape[0],
                        action_size= 3, #env.action_space.shape[0],
                        tau=config.tau,
                        hidden_size=config.hidden_size,
                        learning_rate=config.learning_rate,
                        temp=config.temperature,
                        with_lagrange=config.with_lagrange,
                        cql_weight=config.cql_weight,
                        target_action_gap=config.target_action_gap,
                        device=device)


        for i in range(1, config.episodes+1):
            tot_rewards =0

            for batch_idx, experience in enumerate(dataloader):
                # states, actions, rewards, next_states, dones = experience
                states_1,states_2, actions, rewards, next_states_1, next_states_2,dones = experience
                # tot_rewards += rewards

                states_1 = states_1.to(device).float()
                states_2 = states_2.to(device).float()
                actions = actions.to(device).float()
                rewards = rewards.to(device).float()
                next_states_1 = next_states_1.to(device).float()
                next_states_2 = next_states_2.to(device).float()
                dones = dones.to(device)
                # print("imput experince shapes s,a,r,s',d :",states_1.shape,states_2.shape,actions.shape,rewards.shape,next_states.shape,dones.shape)
                # print("input experince shapes s,a :",states_1.shape,states_2.shape,actions.shape)


                policy_loss, alpha_loss, bellmann_error1, bellmann_error2, cql1_loss, cql2_loss, current_alpha, lagrange_alpha_loss, lagrange_alpha = agent.learn((states_1,states_2, actions, rewards, next_states_1, next_states_2,dones))
                batches += 1

                # in_states= wandb.Image(states_1, caption="Map Inputs")
                # in_next_states= wandb.Image(next_states_1, caption="Next Map Inputs")

                # wandb.log({"Current State": in_states,
                #     "Next State": in_next_states})
     

            print("Policy loss: ",policy_loss)
            # print("episode reward:", rewards)

            
            wandb.log({
                       "Policy Loss": policy_loss,
                       "Alpha Loss": alpha_loss,
                       "Lagrange Alpha Loss": lagrange_alpha_loss,
                       "CQL1 Loss": cql1_loss,
                       "CQL2 Loss": cql2_loss,
                       "Bellman error 1": bellmann_error1,
                       "Bellman error 2": bellmann_error2,
                       "Alpha": current_alpha,
                       "Lagrange Alpha": lagrange_alpha,
                       "Batches": batches,
                       "Episode": i})


            # if i % config.save_every == 0:
            #     save(config, save_name="IQL", model=agent.actor_local, wandb=wandb, ep=0)
            if i % 40 == 0:
                torch.save(agent.state_dict(), os.path.join('./trained_models/','offrl_r5_attn_'+str(i)+'.pkl'))

if __name__ == "__main__":
    config = get_config()
    train(config)
