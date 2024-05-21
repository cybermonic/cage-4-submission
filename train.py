from argparse import ArgumentParser
import os 
from types import SimpleNamespace

from joblib import Parallel, delayed
import torch
from tqdm import tqdm

from CybORG import CybORG
from CybORG.Agents import SleepAgent, EnterpriseGreenAgent, FiniteStateRedAgent
from CybORG.Simulator.Scenarios import EnterpriseScenarioGenerator

from models.cage4 import InductiveGraphPPOAgent
from models.memory_buffer import MultiPPOMemory
from wrapper.graph_wrapper import GraphWrapper
from wrapper.observation_graph import ObservationGraph

SEED = 1337
HYPER_PARAMS = SimpleNamespace(
    N = 25,             # How many episodes before training
    workers = 25,       # How many envs can run in parallel
    bs = 2500,          # How many steps to learn from at a time
    episode_len = 500,
    training_episodes = 500_000, # Realistically, stops improving around 50k
    epochs = 4
)

N_AGENTS = 5 
MAX_THREADS = 36 # 5 per subnet (20 for agent 4, 4 for all others)
torch.manual_seed(SEED)
torch.set_num_threads(MAX_THREADS)

@torch.no_grad()
def generate_episode_job(agents, env, hp, i):
    '''
    Per-process job to generate one episode of memories
    for all 5 agents. Returns `N_AGENTS` memory buffers, 
    and the total reward for the episode. 

    Args: 
        agents:     list of keep.cage4.InductiveGraphAgent objects 
        env:        wrapped cyborg object 
        hp:         hyperparameter namespace 
        i:          process id in range(0, `hp.workers`)
    '''
    torch.set_num_threads(MAX_THREADS // hp.workers)

    # Initialize environment
    env.reset()
    states = env.last_obs
    blocked_rewards = [0]*N_AGENTS

    tot_reward = 0
    memory_buffers = MultiPPOMemory(hp.bs)

    # Begin episode 
    for ts in tqdm(range(hp.episode_len), desc=f'Worker {i}'):
        actions = dict()
        memories = dict()

        # Get actions for all unblocked agents
        for k,(state,blocked) in states.items():
            i = int(k[-1])
            if blocked:
                actions[k] = None
            else:
                action,value,prob = agents[i].get_action((state,blocked))
                memories[i] = (state,action,value,prob)
                actions[k] = action

        next_state, rewards, _,_,_ = env.step(actions)
        rewards = list(rewards.values())
        tot_reward += sum(rewards)/N_AGENTS

        # Delay recieving rewards until multi-step actions are completed. 
        # Agents recieve cumulative reward for all the timesteps 
        # they spent performing their action. 
        for i in range(N_AGENTS):
            if i in memories:
                s,a,v,p = memories[i]
                r = rewards[i] + blocked_rewards[i]
                t = 0 if ts < hp.episode_len-1 else 1

                memory_buffers.remember(i, s,a,v,p, r,t)
                blocked_rewards[i] = 0
            else:
                blocked_rewards[i] += rewards[i]

        states = next_state

    return memory_buffers.mems, tot_reward

def train(agents, hp, seed=SEED):
    [agent.train() for agent in agents]
    log = []

    # Only call constructors once out here to save some time
    envs = []
    for i in range(min(hp.workers, hp.N)):
        sg = EnterpriseScenarioGenerator(
            blue_agent_class=SleepAgent,
            green_agent_class=EnterpriseGreenAgent,
            red_agent_class=FiniteStateRedAgent,
            steps=hp.episode_len,
        )
        env = CybORG(sg, "sim", seed=seed)
        envs.append(GraphWrapper(env))

    # Define learn function for threads to call later so we can 
    # parallelize the backprop step. Use more threads for Agent 4 
    # because they're managing 3 subnets instead of 1 (bigger graph/matrices)
    # Still not perfectly load-balanced, but close enough
    def learn(i):
            if i < 4:
                torch.set_num_threads(MAX_THREADS // 9)
            else:
                torch.set_num_threads((MAX_THREADS // 9) * N_AGENTS)
            return agents[i].learn()

    # Begin training loop 
    for e in range(hp.training_episodes // hp.N):
        e *= hp.N

        # Generate N episodes in parallel 
        out = Parallel(prefer='processes', n_jobs=hp.workers)(
            delayed(generate_episode_job)(agents, envs[i % len(envs)], hp, i) for i in range(hp.N)
        )

        # Concat memories across episodes, and transfer them to agents' 
        # internal memory buffers 
        memories, avg_rewards = zip(*out)
        memories = [list(m) for m in zip(*memories)]
        for i in range(N_AGENTS):
            agents[i].memory.mems = memories[i]

        # Use threads because agents are in heap memory 
        # Parallel backpropagation 
        print("Updating")
        last_losses = Parallel(prefer='threads', n_jobs=N_AGENTS)(
            delayed(learn)(i) for i in range(N_AGENTS)
        )

        losses = ','.join([f'{last_losses[i]:0.4f}' for i in range(N_AGENTS)])
        print(f"[{e}] Loss: [{losses}]")

        # Log average reward across all episodes 
        avg_reward = sum(avg_rewards) / hp.N
        print(f"Avg reward for episode: {avg_reward}")
        log.append((avg_reward,e,sum(last_losses)/N_AGENTS))
        torch.save(log, f'logs/{hp.fnames}.pt')

        # Checkpoint model states 
        for i in range(N_AGENTS):
            agent = agents[i]
            agent.save(outf=f'checkpoints/{hp.fnames}-{i}_checkpoint.pt')

            if e % 10_000 < hp.N and e > hp.N:
                agent.save(outf=f'checkpoints/{hp.fnames}-{i}_{e//1000}k.pt')


if __name__ == '__main__':
    ap = ArgumentParser()
    ap.add_argument('fname', help='Required: the name to save output files as.')
    ap.add_argument('--hidden', action='store', type=int, default=256, help='Dimension of middle layer for actor/critic')
    ap.add_argument('--embedding', action='store', type=int, default=128, help='Dimension of node representation for actor/critic')

    args = ap.parse_args()
    print(args)

    # Add directory for log files
    if not os.path.exists('logs'):
        os.mkdir('logs')

    # Add directory for model weights 
    if not os.path.exists('checkpoints'):
        os.mkdir('checkpoints')

    # Add 5 extra dimensions to observation graph: 
    #   2 for tabular data (gets appended to relevant hosts)
    #   3 for message data (gets appended to relevant subnets): 
    #       1 bit if subnet has comprimised host in it
    #       1 bit if subnet has scanned host in it
    #       1 bit if message was sent successfully 
    #
    # All handled in wrapper.graph_wrapper
    agents = [InductiveGraphPPOAgent(
        ObservationGraph.DIM+5,
        bs=HYPER_PARAMS.bs,
        a_kwargs={'lr': 0.0003, 'hidden1': args.hidden, 'hidden2': args.embedding},
        c_kwargs={'lr': 0.001, 'hidden1': args.hidden, 'hidden2': args.embedding},
        clip=0.2,
        epochs=HYPER_PARAMS.epochs
    ) for _ in range(N_AGENTS)]

    HYPER_PARAMS.fnames = args.fname
    train(agents, HYPER_PARAMS)