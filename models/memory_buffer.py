import torch 

class PPOMemory:
    '''
    Holds memories for agents that are relevant to the 
    PPO optimization procedure
    '''
    def __init__(self, bs):
        self.s = []
        self.a = []
        self.v = []
        self.p = []
        self.r = []
        self.t = []

        self.bs = bs 

    def remember(self, s,a,v,p,r,t):
        '''
        Pushes new memory into the buffer 

        Args:
            s: State
            a: Action
            v: Value (critic output)
            p: Log Prob (actor output)
            r: Reward
            t: Terminal 
        '''
        self.s.append(s)
        self.a.append(a)
        self.v.append(v)
        self.p.append(p)
        self.r.append(r) 
        self.t.append(t)

    def clear(self): 
        '''
        Empties the memory buffer 
        '''
        self.s = []; self.a = []
        self.v = []; self.p = []
        self.r = []; self.t = []

    def get_batches(self):
        '''
        Return chunks of the shuffled memory buffer 
        randomly partitioned into `self.bs`-sized chunks 
        '''
        idxs = torch.randperm(len(self.a))
        batch_idxs = idxs.split(self.bs)

        return self.s, self.a, self.v, \
            self.p, self.r, self.t, batch_idxs


class MultiPPOMemory:
    '''
    Store multiple memory buffers, one for each agent. 
    Used during training to keep agent's observations seperated 
    '''
    def __init__(self, bs, agents=5) -> None:
        self.tot = agents 
        self.bs = bs 
        self.mems = [PPOMemory(bs) for _ in range(agents)]

    def remember(self, idx, *args):
        self.mems[idx].remember(*args)

    def clear(self):
        [mem.clear() for mem in self.mems]
        
    def get_batches(self): 
        offset = 0
        idxs = []
        all_s = []; all_a = []
        all_v = []; all_p = []
        all_r = []; all_t = []

        for i in range(self.tot):
            all_s += self.mems[i].s
            all_a += self.mems[i].a
            all_v += self.mems[i].v
            all_p += self.mems[i].p
            all_r += self.mems[i].r
            all_t += self.mems[i].t
            
            cnt = len(self.mems[i].s)
            idx = torch.randperm(cnt) + offset 
            idxs += list(idx.split(self.bs))
            offset += cnt 

        return all_s, all_a, all_v, all_p, all_r, all_t, idxs