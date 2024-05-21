from collections import defaultdict

from pprint import pprint
import torch

from CybORG.Simulator.Actions.AbstractActions import Remove, Restore, Analyse, Monitor
from CybORG.Simulator.Actions.ConcreteActions.DecoyActions import *
from CybORG.Shared.Enums import TernaryEnum

from wrapper.nodes import SystemNode, ConnectionNode, InternetNode, FileNode, init_decoy

class NodeTracker:
    '''
    Just a hash map with extra steps
    '''
    def __init__(self):
        self.nid = 0
        self.mapping = dict()
        self.inv_mapping = dict()

    def __getitem__(self, node_str):
        node_str = str(node_str)
        if (nid := self.mapping.get(node_str)) is not None:
            return nid

        # Add to dict if it doesn't exist
        self.mapping[node_str] = self.nid
        self.inv_mapping[self.nid] = node_str

        self.nid += 1
        return self.mapping[node_str]

    def pop(self, node_str):
        nid = self.mapping.get(node_str, None)
        if nid:
            self.mapping.pop(node_str)
            self.inv_mapping.pop(nid)

    def get(self, node_str):
        return self.mapping.get(node_str, None)

    def id_to_str(self, nid):
        return self.inv_mapping.get(nid)

    def names(self):
        return list(self.mapping.keys())

class ObservationGraph:
    '''
    The main datastructure powering KEEP. 
    This is a graph representing all activity observable to the agents. 
    There are 4 kinds of nodes: 
        SystemNode: Computers, servers and routers in the network 
        ConnectionNode: Processes that can talk to other hosts
        FileNode: Any new file observed on a host (usually malware)
        InternetNode: A structural node needed for evaluating P(Block/AllowConnection)
    '''
    NTYPES = {SystemNode: 0, ConnectionNode: 1, FileNode: 2, InternetNode: 3}
    INV_NTYPES = [SystemNode, ConnectionNode, FileNode, InternetNode]
    NTYPE_DIMS = [k(0,dict()).dim for k in INV_NTYPES]

    # one-hot vector of node type, plus whatever features that node has
    DIM = len(INV_NTYPES) + sum(NTYPE_DIMS) + 9

    # String representation of what the feature matrix means 
    FEAT_MAP = INV_NTYPES
    for k in INV_NTYPES:
        FEAT_MAP = FEAT_MAP + k(0,dict()).labels

    # How much to offset each node type's feature
    OFFSETS = [len(INV_NTYPES)]
    for n in NTYPE_DIMS[:-1]:
        OFFSETS.append(n + OFFSETS[-1])

    # Used for decoys. Their port ID sometimes isn't explicitly in 
    # the observation, but it is in the src for each decoy type 
    DECOY_TO_PORT = {
        'apache2': 80,
        'haraka': 25,
        'tomcat': 443,
        'vsftpd': 80 # Confirmed in src, but irl it's 20 and 21..
    }
    DECOYS = DECOY_TO_PORT.keys()

    def __init__(self):
        self.nids = NodeTracker()
        self.nodes = dict()
        self.subnet_to_router = dict()

        # Unchanging network topology
        self.permenant_edges = [[],[]]
        self.n_permenant_nodes = 0

        # Keep track of new connections to ports
        self.transient_edges = [[],[]]

        # Keep track of firewall rules
        self.subnet_connectivity = [[], []]

        # Keep track of which nodes are getting deleted when Remove is called
        self.host_to_sussy = defaultdict(list)

        # Keep track of number of servers/hosts on each subnet
        # (Faster accessor than using self.edge_index lookup)
        self.subnet_size = defaultdict(lambda : [0,0])

        # Keep track of which nodes are in which subnets
        self.subnet_masks = dict()

    def setup(self, initial_observation: dict):
        '''
        Needs to be called before ObservationGraph object can be used.
        Parses the initial state to create count of hosts in the network, 
        determine which nodes/edges are permenant parts of the topology, 
        and sets up initial connections running by default on hosts. 

        Args: 
            initial_observation: output of env.reset('Blue').observation
        '''
        # It's the 0th step. Ignore this key 
        succ = initial_observation.pop('success')

        # Set up mapping of IPs to hosts 
        self.ip_map = {
            val['Interface'][0]['ip_address']:host
            for host,val in initial_observation.items()
        }

        # Build network topology graph of subnets
        edges = self.parse_initial_observation(initial_observation)
        src,dst = zip(*edges)

        # Graph of subnets and default open ports doesn't change
        # but connections that we see in observations do. They're transient
        self.permenant_edges = [
            list(src) + list(dst),
            list(dst) + list(src)
        ]
        self.n_permenant_nodes = max(max(src), max(dst))+1

        # Set up masks so we can quickly get nodes relevant to each agent
        self._init_node_masks()

        # Gotta put it back in case other methods need the observation
        initial_observation['success'] = succ

    def set_firewall_rules(self, src, dst):
        '''
        Add edges between all nodes from src to dst 
        Assumes that src and dst are only routers or internet nodes
        (called from inside wrapper)
        '''
        src = [self.nids[s] for s in src]
        dst = [self.nids[d] for d in dst]

        self.subnet_connectivity = [src,dst]

    def _init_node_masks(self):
        '''
        Find all nodes that start with x_zone_subnet_
        Arrange nids in order: server, host, router, all other routers (alphabatized)
        '''
        self.subnet_masks = dict()

        all_sns = set()
        for name in self.nids.mapping.keys():
            if name.endswith('_router'):
                all_sns.add(name.replace('_router', ''))

        all_sns = list(all_sns)
        all_sns.sort()

        for sn in all_sns:
            srv = []
            usr = []
            rtr = []
            agent_controlled = []

            for i in range(6):
                n = self.nids.get(f'{sn}_server_host_{i}')
                if n is not None:
                    srv.append(n)
                else:
                    break

            for i in range(10):
                n = self.nids.get(f'{sn}_user_host_{i}')
                if n is not None:
                    usr.append(n)
                else:
                    break

            me = None
            for o_sn in all_sns:
                n = self.nids.get(f'{o_sn}_router')

                if sn == o_sn:
                    me = n
                else:
                    rtr.append(n)
                    if 'internet' not in o_sn and 'contractor' not in o_sn:
                        agent_controlled.append(n)

            self.subnet_masks[sn] = (
                srv,
                usr,
                [[me] * len(rtr), rtr],
                agent_controlled
            )
    def _remap_sn_mask(self, k, remap):
        '''
        Update masks so when state is reindexed, masks still point
        to the same nodes 
        '''
        (srv,usr,edge,rtrs) = self.subnet_masks[k]

        return (
            torch.tensor([remap[s] for s in srv]),
            torch.tensor([remap[u] for u in usr]),
            torch.tensor([
                [remap[e] for e in edge[0]],
                [remap[e] for e in edge[1]]
            ]),
            torch.tensor([remap[r] for r in rtrs])
        )

    def get_state(self, subnets):
        '''
        Get the current state of the graph, and masks for all hosts 
        in subnet(s) of interest 
        
        Args: 
            subnets: list of routers we want observations w.r.t (strings)
        '''

        # Cat all edges together
        ei = torch.tensor([
            self.permenant_edges[0] + self.transient_edges[0] + self.subnet_connectivity[0] + self.routers,
            self.permenant_edges[1] + self.transient_edges[1] + self.subnet_connectivity[1] + self.routers
        ])

        # Reindex so we use smallest possible feature vector 
        nids, ei = ei.unique(return_inverse=True)
        nid_map = {n.item():i for i,n in enumerate(nids)}

        # Create mapping from old nids to reindexed nids 
        nodes = [self.nodes[n.item()] for n in nids]
        ntypes = [self.NTYPES[type(n)] for n in nodes]

        # Add features for subnet membership
        rtr_map = {r:i for i,r in enumerate(self.routers)}
        transductive_routers = torch.zeros(nids.size(0), 9)

        x = torch.zeros(nids.size(0), self.DIM-9)
        for i,node in enumerate(nodes):
            # Get one-hot ntype feature
            ntype = ntypes[i]
            x[i][ntype] = 1.

            # Label which subnet it's in
            name = self.nids.id_to_str(nids[i].item())
            sn = name[:name.index('subnet') + 6] + '_router'
            sn = self.nids[sn]
            transductive_routers[i, rtr_map[sn]] = 1

            # Get multi-dim feature (if node has features)
            if node.dim:
                offset = self.OFFSETS[ntype]
                x[i][offset : offset + node.dim] = torch.from_numpy(node.get_features())

        x = torch.cat([x, transductive_routers], dim=1)

        # Remap masks s.t. we know which nodes we are interested in doing
        # actions upon 
        masks = [self._remap_sn_mask(sn,nid_map) for sn in subnets]
        return x,ei,masks


    def parse_initial_observation(self, obs):
        '''
        Converts from initial dictionary observation describing 
        entire environment into a graph. 

        Args: 
            obs: output of env.reset('Blue').observation
        '''
        edges = set()
        routers = []

        # Create all the router nodes
        # The way this loop is set up ensures that 
        # nodes are always added in the same order: 
        #   [subnet, servers, hosts]
        # For each subnet  
        for hostname, info in obs.items():
            if 'router' in hostname:
                nid = self.nids[hostname]
                routers.append(nid)
                self.nodes[nid] = SystemNode(nid, info, is_router=True)
                self.subnet_to_router[info['Interface'][0]['Subnet']] = nid
                continue

            # Internet is weird. Corner case delt with below
            if hostname == 'root_internet_host_0':
                continue

            nid = self.nids[hostname]

            if 'server' in hostname:
                is_server = True
            else:
                is_server = False

            self.nodes[nid] = SystemNode(nid, info['System info'], is_server)

            # Add edge from host to subnet router
            for sub in info['Interface']:
                sub = sub['Subnet']

                if (sub_id := self.subnet_to_router.get(sub)) is not None:
                    edges.add((nid, sub_id))

                    router = self.nids.id_to_str(sub_id)
                    if 'server' in hostname:
                        self.subnet_size[router][0] += 1
                    else:
                        self.subnet_size[router][1] += 1

                # Else it's from the internet, so we don't know about its subnet

        # CAGE4 is new in that hosts have ports that are always open
        # even after resetting. Calculate after, so node ordering is groups of
        # subnet, servers, hosts for easier combining w tabular observations
        for hostname, info in obs.items():
            if hostname == 'root_internet_host_0':
                continue

            nid = self.nids[hostname]
            for proc in info.get('Processes', []):
                # Again, represent processes only if they open ports.
                if len(conns := proc.get('Connections', [])):
                    conn = conns[0]
                    proc_str = f'{hostname}:{conn["local_port"]}'
                    pid = self.nids[proc_str]

                    # Add connection nodes to the graph, so they're marked
                    # as default services, but don't create an edge to them (yet). 
                    # Empirically, having edges from default procs to 
                    # host nodes doesn't help very much (at all?), and
                    # increases episode generation time by about 20s 
                    conn['process_name'] = proc['process_name']
                    conn['process_type'] = proc['process_type']
                    self.nodes[pid] = ConnectionNode(pid, conn, is_default=True)

        # Create special internet node
        nid = self.nids['internet_subnet_router']
        self.nodes[nid] = InternetNode(nid)
        routers.append(nid)
        self.routers = routers

        return edges

    def parse_observation(self, obs):
        '''
        Convert dictionary observation from environment into graph edits 
        Also, if the previous action was successful, modify the graph 
        if relevant. 

        Args: 
            obs: Raw observation from environment. 
        '''
        success = obs.pop('success')

        if 'action' in obs:
            act = obs.pop('action')
        else:
            act = None

        if isinstance(act, Restore) and success == TernaryEnum.TRUE:
            # Removes all files/sessions/connections from act.hostname
            # I.e. remove all transient edges involving act.hostname
            host_id = self.nids[act.hostname]

            # Need to remove host related edges, and any ports it may
            # have opened to talk to other nodes
            host_ports = [
                k for (k,v) in self.nids.inv_mapping.items()
                if v.startswith(act.hostname) and k != host_id
            ]

            removed = [host_id] + host_ports

            # Cycle through existing edges and remove any with 
            # restored host as src or dst 
            new_edges = [[],[]]
            for i in range(len(self.transient_edges[0])):
                src = self.transient_edges[0][i]
                dst = self.transient_edges[1][i]

                if src not in removed and dst not in removed:
                    new_edges[0].append(src)
                    new_edges[1].append(dst)

            # Remove restored host from list of suspicious machines
            self.transient_edges = new_edges
            if act.hostname in self.host_to_sussy:
                self.host_to_sussy.pop(act.hostname)

        elif isinstance(act, Remove) and success == TernaryEnum.TRUE:
            # Removes all suspicious sessions from act.hostname
            if act.hostname in self.host_to_sussy:
                sus_ids = self.host_to_sussy.pop()
            else:
                sus_ids = []

            if sus_ids:
                new_edges = [[],[]]
                for i in range(len(self.transient_edges[0])):
                    if  (src := self.transient_edges[0][i]) not in sus_ids and \
                        (dst := self.transient_edges[1][i]) not in sus_ids:

                        new_edges[0].append(src)
                        new_edges[1].append(dst)

                self.transient_edges = new_edges

        elif isinstance(act, DeployDecoy) and success == TernaryEnum.TRUE:
            # Have to parse out which decoy was selected using the
            # observation of which process just started. Hopefully
            # this doesn't cause collisions
            proc = obs[act.hostname]['Processes'][0]
            service = proc['service_name']
            port_num = self.DECOY_TO_PORT[service]

            # Add new process (e.g. port) to act.hostname
            host_id = self.nids[act.hostname]
            port_id = self.nids[f'{act.hostname}:{port_num}']

            # Add edge from port -> host representing external communication
            # being allowed to enter the host through this node
            self.nodes[port_id] = init_decoy(port_id, service)
            self.transient_edges[0].append(port_id)
            self.transient_edges[1].append(host_id)


        edges = set()
        for hostname,info in obs.items():
            host_id = self.nids[hostname]

            # Observation for Monitor, Sleep, or sometimes just passively given regardless
            if (procs := info.get('Processes')):
                for proc in procs:
                    conn = proc.get('Connections')
                    if conn is None:
                        continue

                    if len(conn) > 1:
                        print("Wtf this shouldn't happen")
                        pprint(proc)
                        raise ValueError()

                    conn = conn[0]

                    # Get connection direction 
                    local_addr = conn.get('local_address')
                    local_port = conn.get('local_port')
                    remote_addr = conn.get('remote_address')
                    remote_port = conn.get('remote_port')

                    # Just a new process starting, not a remote event.
                    if not (local_addr and remote_addr):
                        continue

                    # Figure out which hosts these IPs belong to 
                    local_host = self.ip_map[local_addr]
                    lh_id = self.nids[local_host]
                    remote_host = self.ip_map[remote_addr]
                    rh_id = self.nids[remote_host]

                    # Add local and remote connection nodes
                    # If local/remote IPs given, add edges from
                    # connection to those nodes 
                    if local_port:
                        lp_name = f'{local_host}:{local_port}'
                        lp_id = self.nids[lp_name]

                        if local_port > 49152 or self.nodes.get(lp_id) is None:
                            # Just make a new node
                            lp_node = ConnectionNode(lp_id, is_ephemeral=local_port > 49152)
                            self.nodes[lp_id] = lp_node

                        self.transient_edges[0] += [rh_id, lp_id]
                        self.transient_edges[1] += [lp_id, lh_id]

                    if remote_port:
                        rp_name = f'{remote_host}:{remote_port}'
                        rp_id = self.nids[rp_name]

                        if remote_port > 49152 or self.nodes.get(rp_id) is None:
                            rp_node = ConnectionNode(rp_id, is_ephemeral=remote_port > 49152)
                            self.nodes[rp_id] = rp_node

                        self.transient_edges[0] += [lh_id, rp_id]
                        self.transient_edges[1] += [rp_id, rh_id]

                    # Seems like this only happens if proc is suspicious?
                    if 'PID' in proc:
                        self.host_to_sussy[host_id].append(port_id)


            # Observation corresponding to 'Analyse'
            if (files := info.get('Files')):
                for file in files:
                    file_uq_str = f"{hostname}:{file['Path']}\\{file['File Name']}"
                    file_id = self.nids[file_uq_str]
                    self.nodes[file_id] = FileNode(file_id, file)

                    edges.update([
                        (host_id, file_id),
                        (file_id, host_id)
                    ])

        if edges:
            src,dst = zip(*edges)
            self.transient_edges[0] += src
            self.transient_edges[1] += dst

        # Need to put it back in the dict now that we're done with it
        obs['success'] = success