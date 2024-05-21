N_AGENTS = 5
ROUTERS = [
    'admin_network_subnet', 
    'contractor_network_subnet', 
    'internet_subnet', 
    'office_network_subnet', 
    'operational_zone_a_subnet', 
    'operational_zone_b_subnet', 
    'public_access_zone_subnet', 
    'restricted_zone_a_subnet', 
    'restricted_zone_b_subnet'
]
ROUTERS = [r+'_router' for r in ROUTERS]
INTERNET = 2 # Special in that not really a subnet 

# How many bits are used in each subnet feature block
SN_BLOCK_SIZE = 9*3 + 16*2 # 59

# Who can talk to whom without an internet connection 
ACCESSABLE_OFFLINE = {
    'admin_network_subnet_router': ['admin_network_subnet', 'office_network_subnet', 'public_access_zone_subnet'], 
    'contractor_network_subnet_router': ['contractor_network_subnet'], 
    'internet_subnet_router': [], 
    'office_network_subnet_router': ['admin_network_subnet', 'office_network_subnet', 'public_access_zone_subnet'], 
    'operational_zone_a_subnet_router': ['operational_zone_a_subnet', 'restricted_zone_a_subnet'], 
    'operational_zone_b_subnet_router': ['operational_zone_b_subnet', 'restricted_zone_b_subnet'], 
    'public_access_zone_subnet_router': ['admin_network_subnet', 'office_network_subnet', 'public_access_zone_subnet'], 
    'restricted_zone_a_subnet_router': ['operational_zone_a_subnet', 'restricted_zone_a_subnet'], 
    'restricted_zone_b_subnet_router': ['operational_zone_b_subnet', 'restricted_zone_b_subnet']
}

MY_SUBNETS = {
    0: ['restricted_zone_a_subnet'],
    1: ['operational_zone_a_subnet'],
    2: ['restricted_zone_b_subnet'],
    3: ['operational_zone_b_subnet'],
    4: ['admin_network_subnet', 'office_network_subnet', 'public_access_zone_subnet']
}

MAX_SERVERS = 6
MAX_USERS = 10 
MAX_HOSTS = 16
POSSIBLE_NEIGHBORS = 8

from CybORG.Simulator.Actions import Analyse, Remove, Restore, DeployDecoy, Monitor
from CybORG.Simulator.Actions.ConcreteActions.ControlTraffic import BlockTrafficZone, AllowTrafficZone
NODE_ACTIONS = [Analyse, Remove, Restore, DeployDecoy]
EDGE_ACTIONS = [AllowTrafficZone, BlockTrafficZone]
GLOBAL_ACTIONS = [Monitor]

N_NODE_ACTIONS = len(NODE_ACTIONS)
N_EDGE_ACTIONS = len(EDGE_ACTIONS)
N_GLOBAL_ACTIONS = len(GLOBAL_ACTIONS) 
MAX_ACTIONS = \
    N_NODE_ACTIONS*(MAX_HOSTS) + \
    N_EDGE_ACTIONS*(len(ROUTERS)-1) + \
    N_GLOBAL_ACTIONS