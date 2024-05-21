from abc import ABC, abstractmethod
from collections.abc import Iterable
from collections import OrderedDict

import numpy as np

import CybORG.Shared.Enums as Enums

class Node(ABC, object):
    '''
    Ordered dict of features. Keys match CybORG output
    values are initialized to None for enums, -1 for scalars
    '''
    feats: OrderedDict

    '''
    The dimension of each of the features defined in feats
    E.g. if feats is [BooleanEnum, scaler]
    dims would be [2, 1]
    '''
    dims: Iterable[int]


    labels: list

    '''
    Defining __eq__ and __hash__ as so allows us to create a set
    of nodes to avoid duplicates later on
    '''
    def __eq__(self, other):
        if isinstance(other, Node):
            return self.uuid == other.uuid
        else:
            return False

    def __hash__(self):
        return self.uuid

    def __init__(self, uuid: int, observation: dict):
        self.uuid = uuid
        self.parse_observation(observation)


    def parse_observation(self, obs: dict) -> None:
        '''
        Given the dictionary from an observation, update/set
        features (perhaps using the set_features method)

        By default, only pulls out what can be given in an
        observation, but we have some node types with additional
        features. This method should be extended to accomidate
        '''
        for k in self.feats.keys():
            if k in obs:
                self.feats[k] = obs[k]

    def get_features(self) -> np.array:
        '''
        Convert from all the enums to a fixed size vector

        Requires the following fields to be initialized:
            self.dim: The output dimension of the feature vector
            self.dims: The output dimensions of individual one-hot features (sum(dims) == dim)
            self.feats: An ordered dict of the features we want
        '''
        out = np.zeros(self.dim)

        offset = 0
        for i,feat in enumerate(self.feats.values()):
            # Enums are 1-indexed
            if feat:
                if isinstance(feat, Iterable):
                    for f in feat:
                        out[offset + (f.value-1)] = 1
                elif isinstance(feat, float):
                    out[offset] = feat
                elif isinstance(feat, bool):
                    out[offset] = float(feat)
                else:
                    out[offset + (feat.value-1)] = 1

            offset += self.dims[i]

        return out

    def human_readable(self):
        human_readable = []
        for i,feat_str in enumerate(self.feats.keys()):
            if self.dims[i] > 1:
                human_readable += [f'{feat_str}-{j}' for j in range(self.dims[i])]
            else:
                human_readable.append(feat_str)

        return human_readable

class SystemNode(Node):
    '''
    Node representing computers (servers, users, and routers)
    '''
    def __init__(self, uuid: int, observation: dict, is_server=False, is_router=False, crown_jewel=False):
        self.feats = OrderedDict(
            Architecture = None,
            OSDistro = None,
            OSType = None,
            OSVersion = None,
            OSKernelVersion = None,
            os_patches = [],
            crown_jewel=float(crown_jewel),
            user=float(not is_server),
            server=float(is_server),
            router=float(is_router)
        )

        self.dims = [
            len(Enums.Architecture.__members__),
            len(Enums.OperatingSystemDistribution.__members__),
            len(Enums.OperatingSystemType.__members__),
            len(Enums.OperatingSystemVersion.__members__),
            len(Enums.OperatingSystemKernelVersion.__members__),
            len(Enums.OperatingSystemPatch.__members__),
            1,1,1,1
        ]
        self.dim = sum(self.dims)
        super().__init__(uuid, observation)

    @property
    def labels(self):
        return (
            list(Enums.Architecture.__members__.items()) +
            list(Enums.OperatingSystemDistribution.__members__.items()) +
            list(Enums.OperatingSystemType.__members__.items()) +
            list(Enums.OperatingSystemVersion.__members__.items()) +
            list(Enums.OperatingSystemKernelVersion.__members__.items()) +
            list(Enums.OperatingSystemPatch.__members__.items()) +
            ["crown_jewel", "user", "server", "router"]
        )

class ConnectionNode(Node):
    '''
    Node representing processes that communicate with other hosts 
    '''
    def __init__(self, uuid: int, observation: dict=dict(), suspicious_pid: bool=False, is_decoy: bool=False, is_default=False, is_ephemeral=False):
        self.feats = OrderedDict(
            process_name = None,
            process_type = None,
            suspicious_pid=float(suspicious_pid),
            is_decoy=float(is_decoy),
            is_default=float(is_default),
            is_ephemeral=float(is_ephemeral)
        )
        self.dims = [
            len(Enums.ProcessName.__members__),
            len(Enums.ProcessType.__members__),
            1,1,1,1
        ]
        self.dim = sum(self.dims)
        super().__init__(uuid, observation)

    @property
    def labels(self):
        return (
            list(Enums.ProcessName.__members__.items()) +
            list(Enums.ProcessType.__members__.items()) +
            ['suspicious pid', 'is decoy', 'is default', 'is_ephemeral']
        )

def init_decoy(uuid, dtype):
    '''
    ConnectionNode factory. Creates the relevant kind of decoy 
    Args: 
        uuid: unique identifier (int)
        dtype: what kind of decoy we're making (str)
    '''
    d = {
        'apache2': dict(
            process_name=Enums.ProcessName.APACHE2,
            process_type=Enums.ProcessType.WEBSERVER
        ),
        'tomcat': dict(
            # For some reason, listed under proc version enum but not proc name
            process_name=Enums.ProcessName.UNKNOWN,
            process_type=Enums.ProcessType.WEBSERVER
        ),
        'vsftpd': dict(
            # Also not in the enum
            process_name=Enums.ProcessName.UNKNOWN, 
            process_type=Enums.ProcessType.WEBSERVER
        ),
        'haraka': dict(
            # Also in ProcessVersion but not ProcessName
            process_name=Enums.ProcessName.UNKNOWN,
            process_type=Enums.ProcessType.SMTP
        )
    }

    return ConnectionNode(uuid, d[dtype], is_decoy=True)

class InternetNode(Node):
    '''
    No features. Purely a structural node
    '''
    def __init__(self, uuid: int, observation: dict=dict()):
        self.feats = OrderedDict()
        self.dims = []
        self.dim = 0

        super().__init__(uuid, observation)

    @property
    def labels(self):
        return []

class FileNode(Node):
    '''
    Nodes representing files on workstations. 
    These are almost always malicious. 
    '''
    def __init__(self, uuid: int, observation: dict, is_new=True):
        self.feats = OrderedDict(
            [
                (s, None) for s in [
                    'Known File',
                    'Known Path',
                    'User Permissions',
                    'Group Permissions',
                    'Default Permissions'
                ]
            ],
            Version=None,
            Type=None,
            Vendor=None,

            # Additional static features
            is_new=float(is_new),

            # Will be updated if observed later
            Density= -1.,
            Signed= -1.
        )

        self.dims = [
            len(Enums.FileType.__members__),
            len(Enums.Path.__members__),
            8,8,8, # Permissions groups
            len(Enums.FileVersion.__members__),
            len(Enums.FileType.__members__),
            len(Enums.Vendor.__members__),
            1,
            1,1
        ]
        self.dim = sum(self.dims)
        super().__init__(uuid, observation)

    @property
    def labels(self):
        return (
            list(Enums.FileType.__members__.items()) +
            list(Enums.Path.__members__.items()) +
            ['permissions']*24 +
            list(Enums.FileVersion.__members__.items()) +
            list(Enums.FileType.__members__.items()) +
            list(Enums.Vendor.__members__.items()) +
            ['is new', 'density', 'signed']
        )