# KEEP: A GNN-based PPO Model for MARL 

## Graph Schema

<p align="center">
<img src="img/schema.png" alt="Graph schema figure" width=400/>
<br>
<h3 align="center">Figure 1: Graph schema diagram</h3>
</p>

The observation graph parses the dictionaries provided by CybORG into a graph. We track 5 kinds of entities: Hosts (users and servers), Routers, Open Ports, Files, and the internet. Figure 1 shows how these entities interrelate. Hosts communicate with ports, own files, and are members of a subnet managed by a router. Routers communicate with other routers, and communicate with the internet. 

Taking actions in the environment may cause new edges or nodes to be added to the graph. Below is a table of the actions and observations that may cause this to occur.

<h3 align="center">Table 1: Graph edits resulting from actions</h3>

| **Event**         | **Effect**                                                                                                                                                      |
|-------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `Restore`         | Remove all edges to/from files and (non-default) connections into the restored node.                                                                            |
| `DeployDecoy`     | Create a new connection node and add an edge to/from the host that deployed it.                                                                                 |
| `AllowConnection` | Create an edge between the two subnets.                                                                                                                         |
| `BlockConnection` | Delete the edge between the two subnets.                                                                                                                        |
| `Analyze`         | Add a file node, and an edge to the host analyzed if one was found.                                                                                             |
| `Monitor`         | Add any new connection nodes (processes with port information) observed, and create edges between them and the hosts they occur on, as well as the source host. |

## Node Features

The table below lists all features tracked by the observation graph, and the `GraphWrapper` class. 

<h3 align="center">Table 2: Features tracked in the observation graph</h3>

| **Node Type** | **Feature**         | **Description**                                                                                                                      |
|---------------|---------------------|--------------------------------------------------------------------------------------------------------------------------------------|
|   **Hosts/Routers**   | CybORG Enums        | One-hot vectors for OS Version, distro, arch, etc. provided in observation dictionaries                                              |
|               | `isUser`            | If this node is a PC                                                                                                                 |
|               | `isServer`          | If this node is a server                                                                                                             |
|               | `isRouter`          | If this node is a router                                                                                                             |
|   **Ports**   | CybORG Enums        | Port number, process name, process type, etc. provided in observation dictionaries                                                   |
|               | `isEphemeral`       | If this port is ephemeral (gt;49152)                                                                                                 |
|               | `isDefault`         | If this port was seen in the initial observation, and is a service that runs on this machine normally                                |
|               | `isDecoy`           | If this port is a decoy process                                                                                                      |
|   **Files**   | CybORG Enums        | Version, type, vendor, etc. provided in observation dictionaries                                                                     |
|               | Density             | When a file is `Analyze`d, a density value is provided                                                                               |
|               | Signed              | When a file is `Analyze`d, a boolean for if it was signed is provided                                                                |
| **Internet**  | None                | This is a purely structural node type to connect subnets together if they have internet connection                                   |
|    **All**    | Subnet Membership   | One-hot vector denoting which subnet a node belongs to                                                                               |
|               | Node type           | One-hot vector denoting node type (Host, Port, File, or Internet)                                                                    |
|               | Tabular Observation | The observation provided by the `EnterpriseMAE` wrapper is parsed, and the information is concatenated with the relevant host nodes  |
|               | Messages | Any messages recieved from the other agents. Message features are concatenated to the features of the subnets monitored by the agent that sent them. |

Agents also attempt to send each other messages, if the communication policy for the particular phase allows. These messages are only allowed to consist of 8 bits of information. We use these messages to share the information from the `EnterpriseMAE` wrapper that agents recieved about the machines they monitor. For each subnet an agent monitors, it adds 2 bits of information to represent if *any* host on that subnet is comprimised, or has been scanned. We add an additional bit at the end of the message to act as a checkbit, such that if an agent sends 0's, they mean "no comprimise" if the checkbit is present, and, "agent cannot communicate" if it is not. 

## Agent Architecture 

<p align="center">
<img src="img/global_node_arch.png" alt="Graph schema figure"/>
<br>
<h3 align="center">Figure 2: Agent architecture diagram</h3>
</p>