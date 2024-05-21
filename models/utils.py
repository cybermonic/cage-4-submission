import torch 

def combine_subgraphs(states):
    xs,eis = zip(*states)
    
    # ei we need to update each node idx to be
    # ei[i] += len(ei[i-1])
    offset=0
    new_eis=[]
    for i in range(len(eis)):
        new_eis.append(eis[i]+offset)
        offset += xs[i].size(0)

    # X is easy, just cat
    xs = torch.cat(xs, dim=0)
    eis = torch.cat(new_eis, dim=1)

    return xs,eis

def combine_marl_states(s):
    '''
    Combines states given observations of the form:
    x, ei, servers, n_servers, users, n_users, action_edges, is_multi_subnet

    (Note: is_multi_subnet values need to be separated)
    '''
    xs, eis, gvs, srvs, nsrvs, usrs, nusrs, edges, is_multi = [list(element) for element in zip(*s)]

    # Same as utils.combine_subgraphs
    offset=0
    new_ei = []
    new_srv = []
    new_usr = []
    new_edges = []

    for i in range(len(eis)):
        # Need to make them copies
        new_edges.append(edges[i] + offset)
        new_srv.append(srvs[i] + offset)
        new_usr.append(usrs[i] + offset)
        new_ei.append(eis[i] + offset)

        offset += xs[i].size(0)

    # X is easy, just cat
    xs = torch.cat(xs, dim=0)
    gvs = torch.cat(gvs, dim=0)
    srvs = torch.cat(new_srv)
    nsrvs = torch.cat(nsrvs)
    usrs = torch.cat(new_usr)
    nusrs = torch.cat(nusrs)
    eis = torch.cat(new_ei, dim=1)
    edges = torch.cat(new_edges, dim=1)

    # Is_Multi should be the same for all elements
    return xs,eis,gvs, srvs,nsrvs, usrs,nusrs, edges, is_multi[0]