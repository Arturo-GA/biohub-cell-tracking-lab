import numpy as np
from biohub_lab.segmentation_tracklets import recover
def test_preserves_control_excludes_duplicates_and_isolated_candidates():
    nodes={i:dict(t=i,z=0,y=0,x=0) for i in range(3)};edges=[(0,1),(1,2)]
    frames={0:np.array([[0,0,1],[0,0,30],[30,40,40]]),1:np.array([[0,0,31]])}
    new,links,report=recover(nodes,edges,frames,(3,64,64,64))
    assert all(new[k]==v for k,v in nodes.items()) and set(edges)<=set(links)
    assert report['added_nodes']==2 and report['added_edges']==1
    assert new[3]['x']==30 and new[4]['x']==31 and (3,4) in links
def test_links_are_one_to_one_and_do_not_cross_missing_frames():
    frames={0:np.array([[0,0,20],[0,0,22]]),1:np.array([[0,0,21]]),3:np.array([[0,0,21]])}
    nodes,edges,report=recover({},[],frames,(4,64,64,64))
    assert report['added_nodes']==2 and len(edges)==1
    assert sorted(v['t'] for v in nodes.values())==[0,1]
