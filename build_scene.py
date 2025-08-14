#!/usr/bin/env python3
"""
Convert results/static_universe.json → results/substrate.glb
(no more stray “.Object.values” bugs)
"""
import json, struct, pathlib, numpy as np
from pygltflib import GLTF2, Scene, Node, Mesh, Primitive, Material, Buffer, \
                      BufferView, Accessor, Asset, UNSIGNED_INT, FLOAT

ROOT=pathlib.Path(__file__).resolve().parent
SRC =ROOT/"results/static_universe.json"
DST =ROOT/"results/substrate.glb"

def pack(arr:np.ndarray)->bytes:
    return struct.pack("<"+"f"*arr.size,*arr.flatten())

def new_accessor(gltf,bv,off,count,ctype,dtype=FLOAT,minv=None,maxv=None):
    acc=Accessor(bufferView=bv,byteOffset=off,componentType=dtype,
                 count=count,type=ctype,min=minv,max=maxv)
    gltf.accessors.append(acc); return len(gltf.accessors)-1

data=json.loads(SRC.read_text())
node_ids=sorted(map(int,data["nodes"]))
pos=np.array([data["nodes"][str(n)]["position"][:3] for n in node_ids],np.float32)

id_map={nid:i for i,nid in enumerate(node_ids)}
edges=np.array([[id_map[u],id_map[v]] for u,v in data["edges"]],np.uint32).ravel()

# ── build glTF ───────────────────────────────────────────────────────────
g=GLTF2(asset=Asset(version="2.0"),
        scenes=[Scene(nodes=[0])],
        materials=[Material(name="edge",alphaMode="BLEND",
                            pbrMetallicRoughness={"baseColorFactor":[.2,.5,1,.25]}),
                   Material(name="node",alphaMode="BLEND",
                            pbrMetallicRoughness={"baseColorFactor":[.2,.8,1,.5]})])

buf = pack(pos) + edges.tobytes()
g.buffers.append(Buffer(byteLength=len(buf)))

v_pos=len(g.bufferViews); g.bufferViews.append(BufferView(buffer=0,byteOffset=0, byteLength=pos.nbytes))
v_idx=len(g.bufferViews); g.bufferViews.append(BufferView(buffer=0,byteOffset=pos.nbytes,byteLength=edges.nbytes))

a_pos=new_accessor(g,v_pos,0,len(pos),"VEC3",minv=pos.min(0).tolist(),maxv=pos.max(0).tolist())
a_idx=new_accessor(g,v_idx,0,len(edges),"SCALAR",dtype=UNSIGNED_INT)

g.meshes.append(Mesh(primitives=[
        Primitive(attributes={"POSITION":a_pos},indices=a_idx,mode=1,material=0),
        Primitive(attributes={"POSITION":a_pos},mode=0,material=1)]))
g.nodes.append(Node(mesh=0,name="substrate"))

g.set_binary_blob(buf); g.save_binary(DST)
print(f"✓ wrote {len(node_ids)} nodes / {len(edges)//2} edges → {DST}")
