
# quotient_space_topology.py — Quotient space representations of topologies via gluing fundamental polygons
# 
# This module constructs topological spaces by gluing edges of fundamental polygons (squares).
# It implements quotient space representations for various topologies including:
# - Torus, Klein bottle, Möbius strip (via edge gluing)
# - Sphere, sphere_two (via point/boundary identification)
# - Cylinders, hemispheres, and projective plane
#
# Features:
# - Robust sphere_two implementation (atan2 + tiny-grid eps)
# - Fixed corner wrapping for edge cases
# - Hemisphere visualization support
# - Surface edges rendering
# - Glue display for sphere_two
# - Labeled adjacency matrix export

import os, argparse
from typing import Iterable, Tuple, List, Union
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

Node = Union[int, str]

# ================= helpers =================
def lid(i:int,j:int,W:int)->int: return i*W+j
def neighbors(i:int,j:int, kind:int)->Iterable[Tuple[int,int]]:
    base=[(i-1,j),(i+1,j),(i,j-1),(i,j+1)]
    return base if kind==4 else base+[(i-1,j-1),(i-1,j+1),(i+1,j-1),(i+1,j+1)]

# 边界环（无角点重复，双射）
def boundary_index(i:int, j:int, H:int, W:int) -> int:
    if i == 0:                       return j
    elif j == W-1 and 1 <= i <= H-2: return W + (i - 1)
    elif i == H-1:                   return W + (H - 2) + (W - 1 - j)
    elif j == 0 and 1 <= i <= H-2:   return W + (H - 2) + W + (H - 2 - i)
    else: raise ValueError("not on boundary")

def boundary_coords(t:int, H:int, W:int) -> Tuple[int,int]:
    P = 2*(H + W) - 4
    if P <= 0: raise ValueError("invalid grid size")
    t %= P
    if t < W: return 0, t
    t -= W
    if t < (H - 2): return 1 + t, W - 1
    t -= (H - 2)
    if t < W: return H - 1, W - 1 - t
    t -= W
    return (H - 2 - t), 0

# wrap 规则（修复角落双越界：同时处理 x/y，并做兜底）
def apply_wrap(i:int,j:int,H:int,W:int, topo:str)->Tuple[int,int,bool]:
    if 0<=i<H and 0<=j<W: return i,j,True
    if topo in ("plane","sphere","sphere_two","hemisphere_n","hemisphere_s"): return i,j,False

    def wrap_x(ii,jj):
        if jj < 0:
            if topo in ("cylinder_x","torus","klein_y"): return ii, W-1
            elif topo in ("mobius_x","klein_x","proj_plane"): return (H-1-ii, W-1)
        if jj >= W:
            if topo in ("cylinder_x","torus","klein_y"): return ii, 0
            elif topo in ("mobius_x","klein_x","proj_plane"): return (H-1-ii, 0)
        return None

    def wrap_y(ii,jj):
        if ii < 0:
            if topo in ("cylinder_y","torus","klein_x"): return H-1, jj
            elif topo in ("mobius_y","klein_y","proj_plane"): return H-1, (W-1-jj)
        if ii >= H:
            if topo in ("cylinder_y","torus","klein_x"): return 0, jj
            elif topo in ("mobius_y","klein_y","proj_plane"): return 0, (W-1-jj)
        return None

    ii, jj = i, j
    if jj < 0 or jj >= W:
        m = wrap_x(ii, jj)
        if m is None: return ii, jj, False
        ii, jj = m
    if ii < 0 or ii >= H:
        m = wrap_y(ii, jj)
        if m is None: return ii, jj, False
        ii, jj = m

    # Klein/Möbius 的翻折可能再次把另一维推越界，兜底再包一次
    if not (0 <= ii < H and 0 <= jj < W):
        if jj < 0 or jj >= W:
            m = wrap_x(ii, jj)
            if m is not None: ii, jj = m
        if ii < 0 or ii >= H:
            m = wrap_y(ii, jj)
            if m is not None: ii, jj = m

    return (ii, jj, True) if (0 <= ii < H and 0 <= jj < W) else (ii, jj, False)

# ================= adjacency =================
def add(A, idx, u:Node, v:Node):
    if u==v: return
    A[idx[u], idx[v]] = 1

def add_bi(A, idx, u:Node, v:Node, undirected:bool):
    """单向加边；若 undirected=True，再加反向边。"""
    if u==v: return
    A[idx[u], idx[v]] = 1
    if undirected:
        A[idx[v], idx[u]] = 1

def build_adj(H:int,W:int, topo:str, neigh:int=4, undirected:bool=True):
    """
    返回 A, nodes, mapdf
    sphere      : 单点折叠
    sphere_two  : 双半球反向粘合
    hemisphere_*: 半球（开边界，不粘合）
    其它        : 依 wrap 规则
    """
    # --- sphere_two ---
    if topo == "sphere_two":
        Nlayer = H*W
        nodes: List[Node] = [("A",i,j) for i in range(H) for j in range(W)] + \
                            [("B",i,j) for i in range(H) for j in range(W)]
        idx = {n:k for k,n in enumerate(nodes)}
        A = np.zeros((2*Nlayer, 2*Nlayer), dtype=np.int8)

        for layer in ("A","B"):
            for i in range(H):
                for j in range(W):
                    u = (layer,i,j)
                    for (ni,nj) in neighbors(i,j,neigh):
                        if 0<=ni<H and 0<=nj<W:
                            v = (layer,ni,nj)
                            add_bi(A, idx, u, v, undirected)
                        else:
                            # 反向粘合到另一层的对应边界点
                            bi = min(max(ni, 0), H - 1)
                            bj = min(max(nj, 0), W - 1)
                            if 0 < bi < H - 1 and 0 < bj < W - 1:
                                if ni < 0: bi = 0
                                elif ni >= H: bi = H - 1
                                if nj < 0: bj = 0
                                elif nj >= W: bj = W - 1
                            t  = boundary_index(bi, bj, H, W)
                            P  = 2*(H + W) - 4
                            t2 = (P - t) % P
                            bi2, bj2 = boundary_coords(t2, H, W)
                            other = "B" if layer=="A" else "A"
                            v = (other, bi2, bj2)
                            add_bi(A, idx, u, v, undirected)

        # mapdf：与 labeled 邻接矩阵统一的 node_id
        # 使用统一编号 0 到 2*H*W-1，而不是 "A:xxx" 和 "B:xxx"
        mapping = [(k, k, i, j, layer) for k,(layer,i,j) in enumerate(nodes)]
        mapdf = pd.DataFrame(mapping, columns=["rowcol_index","node_id","i","j","layer"])
        return A, nodes, mapdf

    # --- sphere：单点折叠 ---
    if topo == "sphere":
        def is_boundary(i,j): return (i==0 or i==H-1 or j==0 or j==W-1)
        interior = [(i,j) for i in range(1,H-1) for j in range(1,W-1)]
        nodes: List[Node] = [lid(i,j,W) for (i,j) in interior] + ["S"]
        idx={n:k for k,n in enumerate(nodes)}
        A = np.zeros((len(nodes),len(nodes)), dtype=np.int8); S="S"
        for (i,j) in interior:
            u=lid(i,j,W)
            for (ni,nj) in neighbors(i,j,neigh):
                if not (0<=ni<H and 0<=nj<W) or is_boundary(ni,nj):
                    add(A,idx,u,"S")
                else:
                    add(A,idx,u,lid(ni,nj,W))
        if undirected: A=((A+A.T)>0).astype(np.int8)
        mapping=[(idx[lid(i,j,W)], lid(i,j,W), i, j, "int") for (i,j) in interior]
        mapping.append((idx["S"], -1, -1, "S"))
        mapdf=pd.DataFrame(mapping, columns=["rowcol_index","node_id","i","j","layer"])
        return A, nodes, mapdf

    # --- hemisphere_*：开边界（不 wrap，经度接缝会有"缺口"） ---
    if topo in ("hemisphere_n","hemisphere_s"):
        nodes=[n for n in range(H*W)]; idx={n:k for k,n in enumerate(nodes)}
        A=np.zeros((len(nodes),len(nodes)), dtype=np.int8)
        for i in range(H):
            for j in range(W):
                u=lid(i,j,W)
                for (ni,nj) in neighbors(i,j,neigh):
                    if 0<=ni<H and 0<=nj<W:
                        add_bi(A, idx, u, lid(ni,nj,W), undirected)
        mapping=[(idx[lid(i,j,W)], lid(i,j,W), i, j, topo) for i in range(H) for j in range(W)]
        mapdf=pd.DataFrame(mapping, columns=["rowcol_index","node_id","i","j","layer"])
        return A, nodes, mapdf

    # --- other topologies ---
    nodes=[n for n in range(H*W)]; idx={n:k for k,n in enumerate(nodes)}
    A=np.zeros((len(nodes),len(nodes)), dtype=np.int8)
    for i in range(H):
        for j in range(W):
            u=lid(i,j,W)
            for (ni,nj) in neighbors(i,j,neigh):
                ii,jj,ok=apply_wrap(ni,nj,H,W,topo)
                if not ok: continue
                add_bi(A, idx, u, lid(ii,jj,W), undirected)
    mapping=[(idx[lid(i,j,W)], lid(i,j,W), i, j, "single") for i in range(H) for j in range(W)]
    mapdf=pd.DataFrame(mapping, columns=["rowcol_index","node_id","i","j","layer"])
    return A, nodes, mapdf

# ================= coords =================
def coords_for_topo(H:int,W:int, topo:str)->np.ndarray:
    u = np.linspace(0,1,H,endpoint=False)
    v = np.linspace(0,1,W,endpoint=False)
    U,V = np.meshgrid(u,v,indexing="ij")

    if topo == "plane":
        return np.stack([V, -U, np.zeros_like(U)], axis=2).reshape(-1,3)
    if topo == "cylinder_x":
        ang=2*np.pi*V; R=1.0
        return np.stack([R*np.cos(ang), R*np.sin(ang), U], axis=2).reshape(-1,3)
    if topo == "cylinder_y":
        ang=2*np.pi*U; R=1.0
        return np.stack([R*np.cos(ang), R*np.sin(ang), V], axis=2).reshape(-1,3)
    if topo == "mobius_x":
        ang=2*np.pi*V; w0=0.3; w=(U*2-1)*w0
        x=(1+w*np.cos(ang/2))*np.cos(ang); y=(1+w*np.cos(ang/2))*np.sin(ang); z=w*np.sin(ang/2)
        return np.stack([x,y,z],axis=2).reshape(-1,3)
    if topo == "mobius_y":
        ang=2*np.pi*U; w0=0.3; w=(V*2-1)*w0
        x=(1+w*np.cos(ang/2))*np.cos(ang); y=(1+w*np.cos(ang/2))*np.sin(ang); z=w*np.sin(ang/2)
        return np.stack([x,y,z],axis=2).reshape(-1,3)
    if topo == "torus":
        theta=2*np.pi*U; phi=2*np.pi*V; R=2.0; r=0.8
        x=(R+r*np.cos(theta))*np.cos(phi); y=(R+r*np.cos(theta))*np.sin(phi); z=r*np.sin(theta)
        return np.stack([x,y,z],axis=2).reshape(-1,3)
    if topo in ("klein_x","klein_y"):
        if topo=="klein_x": Uang=2*np.pi*U; Vang=2*np.pi*V
        else:               Uang=2*np.pi*V; Vang=2*np.pi*U
        a=2.0
        x=(a+np.cos(Uang/2)*np.sin(Vang)-np.sin(Uang/2)*np.sin(2*Vang))*np.cos(Uang)
        y=(a+np.cos(Uang/2)*np.sin(Vang)-np.sin(Uang/2)*np.sin(2*Vang))*np.sin(Uang)
        z= np.sin(Uang/2)*np.sin(Vang)+np.cos(Uang/2)*np.sin(2*Vang)
        return np.stack([x,y,z],axis=2).reshape(-1,3)
    if topo == "proj_plane":
        ucap=np.pi*U; vcap=np.pi*V
        x=np.sin(2*ucap)*np.sin(vcap); y=np.sin(2*vcap)*np.sin(ucap); z=np.cos(ucap)*np.cos(vcap)
        return np.stack([x,y,z],axis=2).reshape(-1,3)
    if topo == "sphere":
        eps=1e-6
        ui=np.linspace(eps,1-eps,max(H-2,0),endpoint=True)
        vi=np.linspace(0,1,max(W-2,0),endpoint=False)
        if len(ui)==0 or len(vi)==0:
            return np.array([[0.0,0.0,1.05]])  # 极小尺寸退化
        Ui,Vi=np.meshgrid(ui,vi,indexing="ij")
        lat=(Ui-0.5)*np.pi; lon=2*np.pi*Vi; R=1.0
        x=R*np.cos(lat)*np.cos(lon); y=R*np.cos(lat)*np.sin(lon); z=R*np.sin(lat)
        X=np.stack([x,y,z],axis=2).reshape(-1,3)
        S=np.array([[0.0,0.0,R+0.05]])
        return np.vstack([X,S])
    if topo == "sphere_two":
        # --- square -> disk (atan2, full four quadrants) -> hemispheres ---
        is_tiny = (min(H, W) <= 3)
        if is_tiny:
            eps_u = 0.25 if H == 2 else 0.15
            u = np.linspace(eps_u, 1.0 - eps_u, H, endpoint=True)
            v = (np.arange(W) + 0.5) / W
        else:
            u = np.linspace(0, 1, H, endpoint=True)
            v = np.linspace(0, 1, W, endpoint=False)
        U,V = np.meshgrid(u,v,indexing="ij")
        Xs = 2*U - 1.0
        Ys = 2*V - 1.0

        r   = np.maximum(np.abs(Xs), np.abs(Ys))
        ang = np.arctan2(Ys, Xs)

        xd = r * np.cos(ang)
        yd = r * np.sin(ang)

        zcap = np.sqrt(np.clip(1.0 - r**2, 0.0, 1.0))
        XA = np.stack([xd, yd,  zcap], axis=2).reshape(-1,3)   # 北半球
        XB = np.stack([xd, yd, -zcap], axis=2).reshape(-1,3)   # 南半球
        tiny = 1e-6; XA[:,2]+=tiny; XB[:,2]-=tiny
        return np.vstack([XA, XB])
    if topo in ("hemisphere_n","hemisphere_s"):
        u = np.linspace(0,1,H,endpoint=True)
        v = np.linspace(0,1,W,endpoint=False)
        U,V = np.meshgrid(u,v,indexing="ij")
        lat = (U/2.0)*np.pi if topo=="hemisphere_n" else -(U/2.0)*np.pi
        lon = 2*np.pi*V; R=1.0
        x=R*np.cos(lat)*np.cos(lon); y=R*np.cos(lat)*np.sin(lon); z=R*np.sin(lat)
        return np.stack([x,y,z],axis=2).reshape(-1,3)
    return np.stack([V, -U, np.zeros_like(U)], axis=2).reshape(-1,3)

# ================= visualization =================
def plot3d(A: np.ndarray, coords: np.ndarray, topo: str, out_png: str,
           viz_hemisphere: str = "none",
           surface_edges: bool = False,
           edge_samples: int = 16,
           nodes: list = None,
           H: int = None, W: int = None,
           viz_show_glue: bool = False):
    N = A.shape[0]
    if coords.shape[0] != N: raise ValueError("coords size mismatch")
    is_spherical = topo in ("sphere", "sphere_two", "hemisphere_n", "hemisphere_s")
    if is_spherical and viz_hemisphere in ("north", "south"):
        keep = coords[:, 2] >= 0.0 if viz_hemisphere == "north" else coords[:, 2] <= 0.0
    else:
        keep = np.ones(N, dtype=bool)

    fig = plt.figure(figsize=(6, 6)); ax = fig.add_subplot(111, projection='3d')
    ax.scatter(coords[keep,0], coords[keep,1], coords[keep,2], s=6)

    def _plot_arc_on_sphere(u: int, v: int, clip_to_hemisphere: bool):
        Pu, Pv = coords[u], coords[v]
        ru, rv = float(np.linalg.norm(Pu)), float(np.linalg.norm(Pv))
        R = 0.5*(ru+rv) if (ru>0 and rv>0) else 1.0
        au = Pu/(ru if ru!=0 else 1.0); av = Pv/(rv if rv!=0 else 1.0)
        dot = float(np.clip(np.dot(au,av), -1.0, 1.0)); theta = np.arccos(dot)
        if theta < 1e-8:
            xs,ys,zs = [Pu[0],Pv[0]],[Pu[1],Pv[1]],[Pu[2],Pv[2]]
        else:
            tvals = np.linspace(0.0, 1.0, max(2, int(edge_samples)))
            s = np.sin(theta)
            pts = (np.sin((1-tvals)[:,None]*theta)/s)*au + (np.sin((tvals)[:,None]*theta)/s)*av
            pts = pts * R
            if clip_to_hemisphere:
                mask = (pts[:,2] >= 0.0) if viz_hemisphere=="north" else (pts[:,2] <= 0.0)
                pts = pts[mask]
                if pts.shape[0] < 2: return
            xs,ys,zs = pts[:,0], pts[:,1], pts[:,2]
        ax.plot(xs, ys, zs, linewidth=0.9, alpha=0.95)

    def _plot_edge(u: int, v: int):
        if surface_edges and is_spherical:
            if keep[u] and keep[v]: _plot_arc_on_sphere(u, v, clip_to_hemisphere=False)
        else:
            if keep[u] and keep[v]:
                ax.plot([coords[u,0], coords[v,0]],
                        [coords[u,1], coords[v,1]],
                        [coords[u,2], coords[v,2]], linewidth=0.5, alpha=0.6)

    for u in np.where(keep)[0]:
        for v in np.where(A[u]==1)[0]:
            if v<=u: continue
            _plot_edge(u,v)

    # 半球时显示 sphere_two 的跨赤道粘合弧
    if topo=="sphere_two" and viz_hemisphere in ("north","south") and viz_show_glue:
        if nodes is None or H is None or W is None: raise ValueError("viz_show_glue needs nodes/H/W")
        idx = {n:k for k,n in enumerate(nodes)}
        def is_boundary(i,j): return (i==0 or i==H-1 or j==0 or j==W-1)
        for layer, other in (("A","B"),("B","A")):
            for i in range(H):
                for j in range(W):
                    if not is_boundary(i,j): continue
                    u_node=(layer,i,j); t=boundary_index(i,j,H,W)
                    P=2*(H+W)-4; t2=(P-t)%P; bi2,bj2=boundary_coords(t2,H,W)
                    v_node=(other,bi2,bj2); u=idx[u_node]; v=idx[v_node]
                    _plot_arc_on_sphere(u, v, clip_to_hemisphere=True)

    tag=[]
    if viz_hemisphere!="none": tag.append(f"{viz_hemisphere}-hemisphere")
    if surface_edges and is_spherical: tag.append("surface-edges")
    if topo=="sphere_two" and viz_show_glue and viz_hemisphere!="none": tag.append("glue-shown")
    ax.set_title(topo if not tag else f"{topo} ({', '.join(tag)})")
    ax.set_box_aspect([1,1,1]); plt.tight_layout(); plt.savefig(out_png, dpi=200); plt.close()

# ================= CLI & main =================
def parse_topos(arg:str)->List[str]:
    if arg=="all":
        return ["plane","cylinder_x","cylinder_y","mobius_x","mobius_y",
                "torus","klein_x","klein_y","proj_plane",
                "sphere_two","hemisphere_n","hemisphere_s","sphere"]
    return [t.strip() for t in arg.split(",") if t.strip()]

def main():
    ap=argparse.ArgumentParser(description="Quotient space topology builder: adjacency / coords / viz via fundamental polygon gluing.")
    ap.add_argument("--H", type=int, default=6)
    ap.add_argument("--W", type=int, default=8)
    ap.add_argument("--topology", type=str, default="all")
    ap.add_argument("--out", type=str, default="./out")
    ap.add_argument("--neigh", type=int, default=4, choices=[4,8])
    ap.add_argument("--adj", action="store_true")
    ap.add_argument("--coords", action="store_true")
    ap.add_argument("--viz3d", action="store_true")
    ap.add_argument("--directed", action="store_true")
    ap.add_argument("--viz-hemisphere", choices=["none","north","south"], default="none")
    ap.add_argument("--surface-edges", action="store_true")
    ap.add_argument("--edge-samples", type=int, default=16)
    ap.add_argument("--viz-show-glue", action="store_true",
                    help="半球渲染时也显示 sphere_two 的跨赤道粘合边")
    args=ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    topos=parse_topos(args.topology)

    for topo in topos:
        print(f"[*] {topo}: H={args.H}, W={args.W}, neigh={args.neigh}")
        A, nodes, mapdf = build_adj(args.H,args.W, topo, neigh=args.neigh, undirected=(not args.directed))

        # === 保存：邻接矩阵（含/不含标签）与节点映射（顺序与 A 完全一致） ===
        if args.adj:
            base=f"{args.out}/A_{topo}_{args.H}x{args.W}"

            # 0) 行列顺序严格按 rowcol_index 对齐
            mapdf_sorted = mapdf.sort_values("rowcol_index").reset_index(drop=True)
            labels = mapdf_sorted["node_id"].astype(str).tolist()

            # 1) 原始无标签矩阵（保留以兼容）
            np.save(base+".npy", A)
            pd.DataFrame(A).to_csv(base+".csv", index=False, header=False)

            # 2) 带标签矩阵（行列名与 nodes 中 node_id 完全一致）
            dfA_labeled = pd.DataFrame(A, index=labels, columns=labels)
            dfA_labeled.to_csv(base + "_labeled.csv", encoding="utf-8")

            # 3) nodes 文件（按 rowcol_index 顺序输出）
            mapdf_sorted.to_csv(
                f"{args.out}/nodes_{topo}_{args.H}x{args.W}.csv",
                index=False
            )

        if args.coords or args.viz3d:
            X = coords_for_topo(args.H, args.W, topo)
            if args.coords:
                np.save(f"{args.out}/coords_{topo}_{args.H}x{args.W}.npy", X)
                np.savetxt(f"{args.out}/coords_{topo}_{args.H}x{args.W}.csv", X, delimiter=",")
            if args.viz3d:
                plot3d(
                    A, X, topo, f"{args.out}/{topo}_{args.H}x{args.W}_3d.png",
                    viz_hemisphere=args.viz_hemisphere,
                    surface_edges=args.surface_edges,
                    edge_samples=args.edge_samples,
                    nodes=nodes, H=args.H, W=args.W,
                    viz_show_glue=args.viz_show_glue
                )
    print(f"✅ Done. See: {os.path.abspath(args.out)}")

if __name__=="__main__":
    main()

