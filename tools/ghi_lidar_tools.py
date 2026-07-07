"""Herramientas LiDAR del Pipeline GHI (Brush).

Subcomandos (los invoca la pestaña "Pipeline GHI" de brush_app):
  align --workdir W --lidar L   Sim3 LiDAR->COLMAP (PCA 4 orientaciones + ICP).
                                Escribe W/lidar_aligned.ply y W/align_sim3.npy
  color --workdir W             Coloreado robusto two-pass del LiDAR alineado
                                (peso nitidez + 1/z, sin saturados, rechazo de
                                outliers). Escribe W/lidar_colored.ply
  init  --workdir W             Sustituye sparse/0/points3D por el LiDAR
                                coloreado (max 600k pts) como init del 3DGS
  post  --workdir W --lidar L --out DIR --stem NAME
                                Co-registra el ultimo splat exportado al marco
                                LiDAR (metros) y escribe en DIR:
                                  NAME_splat_metros.ply     (SuperSplat/Brush)
                                  NAME_splat_metros_CC.ply  (CloudCompare, RGB)
                                  NAME_LiDAR_color_metros.ply (nube laser+color)

Requiere: numpy, scipy, plyfile, opencv-python-headless (pip install --user).
Layout esperado del workdir (lo crea el panel): images/, sparse/0, sparse_txt/.
"""
import argparse, glob, os, sys
import numpy as np

try:
    from plyfile import PlyData, PlyElement
except ImportError:
    sys.exit("FALTA plyfile: python -m pip install --user plyfile")

C0 = 0.28209479177387814


# ---------------- IO COLMAP (TXT) y PLY ----------------
def qvec2rotmat(q):
    w, x, y, z = q
    return np.array([
        [1-2*y*y-2*z*z, 2*x*y-2*z*w,   2*x*z+2*y*w],
        [2*x*y+2*z*w,   1-2*x*x-2*z*z, 2*y*z-2*x*w],
        [2*x*z-2*y*w,   2*y*z+2*x*w,   1-2*x*x-2*y*y]])


def read_cameras_txt(path):
    cams = {}
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            e = line.split()
            cams[int(e[0])] = dict(model=e[1], w=int(e[2]), h=int(e[3]),
                                   params=np.array([float(x) for x in e[4:]]))
    return cams


def read_images_txt(path):
    imgs = {}
    with open(path) as f:
        lines = f.readlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith('#') or not line.strip():
            i += 1
            continue
        e = line.split()
        q = np.array([float(x) for x in e[1:5]])
        t = np.array([float(x) for x in e[5:8]])
        imgs[e[9]] = dict(R=qvec2rotmat(q), t=t, cam_id=int(e[8]))
        i += 2
    return imgs


def read_points3D_txt(path):
    xyz, err, tlen = [], [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            e = line.split()
            xyz.append([float(e[1]), float(e[2]), float(e[3])])
            err.append(float(e[7]))
            tlen.append((len(e)-8)//2)
    return np.array(xyz), np.array(err), np.array(tlen)


def read_ply_xyz(path):
    v = PlyData.read(path)['vertex'].data
    xyz = np.stack([v['x'], v['y'], v['z']], 1).astype(np.float64)
    return xyz, v


def write_ply_xyzrgb(path, xyz, rgb):
    n = len(xyz)
    arr = np.empty(n, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr['x'], arr['y'], arr['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(path)


# ---------------- align ----------------
def umeyama(src, dst):
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src-mu_s, dst-mu_d
    U, D, Vt = np.linalg.svd((dc.T @ sc)/len(src))
    S = np.eye(3)
    if np.linalg.det(U)*np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S)/((sc**2).sum()/len(src))
    return s, R, mu_d - s*R@mu_s


def check_float32_safe(xyz, origen):
    """Aviso de 'cancelacion catastrofica': los splats van en float32 (~7
    digitos significativos). Coordenadas grandes (UTM/georref) destruyen el
    detalle sub-metrico. Trabajar SIEMPRE en origen local."""
    m = np.abs(xyz).max()
    if m > 50000:
        print(f"AVISO [{origen}]: coordenadas de hasta {m:.0f} — en float32 "
              f"pierdes precision sub-metrica (cancelacion catastrofica).\n"
              f"  Resta un offset local antes de procesar y aplica la "
              f"georreferenciacion solo en el visor/exportacion final.",
              flush=True)


def cmd_align(w, lidar):
    from scipy.spatial import cKDTree
    xyz_c, err, tlen = read_points3D_txt(os.path.join(w, "sparse_txt", "points3D.txt"))
    good = (tlen >= 3) & (err < np.percentile(err, 90))
    xyz_c = xyz_c[good]
    med = np.median(xyz_c, 0)
    xyz_c = xyz_c[np.linalg.norm(xyz_c-med, axis=1) < np.percentile(
        np.linalg.norm(xyz_c-med, axis=1), 95)]
    print(f"sparse COLMAP filtrada: {len(xyz_c)}", flush=True)

    xyz_full, _ = read_ply_xyz(lidar)
    check_float32_safe(xyz_full, "nube LiDAR")
    sub = np.random.default_rng(0).choice(len(xyz_full), min(60000, len(xyz_full)), False)
    xyz_l = xyz_full[sub]
    tree = cKDTree(xyz_c)

    def pca(X):
        _, _, Vt = np.linalg.svd(X - X.mean(0), full_matrices=False)
        return Vt
    Vc, Vl = pca(xyz_c), pca(xyz_l)
    s0 = np.sqrt(((xyz_c-xyz_c.mean(0))**2).sum()/len(xyz_c)) / \
         np.sqrt(((xyz_l-xyz_l.mean(0))**2).sum()/len(xyz_l))

    def icp(s, R, t, iters=40):
        for _ in range(iters):
            srt = s*(R@xyz_l.T).T + t
            d, idx = tree.query(srt, workers=-1)
            m = d <= np.percentile(d, 80)
            s, R, t = umeyama(xyz_l[m], xyz_c[idx[m]])
        srt = s*(R@xyz_l.T).T + t
        d, _ = tree.query(srt, workers=-1)
        return s, R, t, np.sqrt((d[d <= np.percentile(d, 80)]**2).mean())

    best = None
    for sx, sy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
        V = Vl.copy(); V[0] *= sx; V[1] *= sy; V[2] = np.cross(V[0], V[1])
        R0 = Vc.T @ V
        if np.linalg.det(R0) < 0:
            continue
        t0 = xyz_c.mean(0) - s0*R0@xyz_l.mean(0)
        r = icp(s0, R0, t0)
        print(f"  orientacion ({sx:+d},{sy:+d}) rmse={r[3]:.4f}", flush=True)
        if best is None or r[3] < best[3]:
            best = r
    s, R, t, rmse = best
    scene = np.linalg.norm(xyz_c.max(0)-xyz_c.min(0))
    print(f"RMSE {rmse:.4f} ({100*rmse/scene:.2f}% de la escena)", flush=True)
    if rmse/scene > 0.05:
        sys.exit("ALINEACION DUDOSA (>5%): revisa la nube o alinea a mano en CloudCompare")

    out = s*(R@xyz_full.T).T + t
    inten = None
    _, v = read_ply_xyz(lidar)
    if 'intensity' in v.dtype.names:
        inten = np.clip(np.asarray(v['intensity']), 0, 255).astype(np.uint8)
    g = inten if inten is not None else np.full(len(out), 128, np.uint8)
    write_ply_xyzrgb(os.path.join(w, "lidar_aligned.ply"), out, np.stack([g, g, g], 1))
    np.save(os.path.join(w, "align_sim3.npy"), dict(s=s, R=R, t=t, rmse=rmse))
    print("align OK", flush=True)


# ---------------- color ----------------
def cmd_color(w):
    import cv2
    cams = read_cameras_txt(os.path.join(w, "sparse_txt", "cameras.txt"))
    imgs = dict(sorted(read_images_txt(os.path.join(w, "sparse_txt", "images.txt")).items()))
    xyz, _ = read_ply_xyz(os.path.join(w, "lidar_aligned.ply"))
    N = len(xyz)
    imgdir = os.path.join(w, "images")
    DS, SAT = 4, 250
    sharp = {}

    def project(Xc, cam):
        p = cam['params']; model = cam['model']
        x = Xc[:, 0]/Xc[:, 2]; y = Xc[:, 1]/Xc[:, 2]
        if model in ('OPENCV', 'FULL_OPENCV'):
            fx, fy, cx, cy, k1, k2, p1, p2 = p[:8]
            r2 = x*x+y*y; rad = 1+k1*r2+k2*r2*r2
            return (fx*(x*rad+2*p1*x*y+p2*(r2+2*x*x))+cx,
                    fy*(y*rad+p1*(r2+2*y*y)+2*p2*x*y)+cy)
        if model == 'PINHOLE':
            fx, fy, cx, cy = p[:4]
            return fx*x+cx, fy*y+cy
        if model in ('SIMPLE_RADIAL', 'RADIAL'):
            f, cx, cy, k1 = p[0], p[1], p[2], p[3]
            k2 = p[4] if len(p) > 4 else 0.0
            r2 = x*x+y*y; rad = 1+k1*r2+k2*r2*r2
            return f*x*rad+cx, f*y*rad+cy
        raise SystemExit(f"modelo de camara no soportado: {model}")

    def samples(name, im, cam):
        Wp, Hp = cam['w'], cam['h']
        Xc = (im['R']@xyz.T).T + im['t']
        idx = np.where(Xc[:, 2] > 0.05)[0]
        if not len(idx):
            return None
        u, v = project(Xc[idx], cam)
        inb = (u >= 0) & (u < Wp) & (v >= 0) & (v < Hp)
        idx, u, v = idx[inb], u[inb], v[inb]
        if not len(idx):
            return None
        z = Xc[idx, 2]
        Wc = (Wp+DS-1)//DS
        lin = (v/DS).astype(np.int32)*Wc + (u/DS).astype(np.int32)
        zbuf = np.full(((Hp+DS-1)//DS)*Wc, np.inf)
        np.minimum.at(zbuf, lin, z)
        vis = z <= zbuf[lin]*1.03
        idx, u, v, z = idx[vis], u[vis], v[vis], z[vis]
        if not len(idx):
            return None
        img = cv2.imread(os.path.join(imgdir, name), cv2.IMREAD_COLOR)
        if img is None:
            return None
        ui = np.clip(u.astype(np.int32), 0, Wp-1)
        vi = np.clip(v.astype(np.int32), 0, Hp-1)
        col = img[vi, ui][:, ::-1].astype(np.float64)
        ok = col.max(1) < SAT
        idx, col, z = idx[ok], col[ok], z[ok]
        if not len(idx):
            return None
        if name not in sharp:
            g = cv2.imread(os.path.join(imgdir, name), cv2.IMREAD_GRAYSCALE)
            g = cv2.resize(g, (max(1, g.shape[1]//6), max(1, g.shape[0]//6)))
            sharp[name] = cv2.Laplacian(g, cv2.CV_64F).var()
        return idx, col, sharp[name]/np.maximum(z, 0.5)

    acc = np.zeros((N, 3)); wsum = np.zeros(N)
    for k, (name, im) in enumerate(imgs.items()):
        s = samples(name, im, cams[im['cam_id']])
        if s:
            idx, col, wgt = s
            acc[idx] += col*wgt[:, None]; wsum[idx] += wgt
        if k % 50 == 0:
            print(f"  color pass1 [{k}/{len(imgs)}]", flush=True)
    seen1 = wsum > 0
    mean1 = np.zeros((N, 3)); mean1[seen1] = acc[seen1]/wsum[seen1, None]

    acc[:] = 0; wsum[:] = 0
    for k, (name, im) in enumerate(imgs.items()):
        s = samples(name, im, cams[im['cam_id']])
        if s:
            idx, col, wgt = s
            good = np.linalg.norm(col-mean1[idx], axis=1) < 70
            idx, col, wgt = idx[good], col[good], wgt[good]
            acc[idx] += col*wgt[:, None]; wsum[idx] += wgt
        if k % 50 == 0:
            print(f"  color pass2 [{k}/{len(imgs)}]", flush=True)
    seen = wsum > 0
    fb = seen1 & ~seen
    rgb = np.zeros((N, 3), np.uint8)
    rgb[seen] = np.clip(acc[seen]/wsum[seen, None], 0, 255).astype(np.uint8)
    rgb[fb] = np.clip(mean1[fb], 0, 255).astype(np.uint8)
    keep = seen | fb
    print(f"coloreados {keep.sum()}/{N} ({100*keep.sum()/N:.1f}%)", flush=True)
    write_ply_xyzrgb(os.path.join(w, "lidar_colored.ply"), xyz[keep], rgb[keep])
    print("color OK", flush=True)


# ---------------- init ----------------
def cmd_init(w, max_pts=600000):
    xyz, v = read_ply_xyz(os.path.join(w, "lidar_colored.ply"))
    rgb = np.stack([v['red'], v['green'], v['blue']], 1)
    if len(xyz) > max_pts:
        sel = np.random.default_rng(0).choice(len(xyz), max_pts, False)
        xyz, rgb = xyz[sel], rgb[sel]
    sparse0 = os.path.join(w, "sparse", "0")
    lines = [f"{i+1} {x:.5f} {y:.5f} {z:.5f} {r} {g} {b}"
             for i, ((x, y, z), (r, g, b)) in enumerate(zip(xyz, rgb))]
    with open(os.path.join(sparse0, "points3D.txt"), "w") as f:
        f.write("# init desde LiDAR coloreado (Pipeline GHI)\n")
        f.write("\n".join(lines) + "\n")
    # retirar el .bin para que Brush lea el .txt sin ambiguedad
    b = os.path.join(sparse0, "points3D.bin")
    if os.path.exists(b):
        os.replace(b, os.path.join(sparse0, "points3D_colmap_original.bak"))
    print(f"init OK ({len(xyz)} puntos)", flush=True)


# ---------------- post ----------------
def latest_export(w):
    pats = [os.path.join(os.path.dirname(w.rstrip("\\/")),
                         os.path.basename(w.rstrip("\\/")) + "_exports", "*.ply"),
            os.path.join(w, "*_exports", "*.ply"),
            os.path.join(w, "exports", "*.ply")]
    c = [p for pat in pats for p in glob.glob(pat)]
    if not c:
        sys.exit("no se encontro ningun export .ply del entrenamiento")
    return max(c, key=os.path.getmtime)


def cmd_post(w, lidar, out, stem):
    from scipy.spatial import cKDTree
    from scipy.spatial.transform import Rotation as Rot
    d = np.load(os.path.join(w, "align_sim3.npy"), allow_pickle=True).item()
    s, R, t = float(d['s']), d['R'], d['t']
    RT = R.T
    src = latest_export(w)
    print("export usado:", src, flush=True)

    data = PlyData.read(src)['vertex'].data.copy()
    xyz = np.stack([data['x'], data['y'], data['z']], 1).astype(np.float64)
    ok = np.isfinite(xyz).all(1)
    data, xyz = data[ok], xyz[ok]
    c = np.median(xyz, 0)
    near = np.linalg.norm(xyz-c, axis=1) < 60.0*s
    data, xyz = data[near], xyz[near]

    xyz_l = (RT@(xyz-t).T).T/s
    check_float32_safe(xyz_l, "splat co-registrado")
    for i, cn in enumerate(('x', 'y', 'z')):
        data[cn] = xyz_l[:, i].astype(np.float32)
    for cn in ('scale_0', 'scale_1', 'scale_2'):
        data[cn] = (data[cn] + np.log(1.0/s)).astype(np.float32)
    qx, qy, qz, qw = Rot.from_matrix(RT).as_quat()
    ww = data['rot_0'].astype(np.float64); xx = data['rot_1'].astype(np.float64)
    yy = data['rot_2'].astype(np.float64); zz = data['rot_3'].astype(np.float64)
    data['rot_0'] = (qw*ww - qx*xx - qy*yy - qz*zz).astype(np.float32)
    data['rot_1'] = (qw*xx + qx*ww + qy*zz - qz*yy).astype(np.float32)
    data['rot_2'] = (qw*yy - qx*zz + qy*ww + qz*xx).astype(np.float32)
    data['rot_3'] = (qw*zz + qx*yy - qy*xx + qz*ww).astype(np.float32)

    os.makedirs(out, exist_ok=True)
    p1 = os.path.join(out, f"{stem}_splat_metros.ply")
    PlyData([PlyElement.describe(data, 'vertex')], text=False).write(p1)
    print("escrito:", p1, flush=True)

    rgb = np.clip(0.5 + C0*np.stack([data['f_dc_0'], data['f_dc_1'], data['f_dc_2']], 1), 0, 1)
    opac = 1.0/(1.0+np.exp(-data['opacity'].astype(np.float64)))
    solid = opac > 0.2
    p2 = os.path.join(out, f"{stem}_splat_metros_CC.ply")
    write_ply_xyzrgb(p2, xyz_l[solid], (rgb[solid]*255).astype(np.uint8))
    print("escrito:", p2, flush=True)

    cx, cv = read_ply_xyz(os.path.join(w, "lidar_colored.ply"))
    crgb = np.stack([cv['red'], cv['green'], cv['blue']], 1)
    p3 = os.path.join(out, f"{stem}_LiDAR_color_metros.ply")
    write_ply_xyzrgb(p3, (RT@(cx-t).T).T/s, crgb)
    print("escrito:", p3, flush=True)

    lx, _ = read_ply_xyz(lidar)
    dd, _ = cKDTree(lx).query(
        xyz_l[np.random.default_rng(0).choice(np.where(solid)[0],
                                              min(50000, int(solid.sum())), False)],
        workers=-1)
    print(f"solape splat<->LiDAR: mediana {np.median(dd)*100:.1f} cm", flush=True)
    print("post OK", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["align", "color", "init", "post"])
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--lidar")
    ap.add_argument("--out")
    ap.add_argument("--stem", default="escena")
    a = ap.parse_args()
    if a.cmd == "align":
        if not a.lidar:
            sys.exit("align requiere --lidar")
        cmd_align(a.workdir, a.lidar)
    elif a.cmd == "color":
        cmd_color(a.workdir)
    elif a.cmd == "init":
        cmd_init(a.workdir)
    elif a.cmd == "post":
        if not (a.lidar and a.out):
            sys.exit("post requiere --lidar y --out")
        cmd_post(a.workdir, a.lidar, a.out, a.stem)


if __name__ == "__main__":
    main()
