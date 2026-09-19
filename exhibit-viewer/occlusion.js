// Tiny BVH raycaster over the device's collision mesh, used to decide whether an overlay
// point (box corner, label anchor) is hidden behind a wall. Everything is in the source
// (Z-up) frame; the caller converts the camera position into that frame.

const LEAF_SIZE = 8;

/**
 * Voxel occupancy grid built from the Gaussians (tools/make_occupancy.py). Rays are walked
 * voxel by voxel (3D DDA); the first solid voxel outside the object's own box blocks the ray.
 */
export class VoxelOccluder {
    constructor(origin, voxel, dims, bits) {
        this.origin = origin;
        this.voxel = voxel;
        this.nx = dims[0];
        this.ny = dims[1];
        this.nz = dims[2];
        this.bits = bits;
        this.count = 0;
        for (const b of bits) this.count += popcount(b);
    }

    static async load(url) {
        const buf = await (await fetch(url, { cache: 'no-store' })).arrayBuffer();
        const dv = new DataView(buf);
        const origin = [dv.getFloat32(0, true), dv.getFloat32(4, true), dv.getFloat32(8, true)];
        const voxel = dv.getFloat32(12, true);
        const dims = [dv.getUint32(16, true), dv.getUint32(20, true), dv.getUint32(24, true)];
        return new VoxelOccluder(origin, voxel, dims, new Uint8Array(buf, 28));
    }

    solid(ix, iy, iz) {
        if (ix < 0 || iy < 0 || iz < 0 || ix >= this.nx || iy >= this.ny || iz >= this.nz) return false;
        const i = ix + this.nx * (iy + this.ny * iz);
        return (this.bits[i >> 3] >> (i & 7)) & 1;
    }

    /**
     * True when a solid voxel lies on the segment camera -> point. Voxels inside `skipBox`
     * ([minx,miny,minz,maxx,maxy,maxz], source frame) are ignored so an object never occludes
     * itself; the first `skipNear` metres from the camera are ignored too.
     */
    occluded(cx, cy, cz, px, py, pz, skipBox = null, skipNear = 0.25) {
        const dx = px - cx;
        const dy = py - cy;
        const dz = pz - cz;
        const dist = Math.hypot(dx, dy, dz);
        if (dist < 1e-6) return false;
        const vs = this.voxel;
        const ox = this.origin[0];
        const oy = this.origin[1];
        const oz = this.origin[2];
        // start a little away from the camera
        const t0 = Math.min(skipNear, dist);
        let x = cx + (dx / dist) * t0;
        let y = cy + (dy / dist) * t0;
        let z = cz + (dz / dist) * t0;
        let ix = Math.floor((x - ox) / vs);
        let iy = Math.floor((y - oy) / vs);
        let iz = Math.floor((z - oz) / vs);
        const sx = dx > 0 ? 1 : -1;
        const sy = dy > 0 ? 1 : -1;
        const sz = dz > 0 ? 1 : -1;
        const tdx = dx !== 0 ? Math.abs(vs / dx) * dist : Infinity;
        const tdy = dy !== 0 ? Math.abs(vs / dy) * dist : Infinity;
        const tdz = dz !== 0 ? Math.abs(vs / dz) * dist : Infinity;
        const nextBoundary = (p, o, s) => {
            const cell = Math.floor((p - o) / vs);
            return s > 0 ? (cell + 1) * vs + o - p : p - (cell * vs + o);
        };
        let tx = dx !== 0 ? (nextBoundary(x, ox, sx) / Math.abs(dx)) * dist : Infinity;
        let ty = dy !== 0 ? (nextBoundary(y, oy, sy) / Math.abs(dy)) * dist : Infinity;
        let tz = dz !== 0 ? (nextBoundary(z, oz, sz) / Math.abs(dz)) * dist : Infinity;
        // tx/ty/tz are absolute distances along the ray from the camera
        tx += t0;
        ty += t0;
        tz += t0;
        let t = t0;
        const tEnd = dist - vs * 0.5;
        let guard = 0;
        while (t < tEnd && guard++ < 4096) {
            if (this.solid(ix, iy, iz)) {
                if (!skipBox) return true;
                const vx = ox + (ix + 0.5) * vs;
                const vy = oy + (iy + 0.5) * vs;
                const vz = oz + (iz + 0.5) * vs;
                const inside =
                    vx >= skipBox[0] && vx <= skipBox[3] &&
                    vy >= skipBox[1] && vy <= skipBox[4] &&
                    vz >= skipBox[2] && vz <= skipBox[5];
                if (!inside) return true;
            }
            if (tx < ty && tx < tz) {
                ix += sx;
                t = tx;
                tx += tdx;
            } else if (ty < tz) {
                iy += sy;
                t = ty;
                ty += tdy;
            } else {
                iz += sz;
                t = tz;
                tz += tdz;
            }
        }
        return false;
    }
}

function popcount(b) {
    b -= (b >> 1) & 0x55;
    b = (b & 0x33) + ((b >> 2) & 0x33);
    return (b + (b >> 4)) & 0x0f;
}

export class CollisionMesh {
    constructor(positions, indices) {
        this.pos = positions; // Float32Array, xyz per vertex
        this.idx = indices;   // Uint32Array, 3 per triangle
        this.triCount = indices.length / 3;
        this._build();
    }

    static async load(url) {
        const buf = await (await fetch(url, { cache: 'no-store' })).arrayBuffer();
        const dv = new DataView(buf);
        const nv = dv.getUint32(0, true);
        const nf = dv.getUint32(4, true);
        const pos = new Float32Array(buf, 8, nv * 3);
        const idx = new Uint32Array(buf, 8 + nv * 12, nf * 3);
        return new CollisionMesh(pos, idx);
    }

    _build() {
        const n = this.triCount;
        const cen = new Float32Array(n * 3);
        const tmin = new Float32Array(n * 3);
        const tmax = new Float32Array(n * 3);
        const { pos, idx } = this;
        for (let t = 0; t < n; t++) {
            for (let a = 0; a < 3; a++) {
                const v0 = pos[idx[t * 3] * 3 + a];
                const v1 = pos[idx[t * 3 + 1] * 3 + a];
                const v2 = pos[idx[t * 3 + 2] * 3 + a];
                tmin[t * 3 + a] = Math.min(v0, v1, v2);
                tmax[t * 3 + a] = Math.max(v0, v1, v2);
                cen[t * 3 + a] = (v0 + v1 + v2) / 3;
            }
        }
        this.order = new Uint32Array(n);
        for (let i = 0; i < n; i++) this.order[i] = i;

        // node arrays: bounds, left child / first tri, tri count (0 = interior), right child
        const nodesMin = [];
        const nodesMax = [];
        const nodesA = [];
        const nodesB = [];
        const right = [];
        const build = (start, end) => {
            const id = nodesA.length;
            const mn = [Infinity, Infinity, Infinity];
            const mx = [-Infinity, -Infinity, -Infinity];
            const cmn = [Infinity, Infinity, Infinity];
            const cmx = [-Infinity, -Infinity, -Infinity];
            for (let i = start; i < end; i++) {
                const t = this.order[i];
                for (let a = 0; a < 3; a++) {
                    if (tmin[t * 3 + a] < mn[a]) mn[a] = tmin[t * 3 + a];
                    if (tmax[t * 3 + a] > mx[a]) mx[a] = tmax[t * 3 + a];
                    if (cen[t * 3 + a] < cmn[a]) cmn[a] = cen[t * 3 + a];
                    if (cen[t * 3 + a] > cmx[a]) cmx[a] = cen[t * 3 + a];
                }
            }
            nodesMin.push(mn);
            nodesMax.push(mx);
            nodesA.push(0);
            nodesB.push(0);
            right.push(-1);
            const count = end - start;
            if (count <= LEAF_SIZE) {
                nodesA[id] = start;
                nodesB[id] = count;
                return id;
            }
            let axis = 0;
            let ext = cmx[0] - cmn[0];
            for (let a = 1; a < 3; a++) {
                if (cmx[a] - cmn[a] > ext) {
                    ext = cmx[a] - cmn[a];
                    axis = a;
                }
            }
            const sub = this.order.subarray(start, end);
            sub.sort((p, q) => cen[p * 3 + axis] - cen[q * 3 + axis]);
            const mid = start + (count >> 1);
            nodesA[id] = build(start, mid);
            right[id] = build(mid, end);
            nodesB[id] = 0;
            return id;
        };
        build(0, n);
        this.nMin = nodesMin;
        this.nMax = nodesMax;
        this.nA = nodesA;
        this.nB = nodesB;
        this.nR = right;
    }

    /** Distance along the ray to the first hit, or Infinity. `d` must be normalised. */
    raycast(ox, oy, oz, dx, dy, dz, maxT = Infinity) {
        const idx = 1 / dx;
        const idy = 1 / dy;
        const idz = 1 / dz;
        let best = maxT;
        const stack = [0];
        const { nMin, nMax, nA, nB, nR, order, pos, idx: tri } = this;
        while (stack.length) {
            const n = stack.pop();
            const mn = nMin[n];
            const mx = nMax[n];
            // slab test
            let t0 = (mn[0] - ox) * idx;
            let t1 = (mx[0] - ox) * idx;
            let tmin = Math.min(t0, t1);
            let tmax = Math.max(t0, t1);
            t0 = (mn[1] - oy) * idy;
            t1 = (mx[1] - oy) * idy;
            tmin = Math.max(tmin, Math.min(t0, t1));
            tmax = Math.min(tmax, Math.max(t0, t1));
            t0 = (mn[2] - oz) * idz;
            t1 = (mx[2] - oz) * idz;
            tmin = Math.max(tmin, Math.min(t0, t1));
            tmax = Math.min(tmax, Math.max(t0, t1));
            if (tmax < Math.max(tmin, 0) || tmin > best) continue;

            const count = nB[n];
            if (count === 0) {
                stack.push(nA[n], nR[n]);
                continue;
            }
            for (let i = 0; i < count; i++) {
                const t = order[nA[n] + i] * 3;
                const i0 = tri[t] * 3;
                const i1 = tri[t + 1] * 3;
                const i2 = tri[t + 2] * 3;
                // Möller–Trumbore
                const e1x = pos[i1] - pos[i0];
                const e1y = pos[i1 + 1] - pos[i0 + 1];
                const e1z = pos[i1 + 2] - pos[i0 + 2];
                const e2x = pos[i2] - pos[i0];
                const e2y = pos[i2 + 1] - pos[i0 + 1];
                const e2z = pos[i2 + 2] - pos[i0 + 2];
                const px = dy * e2z - dz * e2y;
                const py = dz * e2x - dx * e2z;
                const pz = dx * e2y - dy * e2x;
                const det = e1x * px + e1y * py + e1z * pz;
                if (Math.abs(det) < 1e-9) continue;
                const inv = 1 / det;
                const tx = ox - pos[i0];
                const ty = oy - pos[i0 + 1];
                const tz = oz - pos[i0 + 2];
                const u = (tx * px + ty * py + tz * pz) * inv;
                if (u < 0 || u > 1) continue;
                const qx = ty * e1z - tz * e1y;
                const qy = tz * e1x - tx * e1z;
                const qz = tx * e1y - ty * e1x;
                const v = (dx * qx + dy * qy + dz * qz) * inv;
                if (v < 0 || u + v > 1) continue;
                const tt = (e2x * qx + e2y * qy + e2z * qz) * inv;
                if (tt > 1e-4 && tt < best) best = tt;
            }
        }
        return best === maxT ? Infinity : best;
    }

    /** True when the segment camera -> point is blocked by the mesh (with a small tolerance). */
    occluded(cx, cy, cz, px, py, pz, tol = 0.05) {
        const dx = px - cx;
        const dy = py - cy;
        const dz = pz - cz;
        const dist = Math.hypot(dx, dy, dz);
        if (dist < 1e-6) return false;
        const t = this.raycast(cx, cy, cz, dx / dist, dy / dist, dz / dist, dist - tol);
        return t < dist - tol;
    }
}
