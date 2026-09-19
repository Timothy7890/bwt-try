// Exhibition viewer, scheme 1: per-object bounding boxes + screen labels + leader lines.
// The scene (point_cloud.ply / .sog) is rendered untouched by the SuperSplat viewer. Boxes,
// labels and leaders are drawn as SVG/DOM on top of the canvas by projecting the object
// geometry from objects.json every frame; the device's collision mesh tells us which parts
// are hidden behind walls.

import { Asset, Entity, Layer, Mat4, Vec3 } from 'playcanvas';
import { createViewer } from 'supersplat-viewer';

import { CollisionMesh, VoxelOccluder } from './occlusion.js';

// ------------------------------------------------------------------ config

const params = new URLSearchParams(location.search);
const CONTENT_URL = params.get('content') ?? (await pickDefaultContent());
const OBJECTS_URL = params.get('objects') ?? './data/objects.json';
const OCCLUDER = params.get('occluder') ?? 'voxel'; // voxel (from Gaussians) | mesh (device collision.ply)
const COLLISION_URL = params.get('collision') ?? (OCCLUDER === 'mesh' ? './data/collision.bin' : './data/occupancy.bin');
const SETTINGS_URL = params.get('settings') ?? './settings.json';
const MASKS_URL = params.get('masks') ?? './data/masks/manifest.json';

// the source data is Z-up; this is the same rotation used in the SuperSplat editor
const SOURCE_ROTATION = [-90, 0, 0];

const LABEL_RISE_PX = 56; // how far above the anchor the label sits
const EYE_HEIGHT = parseFloat(params.get('eye') ?? '1.7'); // metres above the floor
const WALK_FOV = parseFloat(params.get('walkfov') ?? '75');
const COLLISION_GLB = params.get('walkmesh') ?? './data/collision.glb';

async function pickDefaultContent() {
    for (const c of ['./data/scene.sog', './data/scene_fast.sog', './data/scene.compressed.ply']) {
        try {
            const r = await fetch(c, { method: 'HEAD' });
            if (r.ok) return c;
        } catch (_) {
            /* try next */
        }
    }
    return './data/scene.ply';
}

// ------------------------------------------------------------------ dom

const $ = (id) => document.getElementById(id);
const dom = {
    viewer: $('viewer'),
    boxes: $('boxes'),
    leaders: $('leaders'),
    labels: $('labels'),
    list: $('object-list'),
    status: $('status'),
    displayMode: $('display-mode'),
    toggleBoxes: $('toggle-boxes'),
    toggleLabels: $('toggle-labels'),
    toggleLeaders: $('toggle-leaders'),
    lineWidth: $('line-width'),
    lineWidthV: $('line-width-v'),
    lineAlpha: $('line-alpha'),
    lineAlphaV: $('line-alpha-v'),
    maskAlpha: $('mask-alpha'),
    maskAlphaV: $('mask-alpha-v'),
    occlusionMode: $('occlusion-mode'),
    btnFrame: $('btn-frame'),
    btnWalk: $('btn-walk'),
    eyeHeight: $('eye-height'),
    eyeHeightV: $('eye-height-v'),
    toggleEye: $('toggle-eye'),
    panel: $('panel')
};

// ------------------------------------------------------------------ viewer

const viewer = await createViewer({
    container: dom.viewer,
    settings: SETTINGS_URL,
    contentUrl: CONTENT_URL,
    collisionUrl: COLLISION_GLB, // enables walk mode (WASD on the floor)
    walkEyeHeight: EYE_HEIGHT,
    walkFov: WALK_FOV
});

const { app, events, state } = viewer;
const cameraEntity = app.root.findByName('camera');

events.on('progress:changed', (p) => {
    dom.status.textContent = `加载场景 ${p}%`;
});

// Re-orient the splat entity the moment the asset resolves (before the viewer measures the
// scene bound), so the Z-up data stands upright.
let splatEntity = null;
const splatReady = new Promise((resolve) => {
    const tryAttach = () => {
        const e = app.root.findByName('gsplat');
        if (!e) return false;
        e.setLocalEulerAngles(...SOURCE_ROTATION);
        splatEntity = e;
        resolve(e);
        return true;
    };
    app.assets.on('load', () => tryAttach());
    events.on('loaded:changed', () => tryAttach());
});

// ------------------------------------------------------------------ data

const [objectsJson, maskManifest, collision] = await Promise.all([
    (await fetch(OBJECTS_URL, { cache: 'no-store' })).json(),
    fetch(MASKS_URL, { cache: 'no-store' }).then((r) => (r.ok ? r.json() : {})).catch(() => ({})),
    (OCCLUDER === 'mesh' ? CollisionMesh.load(COLLISION_URL) : VoxelOccluder.load(COLLISION_URL)).catch((e) => {
        console.warn('occluder unavailable, occlusion disabled', e);
        return null;
    })
]);
dom.occlusionMode.value = collision ? 'hide' : 'ignore';

const EDGES = [0, 1, 1, 3, 3, 2, 2, 0, 4, 5, 5, 7, 7, 6, 6, 4, 0, 4, 1, 5, 2, 6, 3, 7];
const SVG_NS = 'http://www.w3.org/2000/svg';
const svg = (tag, attrs = {}) => {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
};

const objects = objectsJson.objects.map((o) => {
    // 8 corners in source frame. objects.json carries an oriented (yaw-aligned) box; fall
    // back to the axis-aligned bounds for older files. Corner bit i selects min/max on axis i.
    let corners = o.corners;
    if (!corners) {
        corners = [];
        for (let i = 0; i < 8; i++) {
            corners.push([
                i & 1 ? o.bbox_max[0] : o.bbox_min[0],
                i & 2 ? o.bbox_max[1] : o.bbox_min[1],
                i & 4 ? o.bbox_max[2] : o.bbox_min[2]
            ]);
        }
    }
    const boxCentre = o.box_centre ?? o.centre;
    // occlusion is tested slightly inside the box (towards its centre), so the padding that
    // puts corners into the floor or a wall does not count as "hidden"
    const probes = corners.map((c) => {
        const d = [boxCentre[0] - c[0], boxCentre[1] - c[1], boxCentre[2] - c[2]];
        const len = Math.hypot(...d);
        const k = len > 0 ? Math.min(0.1, len * 0.25) / len : 0;
        return [c[0] + d[0] * k, c[1] + d[1] * k, c[2] + d[2] * k];
    });
    const anchorProbe = [o.anchor[0], o.anchor[1], o.anchor[2] - Math.min(0.08, o.size[2] * 0.2)];
    // the object's own voxels must not count as occluders
    const skipBox = [...o.bbox_min.map((v) => v - 0.1), ...o.bbox_max.map((v) => v + 0.1)];
    return { ...o, corners, probes, anchorProbe, skipBox, visible: true, maskWanted: false, maskOccluded: false, screen: new Array(8), occ: new Uint8Array(8), anchorOcc: false };
});
let activeId = null;
const FLOOR_Z = objectsJson.floor_z ?? null; // source frame

await splatReady;

for (const o of objects) {
    // box edges
    const g = svg('g', { stroke: o.color });
    o.edges = [];
    for (let e = 0; e < 12; e++) {
        const line = svg('line');
        g.appendChild(line);
        o.edges.push(line);
    }
    dom.boxes.appendChild(g);
    o.boxG = g;

    // screen label
    const label = document.createElement('div');
    label.className = 'label';
    label.style.setProperty('--c', o.color);
    label.innerHTML = `<span class="dot"></span>${escapeHtml(o.display_name)}<span class="meta">${formatCount(o.count)}</span>`;
    label.addEventListener('click', () => selectObject(o.id, true));
    dom.labels.appendChild(label);
    o.label = label;

    // leader line + anchor dot
    const lg = svg('g');
    const poly = svg('polyline', { stroke: o.color });
    const dot = svg('circle', { r: 3.5, stroke: o.color });
    lg.appendChild(poly);
    lg.appendChild(dot);
    dom.leaders.appendChild(lg);
    o.leader = { g: lg, poly, dot };

    // panel entry
    const li = document.createElement('li');
    li.style.setProperty('--c', o.color);
    li.innerHTML = `<input type="checkbox" checked title="显示/隐藏" /><span class="swatch"></span>` +
        `<span class="name">${escapeHtml(o.display_name)}</span><span class="count">${formatCount(o.count)}</span>`;
    const cb = li.querySelector('input');
    cb.addEventListener('click', (ev) => {
        ev.stopPropagation();
        o.visible = cb.checked;
        refreshMasks();
        requestRedraw();
    });
    li.addEventListener('click', () => selectObject(o.id, true));
    dom.list.appendChild(li);
    o.li = li;
}

// ---- mask splats: one small recoloured splat per class, drawn over the untouched scene.
// They live in their own layer after the world so they always composite on top of the scene
// instead of interleaving with it (which lets dark original splats bleed through).
const maskLayer = new Layer({ name: 'MaskOverlay' });
{
    const worldLayer = app.scene.layers.getLayerByName('World');
    app.scene.layers.insert(maskLayer, app.scene.layers.getTransparentIndex(worldLayer) + 1);
    cameraEntity.camera.layers = [...cameraEntity.camera.layers, maskLayer.id];
}
const MASK_UNIFIED = params.get('maskunified') === '1';
const maskDir = MASKS_URL.slice(0, MASKS_URL.lastIndexOf('/') + 1);
function ensureMask(o) {
    if (o.maskEntity || o.maskLoading) return;
    const info = maskManifest[o.name];
    if (!info) return;
    o.maskLoading = true;
    const url = maskDir + info.file.split('/').pop();
    const asset = new Asset(`mask:${o.name}`, 'gsplat', { url, filename: `${o.name}.ply` });
    asset.on('load', () => {
        // keep the original opacities so the mask strength slider can rescale them live
        const res = asset.resource;
        if (res && res.gsplatData && typeof res.updateColorData === 'function') {
            o.maskResource = res;
            o.maskOrigOpacity = Float32Array.from(res.gsplatData.getProp('opacity'));
            applyMaskAlpha(o, parseFloat(dom.maskAlpha.value));
        }
        const e = new Entity(`mask:${o.name}`, app);
        e.addComponent('gsplat', MASK_UNIFIED ? { unified: true, asset } : { unified: false, asset, layers: [maskLayer.id] });
        splatEntity.addChild(e);
        o.maskEntity = e;
        o.maskLoading = false;
        refreshMasks();
    });
    asset.on('error', (err) => {
        console.warn('mask load failed', o.name, err);
        o.maskLoading = false;
    });
    app.assets.add(asset);
    app.assets.load(asset);
}

// rescale a mask's per-Gaussian opacity in place and re-upload the colour texture
function applyMaskAlpha(o, k) {
    if (!o.maskResource) return;
    const data = o.maskResource.gsplatData;
    const op = data.getProp('opacity');
    const orig = o.maskOrigOpacity;
    for (let i = 0; i < op.length; i++) {
        const a = Math.min(1 - 1e-4, Math.max(1e-4, (1 / (1 + Math.exp(-orig[i]))) * k));
        op[i] = Math.log(a / (1 - a));
    }
    o.maskResource.updateColorData(data);
}

dom.maskAlpha.addEventListener('input', () => {
    const k = parseFloat(dom.maskAlpha.value);
    dom.maskAlphaV.textContent = k.toFixed(2);
    for (const o of objects) applyMaskAlpha(o, k);
    app.renderNextFrame = true;
});

function refreshMasks() {
    const mode = dom.displayMode.value;
    const useMask = mode === 'mask' || mode === 'both';
    for (const o of objects) {
        const want = useMask && o.visible && (activeId === null || o.id === activeId);
        o.maskWanted = want;
        if (want) ensureMask(o);
        if (o.maskEntity) o.maskEntity.enabled = want && !o.maskOccluded;
    }
    app.renderNextFrame = true;
}

dom.status.textContent = `${objects.length} 个对象 · ${CONTENT_URL.split('/').pop()}` + (collision ? (collision instanceof VoxelOccluder ? ` · 遮挡体素 ${collision.count}` : ` · 遮挡网格 ${collision.triCount} 面`) : '');

// ------------------------------------------------------------------ interaction

function selectObject(id, fly) {
    activeId = activeId === id && !fly ? null : id;
    for (const o of objects) {
        const on = o.id === activeId;
        o.label.classList.toggle('active', on);
        o.li.classList.toggle('active', on);
    }
    if (fly && activeId !== null) flyTo(objects.find((o) => o.id === activeId));
    refreshMasks();
    requestRedraw();
}

function flyTo(o) {
    const world = splatEntity.getWorldTransform();
    const centre = world.transformPoint(new Vec3(...o.centre));
    const radius = Math.max(...o.size) * 0.5;
    const eyeLevel = dom.toggleEye.checked && FLOOR_Z !== null;
    const eye = parseFloat(dom.eyeHeight.value);
    // eye-level: stand back far enough that the whole object fits a ~60° view
    const dist = eyeLevel ? Math.max(radius * 2.2 + 0.6, 1.2) : Math.max(radius * 3.2, 1.2);
    const floorY = FLOOR_Z !== null ? world.transformPoint(new Vec3(0, 0, FLOOR_Z)).y : centre.y;
    const camY = eyeLevel ? floorY + eye : centre.y + dist * 0.35;

    // preferred: keep the current horizontal viewing direction, look slightly downward
    const camPos = cameraEntity.getPosition();
    const dir = new Vec3().sub2(camPos, centre);
    dir.y = 0;
    if (dir.length() < 1e-3) dir.set(0, 0, 1);
    dir.normalize();
    const baseYaw = Math.atan2(dir.x, dir.z);

    // try that direction first, then fan out left/right until the camera spot is in free
    // space and has a clear line of sight to the object (so we never land inside a wall)
    const position = new Vec3();
    const candidate = new Vec3();
    const srcCam = new Vec3();
    const srcTarget = new Vec3(...o.centre);
    invWorld.copy(world).invert();
    let chosen = false;
    const offsets = [0, 30, -30, 60, -60, 90, -90, 120, -120, 150, -150, 180];
    for (const off of offsets) {
        const yaw = baseYaw + (off * Math.PI) / 180;
        candidate.set(centre.x + Math.sin(yaw) * dist, camY, centre.z + Math.cos(yaw) * dist);
        if (!(collision instanceof VoxelOccluder)) {
            position.copy(candidate);
            chosen = true;
            break;
        }
        invWorld.transformPoint(candidate, srcCam);
        const ix = Math.floor((srcCam.x - collision.origin[0]) / collision.voxel);
        const iy = Math.floor((srcCam.y - collision.origin[1]) / collision.voxel);
        const iz = Math.floor((srcCam.z - collision.origin[2]) / collision.voxel);
        const inSolid = collision.solid(ix, iy, iz) || collision.solid(ix, iy, iz + 1) || collision.solid(ix, iy, iz - 1);
        const blocked = collision.occluded(srcCam.x, srcCam.y, srcCam.z, srcTarget.x, srcTarget.y, srcTarget.z, o.skipBox, 0.0);
        if (!inSolid && !blocked) {
            position.copy(candidate);
            chosen = true;
            break;
        }
    }
    if (!chosen) {
        position.copy(centre).add(dir.mulScalar(dist));
        position.y = camY;
    }

    events.fire('annotation.activate', {
        camera: { initial: { position: [position.x, position.y, position.z], target: [centre.x, centre.y, centre.z], fov: 60 } }
    });
}

dom.btnFrame.addEventListener('click', () => {
    selectObject(null, false);
    viewer.frameScene();
});
dom.btnWalk.addEventListener('click', () => {
    if (state.cameraMode === 'walk') {
        state.cameraMode = 'orbit';
    } else if (state.walkAllowed) {
        state.cameraMode = 'walk';
    } else {
        dom.status.textContent = '步行模式不可用：碰撞网格未加载';
    }
});
events.on('cameraMode:changed', (m) => {
    dom.btnWalk.textContent = m === 'walk' ? '退出步行模式' : '步行模式（人眼高度）';
});
dom.eyeHeight.addEventListener('input', () => {
    dom.eyeHeightV.textContent = dom.eyeHeight.value;
});

for (const el of [dom.toggleBoxes, dom.toggleLabels, dom.toggleLeaders, dom.occlusionMode]) {
    el.addEventListener('change', requestRedraw);
}
function setDisplayMode(mode) {
    dom.displayMode.value = mode;
    dom.toggleBoxes.checked = mode === 'box' || mode === 'both';
    refreshMasks();
    requestRedraw();
}
dom.displayMode.addEventListener('change', () => setDisplayMode(dom.displayMode.value));

// keyboard: Tab toggles the panel, 0-3 pick the display mode
const KEY_MODES = { 0: 'none', 1: 'box', 2: 'mask', 3: 'both' };
window.addEventListener('keydown', (ev) => {
    const tag = ev.target && ev.target.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (ev.key === 'Tab') {
        ev.preventDefault();
        dom.panel.classList.toggle('hidden');
    } else if ((ev.key in KEY_MODES || ev.code === 'Backquote') && !ev.ctrlKey && !ev.metaKey && !ev.altKey) {
        // the viewer maps 1/2/3 to orbit/fly/walk; we take these keys over (camera modes
        // stay reachable from the viewer's bottom-right buttons)
        ev.preventDefault();
        ev.stopPropagation();
        setDisplayMode(ev.code === 'Backquote' ? 'none' : KEY_MODES[ev.key]);
    }
}, true);
dom.lineWidth.addEventListener('input', () => {
    dom.lineWidthV.textContent = dom.lineWidth.value;
    requestRedraw();
});
dom.lineAlpha.addEventListener('input', () => {
    dom.lineAlphaV.textContent = dom.lineAlpha.value;
    requestRedraw();
});

function requestRedraw() {
    app.renderNextFrame = true;
    updateOverlay();
}

// ------------------------------------------------------------------ per-frame overlay

const tmpV = new Vec3();
const tmpS = new Vec3();
const camSrc = new Vec3();
const invWorld = new Mat4();

function projectSource(world, p, out) {
    // source-frame point -> screen (x, y) and view depth (z); z <= 0 means behind the camera
    world.transformPoint(tmpV.set(p[0], p[1], p[2]), tmpV);
    tmpS.sub2(tmpV, cameraEntity.getPosition());
    const depth = tmpS.dot(cameraEntity.forward);
    cameraEntity.camera.worldToScreen(tmpV, tmpS);
    out[0] = tmpS.x;
    out[1] = tmpS.y;
    out[2] = depth;
    return out;
}

function isOccluded(o, p) {
    return collision instanceof VoxelOccluder
        ? collision.occluded(camSrc.x, camSrc.y, camSrc.z, p[0], p[1], p[2], o.skipBox)
        : collision.occluded(camSrc.x, camSrc.y, camSrc.z, p[0], p[1], p[2]);
}

function updateOverlay() {
    const canvas = app.graphicsDevice.canvas;
    const cw = canvas.clientWidth;
    const ch = canvas.clientHeight;
    const world = splatEntity.getWorldTransform();
    const none = dom.displayMode.value === 'none';
    const showBoxes = dom.toggleBoxes.checked && !none;
    const showLabels = dom.toggleLabels.checked && !none;
    const showLeaders = dom.toggleLeaders.checked && showLabels;
    const occMode = collision ? dom.occlusionMode.value : 'ignore';
    const strokeW = parseFloat(dom.lineWidth.value);
    const alpha = parseFloat(dom.lineAlpha.value);

    // camera position in the source frame, for the occlusion raycasts
    if (occMode !== 'ignore') {
        invWorld.copy(world).invert();
        invWorld.transformPoint(cameraEntity.getPosition(), camSrc);
    }

    for (const o of objects) {
        const isActive = o.id === activeId;
        if (!o.visible) {
            o.boxG.style.display = 'none';
            o.label.style.display = 'none';
            o.leader.g.style.display = 'none';
            continue;
        }

        // corners
        let anyInFront = false;
        let allOcc = occMode !== 'ignore';
        for (let i = 0; i < 8; i++) {
            const s = (o.screen[i] ||= [0, 0, 0]);
            projectSource(world, o.corners[i], s);
            if (s[2] > 0.05) anyInFront = true;
            const c = o.probes[i];
            o.occ[i] = occMode !== 'ignore' && isOccluded(o, c) ? 1 : 0;
            if (!o.occ[i]) allOcc = false;
        }
        const anchor = projectSource(world, o.anchor, [0, 0, 0]);
        o.anchorOcc = occMode !== 'ignore' && isOccluded(o, o.anchorProbe);

        // ---- mask: the overlay splats ignore scene depth, so hide them when the object is
        // (almost) entirely behind something: at least 7 of the 9 probes blocked, or every
        // corner blocked (same rule that hides the whole box). Per-Gaussian occlusion needs
        // the shader route.
        let occCount = o.anchorOcc ? 1 : 0;
        for (let i = 0; i < 8; i++) occCount += o.occ[i];
        const maskOccluded = occMode !== 'ignore' && (allOcc || occCount >= 7);
        if (maskOccluded !== o.maskOccluded) {
            o.maskOccluded = maskOccluded;
            if (o.maskEntity) {
                o.maskEntity.enabled = o.maskWanted && !maskOccluded;
                app.renderNextFrame = true;
            }
        }

        // ---- box
        const drawBox = showBoxes && anyInFront && !(occMode === 'hide' && allOcc);
        o.boxG.style.display = drawBox ? 'block' : 'none';
        if (drawBox) {
            const baseAlpha = alpha * (activeId === null || isActive ? 1 : 0.4);
            o.boxG.setAttribute('stroke-width', isActive ? strokeW * 1.5 : strokeW);
            o.boxG.setAttribute('opacity', baseAlpha);
            for (let e = 0; e < 12; e++) {
                const a = o.screen[EDGES[e * 2]];
                const b = o.screen[EDGES[e * 2 + 1]];
                const line = o.edges[e];
                if (a[2] <= 0.05 || b[2] <= 0.05) {
                    line.classList.add('hidden');
                    continue;
                }
                line.classList.remove('hidden');
                line.setAttribute('x1', a[0]);
                line.setAttribute('y1', a[1]);
                line.setAttribute('x2', b[0]);
                line.setAttribute('y2', b[1]);
                const edgeOcc = o.occ[EDGES[e * 2]] && o.occ[EDGES[e * 2 + 1]];
                if (edgeOcc && occMode === 'hide') {
                    line.classList.add('hidden');
                } else if (edgeOcc && occMode === 'dim') {
                    line.classList.add('occluded');
                    line.setAttribute('opacity', 0.3);
                } else {
                    line.classList.remove('occluded');
                    line.removeAttribute('opacity');
                }
            }
        }

        // ---- label + leader
        const onScreen = anchor[2] > 0.05 && anchor[0] > -100 && anchor[0] < cw + 100 && anchor[1] > -100 && anchor[1] < ch + 100;
        const labelHidden = !showLabels || !onScreen || (occMode === 'hide' && o.anchorOcc && allOcc);
        o.label.style.display = labelHidden ? 'none' : 'block';
        o.leader.g.style.display = labelHidden || !showLeaders ? 'none' : 'block';
        if (!labelHidden) {
            o.label.classList.toggle('occluded', occMode === 'dim' && o.anchorOcc && !isActive);
            const x = anchor[0];
            const y = anchor[1];
            const ly = y - LABEL_RISE_PX;
            o.label.style.left = `${x}px`;
            o.label.style.top = `${ly}px`;
            if (showLeaders) {
                o.leader.g.setAttribute('stroke-width', strokeW);
                o.leader.g.setAttribute('opacity', alpha * (o.anchorOcc && occMode === 'dim' ? 0.4 : 1));
                o.leader.poly.setAttribute('points', `${x},${y} ${x},${ly + 2}`);
                o.leader.dot.setAttribute('cx', x);
                o.leader.dot.setAttribute('cy', y);
            }
        }
    }
}

app.on('framerender', updateOverlay);
requestRedraw();

// ------------------------------------------------------------------ utils

function escapeHtml(s) {
    return String(s).replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}

function formatCount(n) {
    return n >= 10000 ? `${(n / 10000).toFixed(1)}万` : `${n}`;
}

// handy for console poking
window.exhibit = { viewer, objects, collision, selectObject, flyTo, refreshMasks, setDisplayMode, state };
