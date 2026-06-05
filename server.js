const http = require("http");
const fs = require("fs");
const path = require("path");

const DIMS = 16;
const DEFAULT_DOC_MAX_DIST = 0.7;
const HOST = process.env.HOST || "127.0.0.1";
const PORT = Number.parseInt(process.env.PORT || "8080", 10);

function euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}

function cosine(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  if (na < 1e-9 || nb < 1e-9) return 1.0;
  return 1.0 - dot / (Math.sqrt(na) * Math.sqrt(nb));
}

function manhattan(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
  return s;
}

function getDistFn(metric) {
  if (metric === "cosine") return cosine;
  if (metric === "manhattan") return manhattan;
  return euclidean;
}

function confidenceFromDistance(distance) {
  return Math.max(0, Math.min(1, 1 - distance));
}

class BruteForce {
  constructor() {
    this.items = [];
  }

  insert(item) {
    this.items.push(item);
  }

  knn(q, k, dist) {
    return this.items
      .map((v) => [dist(q, v.emb), v.id])
      .sort((a, b) => a[0] - b[0] || a[1] - b[1])
      .slice(0, k);
  }

  remove(id) {
    this.items = this.items.filter((v) => v.id !== id);
  }
}

class KDNode {
  constructor(item) {
    this.item = item;
    this.left = null;
    this.right = null;
  }
}

class KDTree {
  constructor(dims) {
    this.root = null;
    this.dims = dims;
  }

  insert(item) {
    this.root = this.#insert(this.root, item, 0);
  }

  #insert(node, item, depth) {
    if (!node) return new KDNode(item);
    const axis = depth % this.dims;
    if (item.emb[axis] < node.item.emb[axis]) {
      node.left = this.#insert(node.left, item, depth + 1);
    } else {
      node.right = this.#insert(node.right, item, depth + 1);
    }
    return node;
  }

  knn(q, k, dist) {
    const heap = [];
    this.#knn(this.root, q, k, 0, dist, heap);
    return heap.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }

  #pushWorstHeap(heap, pair, k) {
    heap.push(pair);
    heap.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
    if (heap.length > k) heap.shift();
  }

  #knn(node, q, k, depth, dist, heap) {
    if (!node) return;
    const dn = dist(q, node.item.emb);
    if (heap.length < k || dn < heap[0][0]) this.#pushWorstHeap(heap, [dn, node.item.id], k);

    const axis = depth % this.dims;
    const diff = q[axis] - node.item.emb[axis];
    const closer = diff < 0 ? node.left : node.right;
    const farther = diff < 0 ? node.right : node.left;
    this.#knn(closer, q, k, depth + 1, dist, heap);
    if (heap.length < k || Math.abs(diff) < heap[0][0]) {
      this.#knn(farther, q, k, depth + 1, dist, heap);
    }
  }

  rebuild(items) {
    this.root = null;
    for (const item of items) this.insert(item);
  }
}

class SeededRandom {
  constructor(seed = 42) {
    this.state = seed >>> 0;
  }

  next() {
    this.state = (1664525 * this.state + 1013904223) >>> 0;
    return this.state / 0x100000000;
  }
}

class HNSW {
  constructor(m = 16, efBuild = 200) {
    this.G = new Map();
    this.M = m;
    this.M0 = 2 * m;
    this.efBuild = efBuild;
    this.mL = 1.0 / Math.log(m);
    this.topLayer = -1;
    this.entryPt = -1;
    this.rng = new SeededRandom(42);
  }

  randLevel() {
    return Math.floor(-Math.log(Math.max(this.rng.next(), Number.EPSILON)) * this.mL);
  }

  searchLayer(q, ep, ef, layer, dist) {
    const visited = new Set([ep]);
    const candidates = [[dist(q, this.G.get(ep).item.emb), ep]];
    const found = [[candidates[0][0], ep]];

    while (candidates.length) {
      candidates.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
      const [cd, cid] = candidates.shift();
      found.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
      if (found.length >= ef && cd > found[0][0]) break;

      const current = this.G.get(cid);
      if (!current || layer >= current.nbrs.length) continue;
      for (const nid of current.nbrs[layer]) {
        if (visited.has(nid) || !this.G.has(nid)) continue;
        visited.add(nid);
        const nd = dist(q, this.G.get(nid).item.emb);
        found.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
        if (found.length < ef || nd < found[0][0]) {
          candidates.push([nd, nid]);
          found.push([nd, nid]);
          found.sort((a, b) => b[0] - a[0] || b[1] - a[1]);
          if (found.length > ef) found.shift();
        }
      }
    }

    return found.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  }

  selectNbrs(candidates, maxM) {
    return candidates.slice(0, maxM).map((c) => c[1]);
  }

  insert(item, dist) {
    const id = item.id;
    const lvl = this.randLevel();
    this.G.set(id, { item, maxLyr: lvl, nbrs: Array.from({ length: lvl + 1 }, () => []) });

    if (this.entryPt === -1) {
      this.entryPt = id;
      this.topLayer = lvl;
      return;
    }

    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > lvl; lc--) {
      const node = this.G.get(ep);
      if (node && lc < node.nbrs.length) {
        const w = this.searchLayer(item.emb, ep, 1, lc, dist);
        if (w.length) ep = w[0][1];
      }
    }

    for (let lc = Math.min(this.topLayer, lvl); lc >= 0; lc--) {
      const w = this.searchLayer(item.emb, ep, this.efBuild, lc, dist);
      const maxM = lc === 0 ? this.M0 : this.M;
      const selected = this.selectNbrs(w, maxM);
      this.G.get(id).nbrs[lc] = selected;

      for (const nid of selected) {
        const neighbor = this.G.get(nid);
        if (!neighbor) continue;
        while (neighbor.nbrs.length <= lc) neighbor.nbrs.push([]);
        neighbor.nbrs[lc].push(id);
        if (neighbor.nbrs[lc].length > maxM) {
          const ds = neighbor.nbrs[lc]
            .filter((candidateId) => this.G.has(candidateId))
            .map((candidateId) => [
              dist(neighbor.item.emb, this.G.get(candidateId).item.emb),
              candidateId,
            ])
            .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
          neighbor.nbrs[lc] = ds.slice(0, maxM).map((x) => x[1]);
        }
      }
      if (w.length) ep = w[0][1];
    }

    if (lvl > this.topLayer) {
      this.topLayer = lvl;
      this.entryPt = id;
    }
  }

  knn(q, k, ef, dist) {
    if (this.entryPt === -1) return [];
    let ep = this.entryPt;
    for (let lc = this.topLayer; lc > 0; lc--) {
      const node = this.G.get(ep);
      if (node && lc < node.nbrs.length) {
        const w = this.searchLayer(q, ep, 1, lc, dist);
        if (w.length) ep = w[0][1];
      }
    }
    return this.searchLayer(q, ep, Math.max(ef, k), 0, dist).slice(0, k);
  }

  remove(id) {
    if (!this.G.has(id)) return;
    for (const node of this.G.values()) {
      node.nbrs = node.nbrs.map((layer) => layer.filter((nid) => nid !== id));
    }
    this.G.delete(id);
    if (this.entryPt === id) this.entryPt = this.G.size ? this.G.keys().next().value : -1;
    this.topLayer = -1;
    for (const node of this.G.values()) this.topLayer = Math.max(this.topLayer, node.maxLyr);
  }

  getInfo() {
    const maxL = Math.max(this.topLayer + 1, 1);
    const nodesPerLayer = Array(maxL).fill(0);
    const edgesPerLayer = Array(maxL).fill(0);
    const nodes = [];
    const edges = [];

    for (const [id, node] of this.G.entries()) {
      nodes.push({
        id,
        metadata: node.item.metadata,
        category: node.item.category,
        maxLyr: node.maxLyr,
      });
      for (let lc = 0; lc <= node.maxLyr && lc < maxL; lc++) {
        nodesPerLayer[lc]++;
        if (lc < node.nbrs.length) {
          for (const nid of node.nbrs[lc]) {
            if (id < nid) {
              edgesPerLayer[lc]++;
              edges.push({ src: id, dst: nid, lyr: lc });
            }
          }
        }
      }
    }

    return { topLayer: this.topLayer, nodeCount: this.G.size, nodesPerLayer, edgesPerLayer, nodes, edges };
  }
}

class VectorDB {
  constructor(dims) {
    this.store = new Map();
    this.bf = new BruteForce();
    this.kdt = new KDTree(dims);
    this.hnsw = new HNSW(16, 200);
    this.nextId = 1;
    this.dims = dims;
  }

  insert(metadata, category, emb, dist) {
    const item = { id: this.nextId++, metadata, category, emb };
    this.store.set(item.id, item);
    this.bf.insert(item);
    this.kdt.insert(item);
    this.hnsw.insert(item, dist);
    return item.id;
  }

  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.bf.remove(id);
    this.hnsw.remove(id);
    this.kdt.rebuild([...this.store.values()]);
    return true;
  }

  search(q, k, metric, algo) {
    const dist = getDistFn(metric);
    const start = process.hrtime.bigint();
    let raw;
    if (algo === "bruteforce") raw = this.bf.knn(q, k, dist);
    else if (algo === "kdtree") raw = this.kdt.knn(q, k, dist);
    else raw = this.hnsw.knn(q, k, 50, dist);
    const latencyUs = Number((process.hrtime.bigint() - start) / 1000n);
    const hits = raw
      .filter(([, id]) => this.store.has(id))
      .map(([d, id]) => {
        const item = this.store.get(id);
        return { id, meta: item.metadata, cat: item.category, emb: item.emb, dist: d };
      });
    return { hits, latencyUs, algo, metric };
  }

  benchmark(q, k, metric) {
    const dist = getDistFn(metric);
    const time = (fn) => {
      const start = process.hrtime.bigint();
      fn();
      return Number((process.hrtime.bigint() - start) / 1000n);
    };
    return {
      bfUs: time(() => this.bf.knn(q, k, dist)),
      kdUs: time(() => this.kdt.knn(q, k, dist)),
      hnswUs: time(() => this.hnsw.knn(q, k, 50, dist)),
      n: this.store.size,
    };
  }

  all() {
    return [...this.store.values()];
  }

  hnswInfo() {
    return this.hnsw.getInfo();
  }

  size() {
    return this.store.size;
  }
}

class OllamaClient {
  constructor(host = "127.0.0.1", port = 11434) {
    this.host = host;
    this.port = port;
    this.embedModel = "nomic-embed-text";
    this.genModel = "gemma4:e4b";
  }

  async request(pathname, body = null, timeoutMs = 30000) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const res = await fetch(`http://${this.host}:${this.port}${pathname}`, {
        method: body ? "POST" : "GET",
        headers: body ? { "Content-Type": "application/json" } : undefined,
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      });
      if (!res.ok) return null;
      return await res.json();
    } catch {
      return null;
    } finally {
      clearTimeout(timeout);
    }
  }

  async isAvailable() {
    return Boolean(await this.request("/api/tags", null, 2000));
  }

  async embed(text) {
    const data = await this.request(
      "/api/embeddings",
      { model: this.embedModel, prompt: text },
      30000,
    );
    return Array.isArray(data?.embedding) ? data.embedding : [];
  }

  async generate(prompt) {
    const data = await this.request(
      "/api/generate",
      { model: this.genModel, prompt, stream: false },
      180000,
    );
    return data?.response ?? "ERROR: Ollama unavailable. Run: ollama serve";
  }
}

class DocumentDB {
  constructor() {
    this.store = new Map();
    this.hnsw = new HNSW(16, 200);
    this.bf = new BruteForce();
    this.nextId = 1;
    this.dims = 0;
  }

  insert(title, text, emb) {
    if (this.dims === 0) this.dims = emb.length;
    const item = { id: this.nextId++, title, text, emb };
    this.store.set(item.id, item);
    const vectorItem = { id: item.id, metadata: title, category: "doc", emb };
    this.hnsw.insert(vectorItem, cosine);
    this.bf.insert(vectorItem);
    return item.id;
  }

  search(q, k, maxDist = DEFAULT_DOC_MAX_DIST) {
    if (!this.store.size) return [];
    const raw = this.store.size < 10 ? this.bf.knn(q, k, cosine) : this.hnsw.knn(q, k, 50, cosine);
    return raw
      .filter(([d, id]) => this.store.has(id) && d <= maxDist)
      .map(([d, id]) => [d, this.store.get(id)]);
  }

  remove(id) {
    if (!this.store.has(id)) return false;
    this.store.delete(id);
    this.hnsw.remove(id);
    this.bf.remove(id);
    return true;
  }

  all() {
    return [...this.store.values()];
  }

  size() {
    return this.store.size;
  }

  getDims() {
    return this.dims;
  }
}

function parseVec(value) {
  if (!value) return [];
  return String(value)
    .split(",")
    .map((x) => Number.parseFloat(x.trim()))
    .filter((x) => Number.isFinite(x));
}

function chunkText(text, chunkWords = 250, overlapWords = 30) {
  const words = text.trim().split(/\s+/).filter(Boolean);
  if (!words.length) return [];
  if (words.length <= chunkWords) return [text];
  const chunks = [];
  const step = chunkWords - overlapWords;
  for (let i = 0; i < words.length; i += step) {
    const end = Math.min(i + chunkWords, words.length);
    chunks.push(words.slice(i, end).join(" "));
    if (end === words.length) break;
  }
  return chunks;
}

function loadDemo(db) {
  const dist = getDistFn("cosine");
  db.insert("Linked List: nodes connected by pointers", "cs", [0.90,0.85,0.72,0.68,0.12,0.08,0.15,0.10,0.05,0.08,0.06,0.09,0.07,0.11,0.08,0.06], dist);
  db.insert("Binary Search Tree: O(log n) search and insert", "cs", [0.88,0.82,0.78,0.74,0.15,0.10,0.08,0.12,0.06,0.07,0.08,0.05,0.09,0.06,0.07,0.10], dist);
  db.insert("Dynamic Programming: memoization overlapping subproblems", "cs", [0.82,0.76,0.88,0.80,0.20,0.18,0.12,0.09,0.07,0.06,0.08,0.07,0.08,0.09,0.06,0.07], dist);
  db.insert("Graph BFS and DFS: breadth and depth first traversal", "cs", [0.85,0.80,0.75,0.82,0.18,0.14,0.10,0.08,0.06,0.09,0.07,0.06,0.10,0.08,0.09,0.07], dist);
  db.insert("Hash Table: O(1) lookup with collision chaining", "cs", [0.87,0.78,0.70,0.76,0.13,0.11,0.09,0.14,0.08,0.07,0.06,0.08,0.07,0.10,0.08,0.09], dist);
  db.insert("Calculus: derivatives integrals and limits", "math", [0.12,0.15,0.18,0.10,0.91,0.86,0.78,0.72,0.08,0.06,0.07,0.09,0.07,0.08,0.06,0.10], dist);
  db.insert("Linear Algebra: matrices eigenvalues eigenvectors", "math", [0.20,0.18,0.15,0.12,0.88,0.90,0.82,0.76,0.09,0.07,0.08,0.06,0.10,0.07,0.08,0.09], dist);
  db.insert("Probability: distributions random variables Bayes theorem", "math", [0.15,0.12,0.20,0.18,0.84,0.80,0.88,0.82,0.07,0.08,0.06,0.10,0.09,0.06,0.09,0.08], dist);
  db.insert("Number Theory: primes modular arithmetic RSA cryptography", "math", [0.22,0.16,0.14,0.20,0.80,0.85,0.76,0.90,0.08,0.09,0.07,0.06,0.08,0.10,0.07,0.06], dist);
  db.insert("Combinatorics: permutations combinations generating functions", "math", [0.18,0.20,0.16,0.14,0.86,0.78,0.84,0.80,0.06,0.07,0.09,0.08,0.06,0.09,0.10,0.07], dist);
  db.insert("Neapolitan Pizza: wood-fired dough San Marzano tomatoes", "food", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.90,0.86,0.78,0.72,0.08,0.06,0.09,0.07], dist);
  db.insert("Sushi: vinegared rice raw fish and nori rolls", "food", [0.06,0.08,0.07,0.09,0.09,0.06,0.08,0.07,0.86,0.90,0.82,0.76,0.07,0.09,0.06,0.08], dist);
  db.insert("Ramen: noodle soup with chashu pork and soft-boiled eggs", "food", [0.09,0.07,0.06,0.08,0.08,0.09,0.07,0.06,0.82,0.78,0.90,0.84,0.09,0.07,0.08,0.06], dist);
  db.insert("Tacos: corn tortillas with carnitas salsa and cilantro", "food", [0.07,0.09,0.08,0.06,0.06,0.07,0.09,0.08,0.78,0.82,0.86,0.90,0.06,0.08,0.07,0.09], dist);
  db.insert("Croissant: laminated pastry with buttery flaky layers", "food", [0.06,0.07,0.10,0.09,0.10,0.06,0.07,0.10,0.85,0.80,0.76,0.82,0.09,0.07,0.10,0.06], dist);
  db.insert("Basketball: fast-paced shooting dribbling slam dunks", "sports", [0.09,0.07,0.08,0.10,0.08,0.09,0.07,0.06,0.08,0.07,0.09,0.06,0.91,0.85,0.78,0.72], dist);
  db.insert("Football: tackles touchdowns field goals and strategy", "sports", [0.07,0.09,0.06,0.08,0.09,0.07,0.10,0.08,0.07,0.09,0.08,0.07,0.87,0.89,0.82,0.76], dist);
  db.insert("Tennis: racket volleys groundstrokes and Wimbledon serves", "sports", [0.08,0.06,0.09,0.07,0.07,0.08,0.06,0.09,0.09,0.06,0.07,0.08,0.83,0.80,0.88,0.82], dist);
  db.insert("Chess: openings endgames tactics strategic board game", "sports", [0.25,0.20,0.22,0.18,0.22,0.18,0.20,0.15,0.06,0.08,0.07,0.09,0.80,0.84,0.78,0.90], dist);
  db.insert("Swimming: butterfly freestyle backstroke Olympic competition", "sports", [0.06,0.08,0.07,0.09,0.08,0.06,0.09,0.07,0.10,0.08,0.06,0.07,0.85,0.82,0.86,0.80], dist);
}

const db = new VectorDB(DIMS);
const docDB = new DocumentDB();
const ollama = new OllamaClient();
loadDemo(db);

function sendJson(res, status, data) {
  const body = JSON.stringify(data);
  res.writeHead(status, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": "application/json",
  });
  res.end(body);
}

function sendText(res, status, body, contentType = "text/plain") {
  res.writeHead(status, {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
    "Content-Type": contentType,
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let body = "";
    req.on("data", (chunk) => {
      body += chunk;
      if (body.length > 20 * 1024 * 1024) {
        reject(new Error("body too large"));
        req.destroy();
      }
    });
    req.on("end", () => {
      try {
        resolve(body ? JSON.parse(body) : {});
      } catch {
        resolve({});
      }
    });
    req.on("error", reject);
  });
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://${req.headers.host || "localhost"}`);

  if (req.method === "OPTIONS") {
    sendText(res, 204, "");
    return;
  }

  if (req.method === "GET" && url.pathname === "/") {
    const file = path.join(__dirname, "index.html");
    if (!fs.existsSync(file)) {
      sendText(res, 404, "index.html not found");
      return;
    }
    sendText(res, 200, fs.readFileSync(file, "utf8"), "text/html");
    return;
  }

  if (req.method === "GET" && url.pathname === "/search") {
    const q = parseVec(url.searchParams.get("v"));
    if (q.length !== DIMS) return sendJson(res, 200, { error: `need ${DIMS}D vector` });
    const k = Number.parseInt(url.searchParams.get("k") || "5", 10);
    const metric = url.searchParams.get("metric") || "cosine";
    const algo = url.searchParams.get("algo") || "hnsw";
    const out = db.search(q, k, metric, algo);
    return sendJson(res, 200, {
      results: out.hits.map((h) => ({
        id: h.id,
        metadata: h.meta,
        category: h.cat,
        distance: Number(h.dist.toFixed(6)),
        embedding: h.emb.map((x) => Number(x.toFixed(4))),
      })),
      latencyUs: out.latencyUs,
      algo: out.algo,
      metric: out.metric,
    });
  }

  if (req.method === "POST" && url.pathname === "/insert") {
    const body = await readBody(req);
    const metadata = body.metadata || "";
    const category = body.category || "";
    const embedding = Array.isArray(body.embedding) ? body.embedding.map(Number) : [];
    if (!metadata || embedding.length !== DIMS || embedding.some((x) => !Number.isFinite(x))) {
      return sendJson(res, 200, { error: "invalid body" });
    }
    return sendJson(res, 200, { id: db.insert(metadata, category, embedding, getDistFn("cosine")) });
  }

  let match = url.pathname.match(/^\/delete\/(\d+)$/);
  if (req.method === "DELETE" && match) {
    return sendJson(res, 200, { ok: db.remove(Number(match[1])) });
  }

  if (req.method === "GET" && url.pathname === "/items") {
    return sendJson(res, 200, db.all().map((v) => ({
      id: v.id,
      metadata: v.metadata,
      category: v.category,
      embedding: v.emb.map((x) => Number(x.toFixed(4))),
    })));
  }

  if (req.method === "GET" && url.pathname === "/benchmark") {
    const q = parseVec(url.searchParams.get("v"));
    if (q.length !== DIMS) return sendJson(res, 200, { error: `need ${DIMS}D vector` });
    const k = Number.parseInt(url.searchParams.get("k") || "5", 10);
    const metric = url.searchParams.get("metric") || "cosine";
    const b = db.benchmark(q, k, metric);
    return sendJson(res, 200, {
      bruteforceUs: b.bfUs,
      kdtreeUs: b.kdUs,
      hnswUs: b.hnswUs,
      itemCount: b.n,
    });
  }

  if (req.method === "GET" && url.pathname === "/hnsw-info") {
    return sendJson(res, 200, db.hnswInfo());
  }

  if (req.method === "POST" && url.pathname === "/doc/insert") {
    const body = await readBody(req);
    const title = body.title || "";
    const text = body.text || "";
    if (!title || !text) return sendJson(res, 200, { error: "need title and text" });

    const chunks = chunkText(text, 250, 30);
    const ids = [];
    for (let i = 0; i < chunks.length; i++) {
      const emb = await ollama.embed(chunks[i]);
      if (!emb.length) {
        return sendJson(res, 200, {
          error: "Ollama unavailable. Install from https://ollama.com then run: ollama pull nomic-embed-text && ollama pull llama3.2",
        });
      }
      const chunkTitle = chunks.length > 1 ? `${title} [${i + 1}/${chunks.length}]` : title;
      ids.push(docDB.insert(chunkTitle, chunks[i], emb));
    }
    return sendJson(res, 200, { ids, chunks: chunks.length, dims: docDB.getDims() });
  }

  match = url.pathname.match(/^\/doc\/delete\/(\d+)$/);
  if (req.method === "DELETE" && match) {
    return sendJson(res, 200, { ok: docDB.remove(Number(match[1])) });
  }

  if (req.method === "GET" && url.pathname === "/doc/list") {
    return sendJson(res, 200, docDB.all().map((doc) => {
      const preview = doc.text.length > 120 ? `${doc.text.slice(0, 120)}...` : doc.text;
      return {
        id: doc.id,
        title: doc.title,
        preview,
        words: doc.text.trim().split(/\s+/).filter(Boolean).length,
      };
    }));
  }

  if (req.method === "POST" && url.pathname === "/doc/search") {
    const body = await readBody(req);
    const question = body.question || "";
    const k = Number.parseInt(body.k || "3", 10);
    const maxDist = Number.parseFloat(body.maxDist);
    const docMaxDist = Number.isFinite(maxDist) && maxDist > 0 ? maxDist : DEFAULT_DOC_MAX_DIST;
    if (!question) return sendJson(res, 200, { error: "need question" });
    const qEmb = await ollama.embed(question);
    if (!qEmb.length) return sendJson(res, 200, { error: "Ollama unavailable" });
    const hits = docDB.search(qEmb, k, docMaxDist);
    return sendJson(res, 200, {
      requestedK: k,
      returned: hits.length,
      maxDistance: docMaxDist,
      minConfidence: Number(confidenceFromDistance(docMaxDist).toFixed(4)),
      contexts: hits.map(([distance, doc]) => ({
        id: doc.id,
        title: doc.title,
        distance: Number(distance.toFixed(4)),
        confidence: Number(confidenceFromDistance(distance).toFixed(4)),
      })),
    });
  }

  if (req.method === "POST" && url.pathname === "/doc/ask") {
    const body = await readBody(req);
    const question = body.question || "";
    const k = Number.parseInt(body.k || "3", 10);
    const maxDist = Number.parseFloat(body.maxDist);
    const docMaxDist = Number.isFinite(maxDist) && maxDist > 0 ? maxDist : DEFAULT_DOC_MAX_DIST;
    if (!question) return sendJson(res, 200, { error: "need question" });
    const qEmb = await ollama.embed(question);
    if (!qEmb.length) return sendJson(res, 200, { error: "Ollama unavailable" });
    const hits = docDB.search(qEmb, k, docMaxDist);
    const context = hits
      .map(([, doc], i) => `[${i + 1}] ${doc.title}:\n${doc.text}\n`)
      .join("\n");
    const prompt = "You are a helpful assistant. Answer the user's question directly. "
      + "Use the provided context if it contains relevant information. "
      + "If it doesn't, just use your own general knowledge. "
      + "IMPORTANT: Do NOT mention the 'context', 'provided text', or say things like 'the context doesn't mention'. "
      + "Just answer the question naturally.\n\n"
      + `Context:\n${context}Question: ${question}\n\nAnswer:`;
    const answer = await ollama.generate(prompt);
    return sendJson(res, 200, {
      answer,
      model: ollama.genModel,
      requestedK: k,
      returned: hits.length,
      maxDistance: docMaxDist,
      minConfidence: Number(confidenceFromDistance(docMaxDist).toFixed(4)),
      contexts: hits.map(([distance, doc]) => ({
        id: doc.id,
        title: doc.title,
        text: doc.text,
        distance: Number(distance.toFixed(4)),
        confidence: Number(confidenceFromDistance(distance).toFixed(4)),
      })),
      docCount: docDB.size(),
    });
  }

  if (req.method === "GET" && url.pathname === "/status") {
    return sendJson(res, 200, {
      ollamaAvailable: await ollama.isAvailable(),
      embedModel: ollama.embedModel,
      genModel: ollama.genModel,
      docCount: docDB.size(),
      docDims: docDB.getDims(),
      demoDims: DIMS,
      demoCount: db.size(),
    });
  }

  if (req.method === "GET" && url.pathname === "/stats") {
    return sendJson(res, 200, {
      count: db.size(),
      dims: DIMS,
      algorithms: ["bruteforce", "kdtree", "hnsw"],
      metrics: ["euclidean", "cosine", "manhattan"],
    });
  }

  sendJson(res, 404, { error: "not found" });
});

server.listen(PORT, HOST, async () => {
  const ollamaUp = await ollama.isAvailable();
  console.log("=== VectorDB Engine (JavaScript) ===");
  console.log(`http://localhost:${PORT}`);
  console.log(`${db.size()} demo vectors | ${DIMS} dims | HNSW+KD-Tree+BruteForce`);
  console.log(`Ollama: ${ollamaUp ? "ONLINE" : "OFFLINE (install from ollama.com)"}`);
  if (ollamaUp) console.log(`  embed model: ${ollama.embedModel}  gen model: ${ollama.genModel}`);
});
