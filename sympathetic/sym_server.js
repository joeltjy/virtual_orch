// Sympathetic Virtual Orchestra: gesture relay.
//
// Serves public/ (Bob's pages plus conduct.html) and forwards the gesture events conduct.js
// POSTs to /sym as OSC/UDP to virtual_orch's controller port (a browser cannot send UDP).
// Node built-ins only (no npm install).
//
//   node sym_server.js                    # http://localhost:8000/conduct.html -> OSC 127.0.0.1:9001
//   PORT=8000 OSC_HOST=127.0.0.1 OSC_PORT=9001 node sym_server.js
//
// In virtual_orch, enter the same port under "Controller OSC port" and press Connect.

const http = require("http");
const fs = require("fs");
const path = require("path");
const dgram = require("dgram");

const HTTP_PORT = Number(process.env.PORT || 8000);
const OSC_HOST = process.env.OSC_HOST || "127.0.0.1";
const OSC_PORT = Number(process.env.OSC_PORT || 9001);
const PUBLIC_DIR = path.join(__dirname, "public");

const ALLOWED = new Set([
  "/sym/point", "/sym/release", "/sym/expression", "/sym/beat",
  "/sym/stop", "/sym/play", "/sym/reset", "/sym/enable", "/sym/warp",
]);

// ---- minimal OSC 1.0 encoder (s, f, i) --------------------------------------------------
function oscString(s) {
  const raw = Buffer.from(s + "\0", "utf8");
  const padded = Buffer.alloc(Math.ceil(raw.length / 4) * 4);
  raw.copy(padded);
  return padded;
}

function encodeOsc(address, args) {
  let tags = ",";
  const parts = [];
  for (const arg of args) {
    if (typeof arg === "string") {
      tags += "s";
      parts.push(oscString(arg));
    } else if (Number.isInteger(arg) && Math.abs(arg) < 2 ** 31) {
      tags += "i";
      const b = Buffer.alloc(4);
      b.writeInt32BE(arg);
      parts.push(b);
    } else if (typeof arg === "number") {
      tags += "f";
      const b = Buffer.alloc(4);
      b.writeFloatBE(arg);
      parts.push(b);
    } else {
      throw new Error("unsupported OSC argument: " + JSON.stringify(arg));
    }
  }
  return Buffer.concat([oscString(address), oscString(tags), ...parts]);
}

let udp = null;

function sendOsc(address, args) {
  if (udp === null) udp = dgram.createSocket("udp4");
  udp.send(encodeOsc(address, args), targetPort, embeddedSession ? "127.0.0.1" : OSC_HOST);
}

// ---- HTTP ---------------------------------------------------------------------------------
const TYPES = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8", ".json": "application/json", ".png": "image/png",
  ".jpg": "image/jpeg", ".mp3": "audio/mpeg", ".task": "application/octet-stream",
};

function serveStatic(urlPath, res) {
  const rel = urlPath === "/" ? "conduct.html" : urlPath.replace(/^\/+/, "");
  const file = path.normalize(path.join(PUBLIC_DIR, rel));
  if (!file.startsWith(PUBLIC_DIR + path.sep)) {
    res.writeHead(403).end();
    return;
  }
  fs.readFile(file, (err, data) => {
    if (err) {
      res.writeHead(404).end("not found");
      return;
    }
    res.writeHead(200, { "Content-Type": TYPES[path.extname(file)] || "application/octet-stream" });
    res.end(data);
  });
}

function handleSym(req, res) {
  let body = "";
  req.on("data", (chunk) => {
    body += chunk;
    if (body.length > 10000) req.destroy();
  });
  req.on("end", () => {
    try {
      const { address, args = [] } = JSON.parse(body);
      if (!ALLOWED.has(address) || !Array.isArray(args)) throw new Error("bad address");
      sendOsc(address, args);
      if (!embeddedSession && address !== "/sym/expression") console.log(address, ...args);
      res.writeHead(204).end();
    } catch (e) {
      res.writeHead(400).end(String(e.message || e));
    }
  });
}

const embeddedSession = process.argv[2] === "--embedded";
const sessionPrefix = embeddedSession ? "/" + process.argv[3] : "";
const targetPort = embeddedSession ? Number(process.argv[4]) : OSC_PORT;

function start() {
  const server = http.createServer((req, res) => {
    let pathname;
    try {
      pathname = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    } catch {
      res.writeHead(400).end();
      return;
    }
    if (sessionPrefix && !pathname.startsWith(sessionPrefix + "/")) {
      res.writeHead(404).end();
      return;
    }
    pathname = pathname.slice(sessionPrefix.length);
    if (req.method === "POST" && pathname === "/sym") handleSym(req, res);
    else if (req.method === "GET") serveStatic(pathname, res);
    else res.writeHead(405).end();
  });
  return server.listen(embeddedSession ? 0 : HTTP_PORT, "127.0.0.1", () => {
    const port = server.address().port;
    if (embeddedSession) {
      const readyFile = process.argv[5];
      fs.writeFileSync(readyFile + ".tmp", String(port), { mode: 0o600 });
      fs.renameSync(readyFile + ".tmp", readyFile);
    } else {
      console.log(`Conducting page: http://localhost:${port}/conduct.html`);
      console.log(`Gestures -> OSC ${OSC_HOST}:${targetPort}`);
    }
  });
}

if (require.main === module) start();

module.exports = { encodeOsc };
