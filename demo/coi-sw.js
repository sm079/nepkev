// Cross-origin isolation for static hosts that cannot set headers (GitHub Pages): this service worker re-serves the
// page's own files with COOP/COEP, which lets ONNX Runtime's CPU backend use several threads (SharedArrayBuffer).
// Cross-origin requests are left alone; the page only makes CORS requests (jsDelivr, Hugging Face, Google Fonts),
// which COEP `credentialless` allows. The page registers this file and reloads once when it first takes control.

self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", (e) => e.waitUntil(self.clients.claim()));

self.addEventListener("fetch", (e) => {
  const req = e.request;
  if (new URL(req.url).origin !== self.location.origin) return;
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") return;
  e.respondWith(fetch(req).then((res) => {
    if (res.status === 0) return res;
    const headers = new Headers(res.headers);
    headers.set("Cross-Origin-Opener-Policy", "same-origin");
    headers.set("Cross-Origin-Embedder-Policy", "credentialless");
    return new Response(res.body, { status: res.status, statusText: res.statusText, headers });
  }));
});
