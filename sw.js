// DHR Studio Suite — Service Worker v3.0
// AUTO-BUSTING: cache name includes a build timestamp so every deployment
// gets a unique CACHE_NAME → old caches are deleted on activate.
// JS/CSS bundles use network-first so new code always loads immediately.
const BUILD_TS = '__DHR_BUILD_TS__';
const CACHE_NAME = 'dhr-studio-v3-' + BUILD_TS;

const PRECACHE_URLS = ['/'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => cache.addAll(PRECACHE_URLS))
      .then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) =>
      Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      )
    ).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  if (event.request.method !== 'GET') return;
  const url = new URL(event.request.url);

  // HTML navigation: network-first
  if (event.request.mode === 'navigate') {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match('/'))
    );
    return;
  }

  // JS/CSS bundles: network-first so new deployments always load fresh code
  if (url.pathname.match(/\.(js|css)$/)) {
    event.respondWith(
      fetch(event.request)
        .then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        })
        .catch(() => caches.match(event.request))
    );
    return;
  }

  // Fonts, images, audio CDN assets: cache-first (immutable)
  if (
    url.pathname.match(/\.(woff2?|ttf|png|webp|svg|ico|mp3|wav)$/) ||
    url.hostname.includes('cloudfront.net') ||
    url.hostname.includes('fonts.googleapis.com') ||
    url.hostname.includes('fonts.gstatic.com') ||
    url.hostname.includes('tonejs.github.io')
  ) {
    event.respondWith(
      caches.match(event.request).then((cached) => {
        if (cached) return cached;
        return fetch(event.request).then((response) => {
          if (response.ok) {
            const clone = response.clone();
            caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
          }
          return response;
        });
      })
    );
    return;
  }
});
