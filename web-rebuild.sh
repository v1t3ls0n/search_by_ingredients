# from repo root
docker compose stop web
docker compose rm -f web

# remove the marker in case it exists in the image’s writable layer
# (safe to ignore errors if the container isn't up yet)
docker compose run --rm web sh -lc 'rm -f /app/.indexed' || true

# rebuild web without cache to ensure the latest files are baked
docker compose build --no-cache web

# start web
docker compose up -d web

# follow logs and wait for: "Created index recipes_v2 ...", "Indexing into ..."
docker compose logs -f web
