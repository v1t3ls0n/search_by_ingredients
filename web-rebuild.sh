docker compose build --no-cache web
docker compose up -d
docker compose logs -f web
# remove the marker and restart web to re-run indexing from init.sh
docker compose exec web rm -f /app/.indexed
docker compose restart web
# tail logs again
docker compose logs -f web