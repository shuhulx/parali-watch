CONFIG ?= config/punjab_2024.yaml
RUN = uv run --frozen paraliwatch

.PHONY: setup run refresh test lint demo serve query clean

setup:            ## install pinned environment
	uv sync --frozen

run: setup        ## full pipeline end to end (idempotent; ~25 min cold, ~6 min warm)
	$(RUN) run -c $(CONFIG)

refresh: run      ## rerun the configured season; ingestion and ARD reuse current scenes

test:
	uv run --frozen pytest -q

lint:
	uv run --frozen ruff check src tests

demo: serve      ## view the committed 2024 dashboard without running the pipeline

serve:            ## open the dashboard locally
	cd site && python3 -m http.server 8000

query:            ## example: the dispatch question as SQL
	$(RUN) query -c $(CONFIG) "SELECT week, round(sum(burnt_ha)) AS burnt_ha, round(sum(burnt_ha_confirmed)) AS confirmed_ha, round(sum(mcd64_ha)) AS mcd64_ha FROM hex_week GROUP BY week ORDER BY week"

clean:            ## remove derived data (keeps raw MODIS cache)
	rm -rf data/ard data/products data/features data/catalog
