CONFIG ?= config/punjab_2024.yaml
RUN = uv run paraliwatch

.PHONY: setup run refresh test lint serve query clean

setup:            ## install pinned environment
	uv sync --frozen

run: setup        ## full pipeline end to end (idempotent; ~25 min cold, ~6 min warm)
	$(RUN) run -c $(CONFIG)

refresh: run      ## same thing: every stage only does work the catalogue says is missing

test:
	uv run pytest -q

lint:
	uv run ruff check src tests

serve:            ## open the dashboard locally
	cd site && python3 -m http.server 8000

query:            ## example: the dispatch question as SQL
	$(RUN) query "SELECT week, round(sum(burnt_ha)) AS burnt_ha, round(sum(burnt_ha_confirmed)) AS confirmed_ha, round(sum(mcd64_ha)) AS mcd64_ha FROM hex_week GROUP BY week ORDER BY week"

clean:            ## remove derived data (keeps raw MODIS cache)
	rm -rf data/ard data/products data/features data/catalog
