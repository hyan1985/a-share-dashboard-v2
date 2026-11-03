PY ?= python3

.PHONY: help fetch run view test

help:
	@echo "make fetch   # 增量拉取数据"
	@echo "make run     # 运行模块并生成看板"
	@echo "make view    # 浏览器打开 output/index.html"

fetch:
	$(PY) -m src.cli fetch

run:
	$(PY) -m src.cli run

view:
	open output/index.html
