.PHONY: download start stop logs health dev chat pull benchmark refresh-moe-configs use

download:
	./scripts/download-model.sh

start:
	./scripts/start.sh

stop:
	./scripts/stop.sh

logs:
	./scripts/logs.sh

health:
	./scripts/health-check.sh

dev:
	./scripts/dev.sh

chat:
	./scripts/chat.sh

pull:
	docker compose pull

benchmark:
	./scripts/benchmark.py $(LABEL)

refresh-moe-configs:
	./scripts/refresh-moe-configs.sh $(REFRESH_MOE_ARGS)

use:
	./scripts/use.py $(MODEL) $(HARDWARE)
