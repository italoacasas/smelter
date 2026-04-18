SCRIPTS := ./scripts

# Overridable inputs:
#   WORKLOAD=<name>   workload to activate (required for `use`)
#   INSTANCE=<name>   instance within the active workload (required for
#                     `chat` and `bench`; optional filter for `logs`,
#                     `health`, `download`, `refresh-moe-configs`)
#   LABEL=<string>    benchmark run label (default: manual)
#   REFRESH_MOE_ARGS  extra flags forwarded to refresh_moe_configs.py

.DEFAULT_GOAL := help
.PHONY: help use start stop restart dev logs health chat bench download pull refresh-moe-configs

help: ## Show available targets
	@awk 'BEGIN {FS = ":.*?## "; printf "Usage: make <target> [VAR=value]\n\nTargets:\n"} \
		/^[a-zA-Z][a-zA-Z0-9_-]*:.*?## / { printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2 }' $(MAKEFILE_LIST)
	@printf "\nInputs:\n"
	@printf "  WORKLOAD=<name>         workload to activate\n"
	@printf "  INSTANCE=<name>         single instance target\n"
	@printf "  LABEL=<string>          benchmark run label\n"
	@printf "  REFRESH_MOE_ARGS=<...>  flags forwarded to refresh_moe_configs.py\n"

use: ## Switch active workload: make use WORKLOAD=<name>
	@$(SCRIPTS)/use.py $(WORKLOAD)

start: ## Start the active workload (detached)
	@$(SCRIPTS)/start.sh

stop: ## Stop the active workload
	@$(SCRIPTS)/stop.sh

restart: stop start ## Stop then start the active workload

dev: ## Start active workload in foreground with info-level logs
	@$(SCRIPTS)/dev.sh

logs: ## Tail logs for the active workload (or INSTANCE=<name>)
	@$(SCRIPTS)/logs.sh

health: ## Run HTTP health checks (or INSTANCE=<name>)
	@$(SCRIPTS)/health-check.sh

chat: ## Interactive chat REPL (INSTANCE=<name> if workload has multiple)
	@$(SCRIPTS)/chat.sh

bench: ## Run sglang bench_serving (LABEL=... optional; INSTANCE=<name> if multi-instance)
	@$(SCRIPTS)/benchmark.py $(LABEL)

download: ## Download model weights (active workload, or INSTANCE=<name>)
	@$(SCRIPTS)/download-model.sh

pull: ## Pull the Docker image declared in hardware.json
	@$(SCRIPTS)/pull.sh

refresh-moe-configs: ## Refresh MoE config files for the target instance
	@$(SCRIPTS)/refresh_moe_configs.py $(REFRESH_MOE_ARGS)
