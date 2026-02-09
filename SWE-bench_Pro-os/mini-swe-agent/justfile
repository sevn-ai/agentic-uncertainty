USERNAME := "$(whoami)"

# Build the Docker image for SWE Agent
build: 
    docker build \
        -t mini-sweagent-image:{{USERNAME}} \
        -f SWEAgent.Dockerfile . && \
    echo "Built image with tag: mini-sweagent-image:{{USERNAME}}"


# Run the docker container with the necessary mounts, onboarding the modal credentials
# and a results directory to write to as well as the config directories.
run:
    mkdir -p $(pwd)/msweagent_results && \
    docker run -it --rm --ipc=host \
    --name mini-sweagent-{{USERNAME}} \
    --env-file $(pwd)/.env \
    -v "$HOME/.modal.toml:/root/.modal.toml" \
    -v "$(pwd)/config:/app/config" \
    -v "$(pwd)/msweagent_wrapper_configs:/app/msweagent_wrapper_configs" \
    -v "$(pwd)/msweagent_results:/app/msweagent_results" \
    --add-host=host.docker.internal:host-gateway \
    mini-sweagent-image:{{USERNAME}}