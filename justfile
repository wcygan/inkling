set windows-shell := ["powershell.exe", "-NoLogo", "-NoProfile", "-Command"]

# List the recipes
default:
    @just --list

# Check that everything needed is installed, and say how to install what is missing
doctor *args:
    uv run --script scripts/inkling.py doctor {{ args }}

# Show what is ready and what to do next
status:
    uv run --script scripts/inkling.py status

# Download the checkpoint (about 20 GB) at the pinned revision, then verify it
weights *args:
    uv run --script scripts/inkling.py weights {{ args }}

# Verify the checkpoint against the pinned hashes
check-weights *args:
    uv run --script scripts/inkling.py check-weights {{ args }}

# Build the GPU kernels from source into kernels/ (the first run fetches cuda-oxide)
kernels *args:
    uv run --script scripts/inkling.py kernels {{ args }}

# Build the engine (the host side)
build:
    uv run --script scripts/inkling.py build

# Start the server, replacing this repository's running one (overrides: `just serve --max-ctx 131072`)
serve *args:
    uv run --script scripts/inkling.py serve {{ args }}

# Stop this repository's server
stop *args:
    uv run --script scripts/inkling.py stop {{ args }}

# Open Pi on the local server in the current folder (extra args go to pi)
[extension(".ps1")]
[no-cd]
[positional-arguments]
[script("powershell.exe", "-NoLogo", "-NoProfile", "-File")]
pi *args:
    $env:PI_OFFLINE = '1'
    pi --model inkling/inkling-2-mini --thinking low @args
    exit $LASTEXITCODE
