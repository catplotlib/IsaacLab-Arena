# Arena images

Build an image without starting a container:

```bash
./docker/build_docker.sh -t dev -n isaaclab_arena:candidate-dev
./docker/build_docker.sh -t runtime -n isaaclab_arena:candidate-runtime
./docker/build_docker.sh -t dev-curobo -n isaaclab_arena:candidate-dev-curobo
./docker/build_docker.sh -t runtime-curobo -n isaaclab_arena:candidate-runtime-curobo
```

`-R` disables build cache; `-l /tmp/arena-build.log` saves plain BuildKit output
and elapsed time. Builds refresh base-image metadata and use cache by default.
They do not push images or start containers.

| Target | Capabilities | Default local tag |
| --- | --- | --- |
| `runtime` | Simulation, Arena, policy clients, search/review/notebook tools, Hugging Face and OSMO CLIs | `isaaclab_arena:runtime` |
| `dev` | Runtime plus pre-commit, GitHub CLI, and debugpy | `isaaclab_arena:latest` |
| `runtime-curobo` | Runtime plus cuRobo GPU reachability | `isaaclab_arena:runtime-curobo` |
| `dev-curobo` | Runtime with cuRobo plus developer tools | `isaaclab_arena:curobo` |

GR00T, OpenPI, Cosmos, and DreamZero inference servers remain separate from
these Arena images. The default Docker target is `dev` through a final alias.

## Local development

```bash
./docker/run_docker.sh                   # dev, existing image reused
./docker/run_docker.sh -t runtime
./docker/run_docker.sh -c                # dev-curobo
./docker/run_docker.sh -t runtime-curobo
./docker/run_docker.sh -r                # rebuild with cache
./docker/run_docker.sh -R                # rebuild without cache
```

The launcher mounts this checkout into `/workspaces/isaaclab_arena`. Ordinary
source edits require restarting the Python process, not rebuilding the image.
An explicit build copies updated source but reuses unchanged dependency stages.
If a container is already running, the launcher attaches to it; rebuilding its
image does not replace that container. Stop and recreate it separately when you
want to use the rebuilt image.

The `-n` launcher option changes the image repository/name; target selection
sets the tag shown above. Dataset/model/evaluation mounts and clone-specific
container naming retain their existing options.

## Build organization

The Dockerfile defines the stage graph. Scripts in `setup/` own installation
and shell setup; each step mounts only its required inputs. Dependency lists
are derived from `pyproject.toml`. User-facing entries in the existing `dev`
extra remain runtime capabilities. Developer targets explicitly install debugpy
and its shortcut; Jupyter may also bring the debugpy library transitively.

The cuRobo builder compiles wheels independently of Arena source. Runtime
installs the wheel's declared requirements with the existing Torch/CUDA-related
package versions constrained, then installs the wheel without dependencies.
Build provenance is retained at `/usr/local/share/arena/curobo-build.json`.

Direct builds must replace `--build-arg INSTALL_CUROBO=true` with
`--target dev-curobo` or `--target runtime-curobo`. The old build argument no
longer selects a feature. `push_to_ngc.sh` and CI publishing integration are
deferred; do not use the old publisher's `-c` path with this staged Dockerfile.

Local validation includes all test groups, mounted and baked-source smoke
checks, and a real cuRobo GPU IK solve on both cuRobo targets with empty
extension caches. See [the design](../DOCKER_BUILD_DESIGN.md) for scope and
acceptance checks; remote-cache and CI integration are a separate phase.
