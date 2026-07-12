# Training the Catan RL Agent on Google Cloud Compute Engine

This guide takes you from nothing to a running self-play PPO training job on
a Compute Engine VM, with TensorBoard monitoring from your laptop and
checkpoints synced back down.

**Which machine?** Training is dominated by *environment stepping and
batch-1 policy inference on CPU* — the network is a small MLP. A plain
CPU VM is the right default; a GPU only helps once you scale the network up
or batch parallel games. Start with `e2-standard-8`, use Spot pricing.

---

## 0. Prerequisites (once)

1. A Google Cloud account with billing enabled: https://console.cloud.google.com
2. The gcloud CLI installed locally: https://cloud.google.com/sdk/docs/install
3. Log in and create/select a project:

```bash
gcloud auth login
gcloud projects create catan-rl-project --name="Catan RL"   # or reuse an existing one
gcloud config set project catan-rl-project
gcloud services enable compute.googleapis.com
```

Pick a region near you (e.g. `us-east1`, `us-central1`):

```bash
gcloud config set compute/zone us-east1-b
```

## 1. Create the VM

**CPU (recommended first):** 8 vCPUs, Spot (preemptible) pricing — roughly
$0.07–0.10/hr Spot vs ~$0.27/hr on-demand:

```bash
gcloud compute instances create catan-rl-1 \
  --machine-type=e2-standard-8 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=ubuntu-2404-lts-amd64 \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=50GB
```

> Spot VMs can be stopped by Google at any time. Training survives this:
> checkpoints are written every `checkpoint_interval` iterations and
> `--resume` picks up from the latest one (see §5).

**GPU variant (only if you scale the model up):** use an L4 and the Deep
Learning VM image, which ships with drivers + CUDA PyTorch preinstalled:

```bash
gcloud compute instances create catan-rl-gpu \
  --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --provisioning-model=SPOT \
  --instance-termination-action=STOP \
  --image-family=common-cu124-ubuntu-2204-py310 \
  --image-project=deeplearning-platform-release \
  --boot-disk-size=100GB \
  --maintenance-policy=TERMINATE
```

(GPU quota may require a request in *IAM & Admin → Quotas* the first time.
When training on GPU, pass `--device cuda` to the train script.)

## 2. Get the code onto the VM

SSH in:

```bash
gcloud compute ssh catan-rl-1
```

Then either clone from your remote (preferred, if you've pushed this repo):

```bash
git clone https://github.com/<you>/catan-rl.git
cd catan-rl
```

…or, from your **local** machine, copy the working tree up (excluding junk):

```bash
# run locally, from the repo's parent directory
tar --exclude=.venv --exclude=.git --exclude=runs --exclude=__pycache__ \
    --exclude=.playwright-mcp -czf catan-rl.tgz "Catan RL"
gcloud compute scp catan-rl.tgz catan-rl-1:~
gcloud compute ssh catan-rl-1 -- "tar xzf catan-rl.tgz && mv 'Catan RL' catan-rl"
```

## 3. Set up Python on the VM

```bash
sudo apt-get update && sudo apt-get install -y python3-venv python3-pip tmux
cd ~/catan-rl
python3 -m venv .venv
source .venv/bin/activate

# CPU VM: install the CPU-only torch wheel first (much smaller, no CUDA)
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

# sanity check
python -m pytest tests/ -q
python scripts/benchmark_throughput.py --games 50
```

## 4. Launch training under tmux

SSH sessions die; tmux keeps the job alive:

```bash
tmux new -s train
source .venv/bin/activate
python scripts/train_self_play.py --config configs/ppo_baseline.yaml
```

Detach with `Ctrl-b d`, reattach later with `tmux attach -t train`.
(Alternative: `nohup python scripts/train_self_play.py --config configs/ppo_baseline.yaml > train.log 2>&1 &`)

Useful config overrides while iterating:

```bash
python scripts/train_self_play.py --config configs/ppo_baseline.yaml --iterations 100
python scripts/train_self_play.py --config configs/ppo_baseline.yaml --resume   # continue latest ckpt
```

## 5. Monitor with TensorBoard from your laptop

On the VM (inside tmux, a second window: `Ctrl-b c`):

```bash
source .venv/bin/activate
tensorboard --logdir runs/ --port 6006
```

On your **local** machine, open an SSH tunnel:

```bash
gcloud compute ssh catan-rl-1 -- -N -L 6006:localhost:6006
```

Then browse to http://localhost:6006. You should see `train/*` losses,
`game/*` self-play stats, and `eval/*` win rates against the random and
greedy bots and prior checkpoints.

## 6. Pull checkpoints back down

From your local machine:

```bash
gcloud compute scp --recurse catan-rl-1:~/catan-rl/runs/ppo_baseline ./runs/
python scripts/evaluate_checkpoints.py --run runs/ppo_baseline --games 50 --vs random,greedy,prev
python scripts/render_match.py --ckpt runs/ppo_baseline/checkpoints/ckpt_000500.pt --bots greedy,greedy,greedy,greedy --seat 0
```

For long runs, sync checkpoints to a bucket instead (survives VM deletion):

```bash
# once, locally:
gcloud storage buckets create gs://<your-unique-bucket>/
# on the VM, periodically or via cron:
gcloud storage rsync -r runs/ gs://<your-unique-bucket>/runs/
```

## 7. If a Spot VM gets preempted

The VM stops (not deleted). Restart and resume:

```bash
gcloud compute instances start catan-rl-1
gcloud compute ssh catan-rl-1
cd ~/catan-rl && source .venv/bin/activate
tmux new -s train
python scripts/train_self_play.py --config configs/ppo_baseline.yaml --resume
```

## 8. Stop paying

```bash
gcloud compute instances stop catan-rl-1     # keeps the disk (~$2/mo for 50GB), resume anytime
gcloud compute instances delete catan-rl-1   # gone for good — pull checkpoints first
```

## Cost cheat sheet (us-east1, mid-2026 ballpark)

| Setup | On-demand | Spot |
|---|---|---|
| e2-standard-8 (8 vCPU, 32 GB) | ~$0.27/hr | ~$0.08/hr |
| g2-standard-8 + L4 GPU | ~$0.85/hr | ~$0.25/hr |
| 50 GB balanced disk | ~$2/mo | ~$2/mo |

A weekend (48 h) of Spot CPU training ≈ **$4**. Always `stop` or `delete`
the instance when you're done — check with `gcloud compute instances list`.
