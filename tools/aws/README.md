# AWS remote setup

Your Mac stays the place you edit code; the instance runs experiments. One command starts and
connects, connections recover by themselves, and the instance stops itself when idle.

| Command (on the Mac) | What it does |
| --- | --- |
| `tools/aws/up.sh` | Start the instance, wait for SSH, point `cse291a` at its new IP |
| `cse` | Shell on the instance inside tmux; reconnects automatically if the network drops |
| `tools/aws/push.sh` | Send your code (incl. `.git`) to the instance |
| `tools/aws/pull.sh` | Copy the instance's `runs/` into `runs_aws/` on the Mac |
| `tools/aws/down.sh` | Pull results, then stop the instance (refuses while a sweep runs) |

On the instance, `idle_shutdown.sh` stops it after 30 minutes with no SSH session, no sweep and
no llama-server. A sweep left running in tmux keeps it alive and it stops 30 minutes after the
sweep ends. This only works if the instance's **shutdown behavior is "Stop"**.

## One-time setup

**1. Launch the instance** (EC2 console, region from the course instructions, e.g. us-west-2):
**Deep Learning Base AMI with Single CUDA (Amazon Linux 2023), 64-bit (x86)**, `g6f.2xlarge`
(fallback `g4dn.xlarge`), key pair `cse291a`, a security group allowing SSH (22) from your IP only,
the smallest root volume the AMI allows (gp3), and under Advanced details **Shutdown behavior: Stop**.
The login user is `ec2-user` (already set in `ssh_config`; use `ubuntu` on an Ubuntu AMI). Copy the instance ID (`i-...`).

**2. Mac: key, SSH config, shell shortcuts**

```bash
mv ~/Downloads/cse291a.pem ~/.ssh/ && chmod 400 ~/.ssh/cse291a.pem
mkdir -p ~/.ssh/config.d
cp ~/Desktop/UCSD/FALL2026/cse291/agentserve/tools/aws/ssh_config ~/.ssh/config.d/cse291a
# the Include line must be at the very top of ~/.ssh/config
grep -q 'config.d/cse291a' ~/.ssh/config 2>/dev/null || \
  { printf 'Include config.d/cse291a*\n\n' | cat - ~/.ssh/config 2>/dev/null > /tmp/sshc; mv /tmp/sshc ~/.ssh/config; }
chmod 600 ~/.ssh/config

cat >> ~/.zshrc <<'EOF'
# cse291a
export CSE291A_INSTANCE_ID=i-REPLACE_ME
export AWS_REGION=us-west-2
export PATH="$HOME/Desktop/UCSD/FALL2026/cse291/agentserve/tools/aws:$PATH"
cse() { until ssh -t cse291a 'tmux new -A -s main'; do echo "connection lost, reconnecting in 5 s..."; sleep 5; done; }
EOF
source ~/.zshrc
```

**3. AWS CLI** (so `up.sh` / `down.sh` can start and stop the instance):
`brew install awscli`, then `aws configure` with the access keys from your course AWS account.
If the account gives you no access keys, start and stop the instance in the console and run
`set_ip.sh <public-ip>` after each start instead of `up.sh`; everything else is the same.

**4. First connection and bootstrap** (~15 min: build, model download, venv)

```bash
up.sh                     # or: set_ip.sh <ip>
push.sh
ssh cse291a 'bash ~/cse291/agentserve/tools/aws/bootstrap.sh'
cse                       # then, inside: python -m agentserve.sweep configs/sweep_mock.json
```

**5. VS Code (optional):** install "Remote - SSH", then *Connect to Host* → `cse291a` and open
`~/cse291/agentserve`. It reconnects on its own after drops and shares the SSH connection above.
Use it to browse files and run terminals; keep editing code on the Mac and `push.sh` it, so there
is only one copy you change.

## Every session

```bash
up.sh && push.sh && cse          # start, sync code, get a tmux shell
# inside tmux on the instance:
python -m agentserve.sweep configs/sweep_ksweep.json
# detach with Ctrl-b d; close the laptop if you like; the sweep keeps running
pull.sh                          # any time: copy results to runs_aws/
down.sh                          # when done (or just leave it: idle auto-stop)
```

`localhost:18080` on the Mac reaches the instance's llama-server while you are connected,
for example `curl localhost:18080/slots`.

## Costs to keep in mind

- g6f.2xlarge is $0.48/hr, so the $3/day cap allows about 5.5 hours of running per day.
- The disk bills even while stopped (gp3 is about $0.08 per GB-month: a 75 GB root volume is about $0.20/day); the public IPv4 address bills only while running.
- No Elastic IP is used: it would bill around the clock. `up.sh` handles the changing IP instead.
