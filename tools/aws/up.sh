#!/usr/bin/env bash
# Start the instance, wait until SSH answers, and point the "cse291a" SSH host at its new IP.
# A stopped-and-started instance gets a new public IP, so run this at the start of every session.
#
# Needs: AWS CLI with credentials (`aws configure` or AWS_PROFILE), and in your shell profile:
#   export CSE291A_INSTANCE_ID=i-0123456789abcdef0
#   export AWS_REGION=us-west-2
# Without the AWS CLI: start the instance in the console and run tools/aws/set_ip.sh <public-ip>.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
id="${CSE291A_INSTANCE_ID:?set CSE291A_INSTANCE_ID, e.g. in ~/.zshrc}"
region="${AWS_REGION:-us-west-2}"

state=$(aws ec2 describe-instances --region "$region" --instance-ids "$id" \
        --query 'Reservations[0].Instances[0].State.Name' --output text)
if [ "$state" != "running" ]; then
  echo "starting $id (was $state) ..."
  aws ec2 start-instances --region "$region" --instance-ids "$id" >/dev/null
  aws ec2 wait instance-running --region "$region" --instance-ids "$id"
fi
ip=$(aws ec2 describe-instances --region "$region" --instance-ids "$id" \
     --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
"$here/set_ip.sh" "$ip"

# SSH is allowed only from your own IP; when you are on a new network, add its IP to the
# instance's security group (an "already exists" answer just means it is allowed already).
sg=$(aws ec2 describe-instances --region "$region" --instance-ids "$id" \
     --query 'Reservations[0].Instances[0].SecurityGroups[0].GroupId' --output text)
myip=$(curl -fsS https://checkip.amazonaws.com | tr -d '[:space:]')
if aws ec2 authorize-security-group-ingress --region "$region" --group-id "$sg" \
     --protocol tcp --port 22 --cidr "$myip/32" >/dev/null 2>&1; then
  echo "allowed SSH from this network ($myip)"
fi

echo -n "waiting for SSH on $ip "
for _ in $(seq 60); do
  if nc -z -G 3 "$ip" 22 2>/dev/null || nc -z -w 3 "$ip" 22 2>/dev/null; then
    echo " ok"; echo "ready: ssh cse291a   (or the reconnecting shell: cse)"; exit 0
  fi
  echo -n "."; sleep 5
done
echo; echo "SSH did not answer within 5 minutes; check the instance in the console" >&2; exit 1
