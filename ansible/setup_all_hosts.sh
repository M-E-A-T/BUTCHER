#!/bin/bash
# Ensure SSH key exists + install SSH keys with sshpass

PASSWORD="meat"

# generate key if none exists
if [ ! -f "$HOME/.ssh/id_ed25519" ] && [ ! -f "$HOME/.ssh/id_rsa" ]; then
    echo "=== No SSH identities found. Generating one... ==="
    ssh-keygen -t ed25519 -N "" -f "$HOME/.ssh/id_ed25519"
    echo "=== Key generated successfully ==="
else
    echo "=== SSH identity already exists ==="
fi

hosts=(
"7.7.7.201 meat01"
"7.7.7.203 meat03"
"7.7.7.208 meat08"
"7.7.7.209 meat09"
"7.7.7.210 meat10"
"7.7.7.211 meat11"
"7.7.7.214 meat14"
"7.7.7.216 meat16"
"7.7.7.217 meat17"
"7.7.7.218 meat18"
"7.7.7.219 meat19"
"7.7.7.220 meat20"
"7.7.7.221 meat-21"
"7.7.7.222 meat22"
"7.7.7.224 meat24"
)

echo "=== Installing SSH keys with auto-password ==="

for entry in "${hosts[@]}"; do
    ip=$(echo $entry | awk '{print $1}')
    user=$(echo $entry | awk '{print $2}')

    echo "----- $ip ($user) -----"

    # Remove old fingerprint
    ssh-keygen -R "$ip" >/dev/null 2>&1

    # Push SSH key
    sshpass -p "$PASSWORD" ssh-copy-id \
        -o StrictHostKeyChecking=no \
        "$user@$ip"

    # Test login
    sshpass -p "$PASSWORD" ssh \
        -o StrictHostKeyChecking=no \
        "$user@$ip" "echo OK" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "[OK] Successfully installed key on $ip"
    else
        echo "[FAIL] Could not log into $ip"
    fi

    echo ""
done

echo "=== Done ==="
