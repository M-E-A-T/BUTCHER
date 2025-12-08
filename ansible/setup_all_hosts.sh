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
"7.7.7.204 meat04"


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
