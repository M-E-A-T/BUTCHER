#!/bin/bash
#make sure to install ssh pass 3 8 9 21 are weird

PASSWORD="meat"

hosts=(
"7.7.7.201 meat01"

"7.7.7.210 meat10"
"7.7.7.211 meat11"
"7.7.7.214 meat14"
"7.7.7.216 meat16"
"7.7.7.217 meat17"
"7.7.7.218 meat18"
"7.7.7.219 meat19"
"7.7.7.220 meat20"
"7.7.7.222 meat22"
"7.7.7.224 meat24"
)

echo "=== Installing SSH keys with auto-password ==="

for entry in "${hosts[@]}"; do
    ip=$(echo $entry | awk '{print $1}')
    user=$(echo $entry | awk '{print $2}')

    echo "----- $ip ($user) -----"

    # Remove old host fingerprint
    ssh-keygen -R "$ip" >/dev/null 2>&1

    # Automatically push your SSH key (no password prompts)
    sshpass -p "$PASSWORD" ssh-copy-id \
        -o StrictHostKeyChecking=no \
        "$user@$ip"

    # Test the connection
    sshpass -p "$PASSWORD" ssh \
        -o StrictHostKeyChecking=no \
        -o PasswordAuthentication=yes \
        -o BatchMode=no \
        "$user@$ip" "echo OK" 2>/dev/null

    if [ $? -eq 0 ]; then
        echo "[OK] Successfully installed key on $ip"
    else
        echo "[FAIL] Could not log into $ip"
    fi

    echo ""
done

echo "=== Done ==="
