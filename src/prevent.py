"""WIPS response: log every alert and add the offending MAC to a blocklist.

Enforcement is OFF by default. With enforce=True it asks a hostapd access point
that YOU own to deny the MAC. A deauth flood usually SPOOFS the AP/client MACs,
so list your own AP in `protected` and treat blocklist entries as 'claimed source'."""
import os
import subprocess
import time


class Responder:
    def __init__(self, log="results/alerts.log", blocklist="results/blocklist.txt",
                 cooldown=5.0, protected=(), enforce=False, iface=None):
        self.log, self.blocklist = log, blocklist
        self.cooldown = cooldown
        self.protected = {m.lower() for m in protected}
        self.enforce, self.iface = enforce, iface
        self.last_alert, self.blocked = {}, set()
        os.makedirs(os.path.dirname(log), exist_ok=True)
        if os.path.exists(blocklist):
            self.blocked = {ln.split()[0] for ln in open(blocklist)
                            if ln.strip() and not ln.startswith("#")}

    def handle(self, mac, t, details):
        """Returns the alert line, or None if suppressed by the cooldown."""
        mac = str(mac).lower()
        if t - self.last_alert.get(mac, -1e18) < self.cooldown:
            return None
        self.last_alert[mac] = t
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        if mac in self.protected:
            action = "logged only (protected MAC)"
        elif mac in self.blocked:
            action = "already on blocklist"
        else:
            self.blocked.add(mac)
            with open(self.blocklist, "a") as fh:
                fh.write(f"{mac} # {stamp} {details}\n")
            action = "added to blocklist"
            if self.enforce and self.iface:
                subprocess.run(["hostapd_cli", "-i", self.iface, "deny_acl", "ADD_MAC", mac],
                               check=False)
                action += " + deny_acl sent to AP"
        line = f"[{stamp}] DEAUTH FLOOD ALERT mac={mac} {details} -> {action}"
        with open(self.log, "a") as fh:
            fh.write(line + "\n")
        return line
